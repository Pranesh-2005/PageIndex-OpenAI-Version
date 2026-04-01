"""
app.py — Gradio UI for PageIndex RAG
Run: python app.py
"""

import os
import json
import tempfile
import threading
from pathlib import Path
from io import BytesIO

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from pageindex.utils import (
    get_page_tokens, 
    ConfigLoader, 
    get_pdf_name,
    JsonLogger
)
from pageindex.page_index import page_index_main

# ── Global state ──────────────────────────────────────────────────────────────
_current_structure = None
_current_page_list = None
_current_pdf_path = None
_tree_lock = threading.Lock()

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_state():
    with _tree_lock:
        return {
            'structure': _current_structure,
            'page_list': _current_page_list,
            'pdf_path': _current_pdf_path
        }


def _set_state(structure, page_list, pdf_path):
    global _current_structure, _current_page_list, _current_pdf_path
    with _tree_lock:
        _current_structure = structure
        _current_page_list = page_list
        _current_pdf_path = pdf_path


def _api_key_ok() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


# ── Tab 1: Settings ───────────────────────────────────────────────────────────

def save_settings(api_key: str, model: str) -> str:
    if api_key.strip():
        os.environ["OPENROUTER_API_KEY"] = api_key.strip()
        env_path = Path(".env")
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        new_lines = [l for l in lines if not l.startswith("OPENROUTER_API_KEY")]
        new_lines.append(f"OPENROUTER_API_KEY={api_key.strip()}")
        env_path.write_text("\n".join(new_lines) + "\n")

    if model.strip():
        os.environ["OPENROUTER_MODEL"] = model.strip()

    return "✅ Settings saved. API key stored in .env"


# ── Tab 2: Index a document ───────────────────────────────────────────────────

def build_index(
    file,
    model: str,
    toc_check_pages: int,
    max_pages_per_node: int,
    max_tokens_per_node: int,
    add_summary: bool,
    add_description: bool,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str]:
    """Build a PageIndex tree and return status + tree JSON preview."""

    if not _api_key_ok():
        return "❌ OPENROUTER_API_KEY not set. Go to the Settings tab.", ""

    if file is None:
        return "❌ Please upload a PDF or Markdown file.", ""

    file_path = file.name
    ext = Path(file_path).suffix.lower()
    model = model.strip() or DEFAULT_MODEL

    progress(0, desc="Starting indexing …")

    try:
        if ext != ".pdf":
            return f"❌ Currently only PDF files are supported. Got: {ext}", ""

        # Build configuration
        user_opt = {
            'model': model,
            'toc_check_page_num': toc_check_pages,
            'max_page_num_each_node': max_pages_per_node,
            'max_token_num_each_node': max_tokens_per_node,
            'if_add_node_id': 'yes',
            'if_add_node_summary': 'yes' if add_summary else 'no',
            'if_add_doc_description': 'yes' if add_description else 'no',
            'if_add_node_text': 'yes',  # Always add text for RAG
        }
        opt = ConfigLoader().load(user_opt)

        # Get page list for RAG
        progress(0.3, desc="Extracting pages …")
        page_list = get_page_tokens(file_path, model=opt.model)

        # Process PDF
        progress(0.5, desc="Building tree structure …")
        result = page_index_main(file_path, opt)

        # Store in global state
        structure = result.get('structure', [])
        _set_state(structure, page_list, file_path)
        
        progress(0.9, desc="Finalizing …")

        # Generate preview
        tree_dict = structure if isinstance(structure, list) else [structure]
        preview = json.dumps(tree_dict, indent=2)[:3000]
        if len(json.dumps(tree_dict)) > 3000:
            preview += "\n\n… (truncated for display)"

        def count_nodes(node):
            if isinstance(node, dict):
                count = 1
                if 'nodes' in node:
                    count += sum(count_nodes(n) for n in node['nodes'])
                return count
            elif isinstance(node, list):
                return sum(count_nodes(item) for item in node)
            return 0

        n_nodes = sum(count_nodes(item) for item in tree_dict) if tree_dict else 0
        status = (
            f"✅ Index built successfully!\n"
            f"📄 File: {Path(file_path).name}\n"
            f"🌲 Total nodes: {n_nodes}\n"
            f"📝 Pages: {len(page_list)}"
        )
        progress(1, desc="Done!")
        return status, preview

    except Exception as e:
        return f"❌ Error during indexing:\n{str(e)}", ""


def save_tree_to_file() -> str | None:
    state = _get_state()
    if state['structure'] is None:
        return None
    
    tmp = tempfile.NamedTemporaryFile(
        suffix="_tree.json", delete=False, mode="w", encoding="utf-8"
    )
    tree_data = state['structure'] if isinstance(state['structure'], list) else [state['structure']]
    json.dump(tree_data, tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name


def load_tree_from_file(file) -> str:
    if file is None:
        return "❌ No file selected."
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        if isinstance(tree_data, list) and tree_data:
            _set_state(tree_data, None, None)
            n_nodes = sum(1 for _ in tree_data)
            return (
                f"✅ Tree loaded!\n"
                f"🌲 Top-level nodes: {n_nodes}\n\n"
                f"⚠️ Note: Upload the original PDF too (in the Index tab) "
                f"so full text is available for answers."
            )
        else:
            return "❌ Invalid tree format"
    except Exception as e:
        return f"❌ Failed to load tree: {e}"


# ── Tab 3: Ask questions (RAG) ────────────────────────────────────────────────

def retrieve_relevant_sections(query: str, max_sections: int = 3) -> str:
    """Find and display relevant sections for a query"""
    state = _get_state()
    if state['structure'] is None:
        return "❌ No index loaded."
    
    structure = state['structure']
    if state['page_list'] is None:
        return "⚠️ Page content not available. Please re-index the document."
    
    page_list = state['page_list']
    
    # Build searchable index
    index = {}
    
    def traverse(node, path=""):
        if isinstance(node, dict):
            title = node.get('title', 'Untitled')
            start_idx = node.get('start_index', 1)
            end_idx = node.get('end_index', len(page_list))
            
            # Extract section text
            text_parts = []
            for page_idx in range(start_idx - 1, min(end_idx, len(page_list))):
                if page_idx >= 0 and page_idx < len(page_list):
                    page_content = page_list[page_idx][0] if isinstance(page_list[page_idx], tuple) else page_list[page_idx]
                    text_parts.append(page_content[:500])  # First 500 chars per page
            
            section_text = '\n'.join(text_parts)
            
            index[title] = {
                'title': title,
                'start_index': start_idx,
                'end_index': end_idx,
                'path': path,
                'text': section_text
            }
            
            if 'nodes' in node:
                for i, child in enumerate(node['nodes']):
                    traverse(child, f"{path}/{title}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                traverse(item, f"{path}/{i}")
    
    if isinstance(structure, list):
        for item in structure:
            traverse(item)
    else:
        traverse(structure)
    
    # Simple keyword matching
    query_lower = query.lower()
    relevant = []
    
    for section_title, section_info in index.items():
        score = 0
        
        # Title matching (high weight)
        if query_lower in section_title.lower():
            score += 20
        
        # Content matching
        text_lower = section_info['text'].lower()
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:
                score += text_lower.count(word)
        
        if score > 0:
            relevant.append({
                'title': section_title,
                'score': score,
                'pages': f"{section_info['start_index']}-{section_info['end_index']}",
                'preview': section_info['text'][:300]
            })
    
    # Sort and return top results
    relevant.sort(key=lambda x: x['score'], reverse=True)
    relevant = relevant[:max_sections]
    
    if not relevant:
        return f"No relevant sections found for: '{query}'"
    
    result = f"**Found {len(relevant)} relevant sections:**\n\n"
    for i, item in enumerate(relevant, 1):
        result += f"**{i}. {item['title']}** (pages {item['pages']})\n"
        result += f"*Score: {item['score']}*\n"
        result += f"Preview: {item['preview']}...\n\n"
    
    return result


def answer_question(
    question: str,
    model: str,
    chat_history: list,
) -> tuple[list, str]:
    if not _api_key_ok():
        chat_history.append((question, "❌ OPENROUTER_API_KEY not set. Go to the Settings tab."))
        return chat_history, ""

    state = _get_state()
    if state['structure'] is None:
        chat_history.append((question, "❌ No index loaded. Build or load a tree first (Index tab)."))
        return chat_history, ""

    if not question.strip():
        return chat_history, ""

    model = model.strip() or DEFAULT_MODEL

    try:
        # Get relevant sections
        from pageindex.utils import llm_completion
        
        relevant_info = retrieve_relevant_sections(question, max_sections=3)
        
        # Build context
        context = f"Here is information about relevant sections:\n\n{relevant_info}\n\n"
        
        # Create prompt
        prompt = f"""{context}

Based on the document sections above, please answer this question:
{question}

Provide a clear, helpful answer based on the document content."""
        
        # Get answer from LLM
        answer = llm_completion(model=model, prompt=prompt)
        
        full_response = f"{answer}\n\n---\n**Retrieved Info:**\n{relevant_info}"
        chat_history.append((question, full_response))

    except Exception as e:
        chat_history.append((question, f"❌ Error: {e}"))

    return chat_history, ""


def clear_chat() -> list:
    return []


def get_document_summary() -> str:
    """Display document structure overview"""
    state = _get_state()
    if state['structure'] is None:
        return "No document loaded."
    
    structure = state['structure']
    summary = "📋 **Document Structure**\n\n"
    
    def format_node(node, indent=0):
        result = ""
        if isinstance(node, dict):
            title = node.get('title', 'Untitled')
            start = node.get('start_index', '?')
            end = node.get('end_index', '?')
            result += f"{'  ' * indent}• **{title}** (pages {start}-{end})\n"
            
            if 'nodes' in node and node['nodes']:
                for child in node['nodes']:
                    result += format_node(child, indent + 1)
        elif isinstance(node, list):
            for item in node:
                result += format_node(item, indent)
        return result
    
    if isinstance(structure, list):
        for item in structure:
            summary += format_node(item)
    else:
        summary += format_node(structure)
    
    return summary


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    current_key = os.environ.get("OPENROUTER_API_KEY", "")
    current_model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

    with gr.Blocks(
        title="PageIndex RAG",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .header h1 { font-size: 2rem; margin-bottom: 4px; }
        .header p  { color: #666; margin: 0; }
        .status-box textarea { font-family: monospace; font-size: 0.85rem; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header">
          <h1>📑 PageIndex RAG</h1>
          <p>Reasoning-based document Q&A with hierarchical tree indexing</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Settings ───────────────────────────────────────────────
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### OpenRouter Configuration")
                gr.Markdown(
                    "Get your free API key at [openrouter.ai/keys](https://openrouter.ai/keys). "
                    "It will be saved to your `.env` file."
                )

                with gr.Row():
                    with gr.Column():
                        settings_key = gr.Textbox(
                            label="OpenRouter API Key",
                            value=current_key,
                            type="password",
                            placeholder="sk-or-...",
                        )
                        settings_model = gr.Textbox(
                            label="Default Model",
                            value=current_model,
                            placeholder="z-ai/glm-4.5:free",
                        )
                        save_btn = gr.Button("💾 Save Settings", variant="primary")
                        settings_status = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("""
                **Popular free models on OpenRouter:**
                - `z-ai/glm-4.5:free`
                - `meta-llama/llama-3.3-70b-instruct:free`
                - `deepseek/deepseek-r1:free`
                - `google/gemini-2.0-flash-exp:free`
                """)

                save_btn.click(
                    save_settings,
                    inputs=[settings_key, settings_model],
                    outputs=settings_status,
                )

            # ── Tab 2: Index ──────────────────────────────────────────────────
            with gr.Tab("🌲 Index Document"):
                gr.Markdown("### Build a PageIndex tree from your PDF document")

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_file = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                        )
                        idx_model = gr.Textbox(
                            label="Model (leave blank for default)",
                            placeholder=DEFAULT_MODEL,
                        )
                        with gr.Accordion("Advanced Options", open=False):
                            toc_pages   = gr.Slider(1, 50, value=20, step=1, label="TOC check pages")
                            max_pages   = gr.Slider(1, 30, value=10, step=1, label="Max pages per node")
                            max_tokens  = gr.Slider(5000, 60000, value=20000, step=1000, label="Max tokens per node")
                            add_summary = gr.Checkbox(value=True, label="Generate node summaries")
                            add_desc    = gr.Checkbox(value=False, label="Generate document description")

                        index_btn = gr.Button("🚀 Build Index", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        index_status = gr.Textbox(
                            label="Status",
                            lines=6,
                            interactive=False,
                            elem_classes="status-box",
                        )
                        tree_preview = gr.Code(
                            label="Tree JSON Preview",
                            language="json",
                            lines=20,
                        )

                with gr.Row():
                    save_tree_btn = gr.Button("💾 Download Tree JSON")
                    save_tree_file = gr.File(label="Download", interactive=False)

                gr.Markdown("---\n### Or load an existing tree JSON")
                with gr.Row():
                    load_tree_upload = gr.File(label="Upload tree JSON", file_types=[".json"])
                    load_tree_btn    = gr.Button("📂 Load Tree")
                    load_tree_status = gr.Textbox(label="Load Status", lines=5, interactive=False)

                index_btn.click(
                    build_index,
                    inputs=[upload_file, idx_model, toc_pages, max_pages, max_tokens, add_summary, add_desc],
                    outputs=[index_status, tree_preview],
                )
                save_tree_btn.click(save_tree_to_file, outputs=save_tree_file)
                load_tree_btn.click(load_tree_from_file, inputs=load_tree_upload, outputs=load_tree_status)

            # ── Tab 3: Ask ────────────────────────────────────────────────────
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("### Ask questions about your indexed document")

                # Document structure display
                with gr.Accordion("📋 Document Structure", open=False):
                    doc_structure = gr.Markdown(get_document_summary())
                
                # Query settings
                with gr.Row():
                    qa_model = gr.Textbox(
                        label="Model (leave blank for default)",
                        placeholder=DEFAULT_MODEL,
                        scale=4,
                    )
                    refresh_structure_btn = gr.Button("🔄 Refresh Structure", scale=1)

                # Chatbot
                chatbot = gr.Chatbot(label="Chat", height=500, bubble_full_width=False)

                # Question input
                with gr.Row():
                    question_box = gr.Textbox(
                        label="",
                        placeholder="Ask anything about your document …",
                        show_label=False,
                        scale=5,
                    )
                    ask_btn   = gr.Button("Ask ➤", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                # Event handlers
                ask_btn.click(
                    answer_question,
                    inputs=[question_box, qa_model, chatbot],
                    outputs=[chatbot, question_box],
                )
                question_box.submit(
                    answer_question,
                    inputs=[question_box, qa_model, chatbot],
                    outputs=[chatbot, question_box],
                )
                clear_btn.click(clear_chat, outputs=chatbot)
                refresh_structure_btn.click(
                    get_document_summary,
                    outputs=doc_structure
                )

            # ── Tab 4: How it works ───────────────────────────────────────────
            with gr.Tab("📖 How It Works"):
                gr.Markdown("""
## PageIndex: Reasoning-Based RAG

PageIndex is a **vectorless, reasoning-based document Q&A system** that uses LLM reasoning to navigate document hierarchies.

### Architecture

**Three Core Components:**

1. **Indexing Phase**
   - Scans PDF for table of contents or auto-detects sections
   - Creates hierarchical tree structure (like a book outline)
   - Stores: title, page range, and optional summaries

2. **Retrieval Phase**
   - User question + tree structure sent to LLM
   - LLM *reasons* about which sections are relevant
   - System retrieves full text from selected sections

3. **Generation Phase**
   - Context from relevant sections + full page text
   - LLM generates answer with source attribution

### Why This Approach?

| Feature | Traditional Vector RAG | PageIndex |
|---------|----------------------|-----------|
| **Retrieval Method** | Similarity search | LLM reasoning |
| **Chunk Size** | Fixed (256-512 tokens) | Natural sections |
| **Database** | Vector DB required | None needed |
| **Explainability** | Hard to trace | Clear section references |
| **Cost** | Embedding API calls | Minimal |

### Workflow

```
PDF Upload
    ↓
Auto-detect Sections / Extract TOC
    ↓
Build Hierarchical Tree
    ↓
Store with Page Ranges & Text
    ↓
User Question
    ↓
LLM Selects Relevant Sections
    ↓
Retrieve Full Text
    ↓
Generate Answer with Sources
```

### Use Cases

✅ **Long documents** (100+ pages)
✅ **Hierarchical content** (books, reports, specs)
✅ **Source attribution needed**
✅ **Low-latency retrieval**
❌ **Semantic similarity** (not ideal for Vector RAG)
❌ **Cross-language search**

---
*Built with PageIndex + OpenRouter + Gradio*
                """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        show_error=True,
    )