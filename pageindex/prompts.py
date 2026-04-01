"""
All prompts used by PageIndex for tree building and retrieval.
"""

# ── Index-building prompts ───────────────────────────────────────────────────

TOC_DETECT_SYSTEM = """\
You are an expert document analyst. Your job is to detect whether a document \
has an explicit table of contents (TOC) and, if so, extract its structure.
Respond ONLY in valid JSON — no markdown fences, no preamble.
"""

TOC_DETECT_USER = """\
Below are the first {n_pages} pages of a document. 
Determine if there is an explicit table of contents. If yes, extract the structure.

Pages:
{pages_text}

Respond with JSON:
{{
  "has_toc": true | false,
  "toc": [
    {{"title": "Section title", "page": <int or null>}},
    ...
  ]
}}
If no TOC exists, return {{"has_toc": false, "toc": []}}.
"""

# ─────────────────────────────────────────────────────────────────────────────

SEGMENT_SYSTEM = """\
You are an expert document analyst. Your task is to identify natural section \
boundaries in a portion of a document and produce a concise summary of each section.
Respond ONLY in valid JSON — no markdown fences, no preamble.
"""

SEGMENT_USER = """\
Below is a portion of a document spanning pages {start_page} to {end_page}.
Identify the natural sections (chapters, headings, topics) within this range.
For each section produce:
  - title: a short descriptive title
  - start_page: first page of the section (integer)
  - end_page: last page of the section (integer)
  - summary: 2-4 sentence summary of the section content

Document text:
{text}

Respond with JSON:
{{
  "sections": [
    {{
      "title": "...",
      "start_page": <int>,
      "end_page": <int>,
      "summary": "..."
    }},
    ...
  ]
}}
"""

# ─────────────────────────────────────────────────────────────────────────────

DOC_DESCRIPTION_SYSTEM = """\
You are an expert document analyst. Write a concise overall description of the \
document based on its table of contents / tree structure.
Respond ONLY in valid JSON — no markdown fences, no preamble.
"""

DOC_DESCRIPTION_USER = """\
Here is the tree index of a document:
{tree_json}

Write a 3-5 sentence description of the document as a whole.

Respond with JSON:
{{"description": "..."}}
"""

# ── Retrieval prompts ────────────────────────────────────────────────────────

TREE_SEARCH_SYSTEM = """\
You are an expert document analyst performing reasoning-based retrieval. \
Given a user question and a document tree index (titles + summaries), \
identify which nodes are most likely to contain the answer.
Reason step-by-step, then output the node IDs to drill into.
Respond ONLY in valid JSON — no markdown fences, no preamble.
"""

TREE_SEARCH_USER = """\
Question: {question}

Document tree index:
{tree_summary}

Which nodes should be retrieved to answer the question?
Respond with JSON:
{{
  "reasoning": "brief explanation of why these nodes are relevant",
  "node_ids": ["0001", "0003", ...]
}}
"""

# ─────────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """\
You are an expert analyst. Answer the user's question using ONLY the provided \
document excerpts. Cite section titles when relevant. Be precise and concise.
"""

ANSWER_USER = """\
Question: {question}

Retrieved document sections:
{sections_text}

Answer the question based on the above content.
"""
