from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from types import SimpleNamespace as config
from openai import OpenAI
import os
import json
import logging
import time
import asyncio
import re
import PyPDF2
import copy
import yaml
import pymupdf

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")


def get_openai_client():
    """Initialize and return OpenAI client with OpenRouter support"""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://api.tokenfactory.nebius.com/v1/") if os.getenv("OPENROUTER_API_KEY") else None
    
    if not api_key:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or CHATGPT_API_KEY")
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def count_tokens(text, model=None):
    if not text:
        return 0
    # Rough estimation: ~4 characters per token
    return len(text) // 4


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    if model:
        model = model.removeprefix("litellm/")
    
    client = get_openai_client()
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""


async def llm_acompletion(model, prompt):
    if model:
        model = model.removeprefix("litellm/")
    
    client = get_openai_client()
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return ""


def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        end_idx = response.find("```", start_idx)
        if end_idx != -1:
            return response[start_idx:end_idx].strip()
    
    end_idx = response.rfind("```")
    if end_idx != -1:
        return response[:end_idx].strip()
    
    json_content = response.strip()
    return json_content


def extract_json(content):
    try:
        json_content = get_json_content(content)
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
        return {}


def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = node_id
        node_id += 1
        if 'nodes' in data:
            for node in data['nodes']:
                node_id = write_node_id(node, node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id


def get_nodes(structure):
    if isinstance(structure, dict):
        return structure.get('nodes', [])
    elif isinstance(structure, list):
        return structure
    return []


def structure_to_list(structure):
    if isinstance(structure, dict):
        return [structure]
    elif isinstance(structure, list):
        return structure
    return []


def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if 'nodes' not in structure or not structure['nodes']:
            return [structure]
        leaf_nodes = []
        for node in structure['nodes']:
            leaf_nodes.extend(get_leaf_nodes(node))
        return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []


def is_leaf_node(data, node_id):
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            if 'nodes' in data:
                for node in data['nodes']:
                    result = find_node(node, node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    node = find_node(data, node_id)
    if node and not node.get('nodes'):
        return True
    return False


def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title


def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text):
    start_page = -1
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    return filename.replace('/', replacement)


def get_pdf_name(pdf_path):
    if isinstance(pdf_path, str):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    elif isinstance(pdf_path, BytesIO):
        pdf_name = 'document'
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

    def log(self, level, message, **kwargs):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def info(self, message, **kwargs):
        self.log('INFO', message, **kwargs)

    def error(self, message, **kwargs):
        self.log('ERROR', message, **kwargs)

    def debug(self, message, **kwargs):
        self.log('DEBUG', message, **kwargs)

    def exception(self, message, **kwargs):
        self.log('EXCEPTION', message, **kwargs)

    def _filepath(self):
        return self.file_path


def list_to_tree(data):
    def get_parent_structure(structure):
        if isinstance(structure, dict):
            return structure.get('parent_structure', None)
        return None

    nodes = {}
    root_nodes = []

    for item in data:
        item_id = item.get('list_index')
        parent_id = item.get('parent_structure')
        nodes[item_id] = {'item': item, 'children': []}

        if parent_id is None:
            root_nodes.append(item_id)

    for item_id, node in nodes.items():
        parent_id = node['item'].get('parent_structure')
        if parent_id is not None and parent_id in nodes:
            nodes[parent_id]['children'].append(item_id)

    def build_tree(item_id):
        node = nodes[item_id]['item'].copy()
        children_ids = nodes[item_id]['children']
        if children_ids:
            node['nodes'] = [build_tree(child_id) for child_id in children_ids]
        return node

    def clean_node(node):
        if 'nodes' in node and not node['nodes']:
            del node['nodes']
        elif 'nodes' in node:
            node['nodes'] = [clean_node(n) for n in node['nodes']]
        return node

    return [clean_node(build_tree(item_id)) for item_id in root_nodes]


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface = {
            'title': 'Preface',
            'physical_index': 1,
            'list_index': -1,
            'parent_structure': None,
            'level': 1
        }
        data.insert(0, preface)
    return data


def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        pages = [page.extract_text() for page in pdf_reader.pages]
    elif pdf_parser == "PyMuPDF":
        doc = pymupdf.open(pdf_path)
        pages = [page.get_text() for page in doc]
    else:
        raise ValueError(f"Unknown PDF parser: {pdf_parser}")

    token_lengths = [count_tokens(page, model) for page in pages]
    
    # Return as list of tuples (page_text, token_count)
    # Ensure token_count is always an integer
    return [(page, int(tokens)) for page, tokens in zip(pages, token_lengths)]


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<start_index_{page_num+1}>\n{pdf_pages[page_num]}\n"
    return text


def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return len(pdf_reader.pages)


def post_processing(structure, end_physical_index):
    """Convert structure items to tree format with start_index and end_index"""
    result = []
    for i, item in enumerate(structure):
        node = {
            'title': item.get('title'),
            'start_index': item.get('physical_index'),
        }
        
        # Set end_index based on next item's start_index or document end
        if i < len(structure) - 1:
            node['end_index'] = structure[i + 1].get('physical_index', end_physical_index)
        else:
            node['end_index'] = end_physical_index
        
        # Copy other fields like 'structure', 'node_id', etc if they exist
        for key in item:
            if key not in ['title', 'physical_index', 'page']:
                node[key] = item[key]
        
        result.append(node)
    
    return result

def clean_structure_post(data):
    if isinstance(data, dict):
        return {k: clean_structure_post(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [clean_structure_post(item) for item in data]
    return data


def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def print_toc(tree, indent=0):
    if isinstance(tree, dict):
        print(' ' * indent + f"- {tree.get('title', 'Untitled')}")
        if 'nodes' in tree:
            for node in tree['nodes']:
                print_toc(node, indent + 2)
    elif isinstance(tree, list):
        for item in tree:
            print_toc(item, indent)


def print_json(data, max_len=40, indent=2):
    def truncate_str(s, max_len):
        return s[:max_len] + '...' if len(s) > max_len else s

    def print_obj(obj, depth=0):
        prefix = ' ' * (depth * indent)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    print(f"{prefix}{k}:")
                    print_obj(v, depth + 1)
                else:
                    print(f"{prefix}{k}: {truncate_str(str(v), max_len)}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                print(f"{prefix}[{i}]:")
                print_obj(item, depth + 1)

    print_obj(data)


def remove_structure_text(data):
    if isinstance(data, dict):
        return {k: remove_structure_text(v) for k, v in data.items() if k != 'text'}
    elif isinstance(data, list):
        return [remove_structure_text(item) for item in data]
    return data


def check_token_limit(structure, limit=110000):
    def count_tokens_recursive(data):
        if isinstance(data, dict):
            total = 0
            if 'text' in data:
                total += count_tokens(data['text'])
            if 'nodes' in data:
                for node in data['nodes']:
                    total += count_tokens_recursive(node)
            return total
        elif isinstance(data, list):
            return sum(count_tokens_recursive(item) for item in data)
        return 0

    return count_tokens_recursive(structure) <= limit


def convert_physical_index_to_int(data):
    """Convert physical_index from string format to integer"""
    if isinstance(data, dict):
        if 'physical_index' in data:
            idx = data['physical_index']
            if isinstance(idx, str):
                # Handle format like '<physical_index_1>' or just '1'
                if idx.startswith('<physical_index_'):
                    # Extract number from '<physical_index_N>'
                    idx = idx.replace('<physical_index_', '').replace('>', '')
                try:
                    data['physical_index'] = int(idx)
                except (ValueError, TypeError):
                    # If conversion fails, remove the field
                    data['physical_index'] = None
        if 'nodes' in data:
            for node in data['nodes']:
                convert_physical_index_to_int(node)
    elif isinstance(data, list):
        for item in data:
            convert_physical_index_to_int(item)
    return data

def convert_page_to_int(data):
    """Convert page from string format to integer"""
    if isinstance(data, dict):
        if 'page' in data:
            page = data['page']
            if isinstance(page, str):
                # Handle format like '<page_1>' or just '1'
                if page.startswith('<page_'):
                    page = page.replace('<page_', '').replace('>', '')
                try:
                    data['page'] = int(page)
                except (ValueError, TypeError):
                    data['page'] = None
        if 'nodes' in data:
            for node in data['nodes']:
                convert_page_to_int(node)
    elif isinstance(data, list):
        for item in data:
            convert_page_to_int(item)
    
    return data

def add_node_text(node, pdf_pages):
    if 'physical_index' in node and 'text' not in node:
        start_idx = node['physical_index']
        end_idx = node.get('physical_index_end', start_idx)
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_idx, end_idx)
    if 'nodes' in node:
        for child in node['nodes']:
            add_node_text(child, pdf_pages)


def add_node_text(node, pdf_pages):
    if 'start_index' in node and 'text' not in node:
        start_idx = node['start_index']
        end_idx = node.get('end_index', start_idx)
        # Extract text from page tuples
        page_texts = []
        for page_num in range(start_idx - 1, end_idx):
            if page_num >= 0 and page_num < len(pdf_pages):
                page_content = pdf_pages[page_num][0] if isinstance(pdf_pages[page_num], tuple) else pdf_pages[page_num]
                page_texts.append(page_content)
        node['text'] = ''.join(page_texts)
    if 'nodes' in node:
        for child in node['nodes']:
            add_node_text(child, pdf_pages)


def add_node_text_with_labels(node, pdf_pages):
    if 'start_index' in node and 'text' not in node:
        start_idx = node['start_index']
        end_idx = node.get('end_index', start_idx)
        # Extract text from page tuples with labels
        page_texts = []
        for page_num in range(start_idx - 1, end_idx):
            if page_num >= 0 and page_num < len(pdf_pages):
                page_content = pdf_pages[page_num][0] if isinstance(pdf_pages[page_num], tuple) else pdf_pages[page_num]
                page_texts.append(f"<start_index_{page_num+1}>\n{page_content}\n")
        node['text'] = ''.join(page_texts)
    if 'nodes' in node:
        for child in node['nodes']:
            add_node_text_with_labels(child, pdf_pages)

async def generate_node_summary(node, model=None):
    if 'text' not in node:
        return
    
    prompt = f"""Summarize the following text in 2-3 sentences:
    
{node['text'][:2000]}

Return only the summary, no additional text."""
    
    summary = await llm_acompletion(model=model, prompt=prompt)
    node['summary'] = summary


async def generate_summaries_for_structure(structure, model=None):
    if isinstance(structure, dict):
        await generate_node_summary(structure, model)
        if 'nodes' in structure:
            for node in structure['nodes']:
                await generate_summaries_for_structure(node, model)
    elif isinstance(structure, list):
        for item in structure:
            await generate_summaries_for_structure(item, model)


def create_clean_structure_for_description(structure):
    if isinstance(structure, dict):
        clean = {'title': structure.get('title', 'Untitled')}
        if 'nodes' in structure:
            clean['nodes'] = [create_clean_structure_for_description(node) for node in structure['nodes']]
        return clean
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    return {}


def generate_doc_description(structure, model=None):
    clean_structure = create_clean_structure_for_description(structure)
    prompt = f"""Based on this document structure, generate a 2-3 sentence description of what the document is about:

{json.dumps(clean_structure, indent=2)[:2000]}

Return only the description, no additional text."""
    
    description = llm_completion(model=model, prompt=prompt)
    return description


def reorder_dict(data, key_order):
    if isinstance(data, dict):
        return {k: reorder_dict(data[k], key_order) for k in key_order if k in data}
    elif isinstance(data, list):
        return [reorder_dict(item, key_order) for item in data]
    return data


def format_structure(structure, order=None):
    if order is None:
        order = ['title', 'physical_index', 'node_id', 'summary', 'nodes']
    return reorder_dict(structure, order)


class ConfigLoader:
    def __init__(self, config_path='pageindex/config.yaml'):
        self.config_path = config_path
        self.default_config = self._load_default_config()

    def _load_default_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def load(self, user_config=None):
        if user_config is None:
            user_config = {}
        
        config_dict = {**self.default_config, **user_config}
        return config(**config_dict)


def create_node_mapping(tree):
    mapping = {}
    
    def traverse(node, path=''):
        if isinstance(node, dict):
            node_id = node.get('node_id')
            if node_id is not None:
                mapping[node_id] = path or node.get('title', 'root')
            if 'nodes' in node:
                for i, child in enumerate(node['nodes']):
                    traverse(child, f"{path}/{i}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                traverse(item, f"{path}/{i}")
    
    traverse(tree)
    return mapping


def print_tree(tree, indent=0):
    if isinstance(tree, dict):
        print('  ' * indent + f"+ {tree.get('title', 'Untitled')}")
        if 'nodes' in tree:
            for node in tree['nodes']:
                print_tree(node, indent + 1)
    elif isinstance(tree, list):
        for item in tree:
            print_tree(item, indent)


def print_wrapped(text, width=100):
    for line in text.split('\n'):
        print('\n'.join([line[i:i+width] for i in range(0, len(line), width)]))