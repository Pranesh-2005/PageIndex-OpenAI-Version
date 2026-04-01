"""
example_rag.py — Minimal end-to-end PageIndex RAG example.

Steps:
  1. Build the tree index from a PDF
  2. Save it to disk
  3. Load it back
  4. Ask questions

Usage:
    ANTHROPIC_API_KEY=sk-... python example_rag.py --pdf /path/to/doc.pdf
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pageindex import PageIndexBuilder, PageIndexTree, ask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--tree", default=None, help="Pre-built tree JSON (skip indexing)")
    parser.add_argument("--model", default="z-ai/glm-4.5-air:free")
    args = parser.parse_args()

    # ── Build or load tree ────────────────────────────────────────────────────
    if args.tree:
        print(f"Loading existing tree from {args.tree} …")
        tree = PageIndexTree.load(args.tree)
        # Re-attach pages so retrieval can return full text
        from pageindex.utils import extract_pages_from_pdf
        tree.pages = extract_pages_from_pdf(args.pdf)
    else:
        builder = PageIndexBuilder(model=args.model)
        tree = builder.build(args.pdf)

        tree_path = Path(args.pdf).stem + "_tree.json"
        tree.save(tree_path)
        print(f"\nTree saved to {tree_path}. Use --tree {tree_path} to skip indexing next time.\n")

    # ── Interactive Q&A ───────────────────────────────────────────────────────
    print("\nPageIndex RAG ready. Type your questions (Ctrl+C to quit).\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        result = ask(question, tree, model=args.model)
        print("\n── Answer ────────────────────────────────────────────────")
        print(result["answer"])
        print("\n── Retrieved Sections ────────────────────────────────────")
        for n in result["retrieved_nodes"]:
            print(f"  [{n['node_id']}] {n['title']}  (pages {n['pages']})")
        print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
