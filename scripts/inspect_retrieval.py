"""查看一个 query 的 RAG 检索链路。

示例：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_retrieval.py \
    --query "Skills 是什么" \
    --top-k 5

  PYTHONPATH=. ./.venv/bin/python scripts/inspect_retrieval.py \
    --query "WAI-ARIA 和虚拟列表有什么区别？" \
    --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.knowledge.search_inspector import inspect_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 RAG 检索链路解释。")
    parser.add_argument("--query", required=True, help="要检查的 query。")
    parser.add_argument("--top-k", type=int, default=8, help="每个阶段展示条数。")
    parser.add_argument(
        "--context-preview-chars",
        type=int,
        default=360,
        help="最终 context preview 字符数。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    return parser.parse_args()


def _print_hits(title: str, hits: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not hits:
        print("(none)")
        return

    for index, hit in enumerate(hits, start=1):
        print(
            f"{index}. {hit['id']} doc={hit['doc_id']} "
            f"score={hit['score']} sem={hit['semantic_score']} "
            f"lex={hit['keyword_score_norm']} sources={hit['retrieval_sources']}"
        )
        if hit.get("section_title"):
            print(f"   section: {hit['section_title']}")
        print(f"   preview: {hit['preview']}")


def print_text_report(report: dict) -> None:
    print(f"query: {report['query']}")
    print(f"query_type: {report['query_type']}")
    print(f"classification: {report['query_classification']}")
    print(f"timings_ms: {report['timings_ms']}")
    print(f"retrieval_debug: {report['retrieval_debug']}")
    print(f"errors: {report['errors'] or '-'}")

    _print_hits("Dense Hits", report["dense_hits"])
    _print_hits("Lexical Hits", report["lexical_hits"])
    _print_hits("Hybrid Hits", report["hybrid_hits"])
    _print_hits("Returned Hits", report["returned_hits"])
    _print_hits("Filtered Hits", report["filtered_hits"])
    _print_hits("Reranked Hits", report["reranked_hits"])
    _print_hits("Merged Hits", report["merged_hits"])

    print("\nCitations")
    print("---------")
    for citation in report["citations"]:
        print(
            f"{citation['ref']} doc={citation['doc_id']} "
            f"chunk={citation['chunk_id']} source={citation['source']}"
        )

    print("\nContext Preview")
    print("---------------")
    print(report["context_preview"] or "(empty)")


def main() -> None:
    args = parse_args()
    report = inspect_retrieval(
        args.query,
        top_k=args.top_k,
        context_preview_chars=args.context_preview_chars,
    )
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_text_report(payload)


if __name__ == "__main__":
    main()
