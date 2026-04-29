"""查看知识库文档的 chunk 质量统计。

示例：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_knowledge_chunks.py \
    --doc-id doc-37102bfd03fd315d

  PYTHONPATH=. ./.venv/bin/python scripts/inspect_knowledge_chunks.py \
    --doc-id doc-37102bfd03fd315d --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.knowledge.chunk_inspector import (
    ChunkQualityThresholds,
    inspect_document_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看知识库文档 chunk 质量。")
    parser.add_argument("--doc-id", required=True, help="要检查的文档 doc_id。")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="展示前几个 chunk 样例，默认 5。",
    )
    parser.add_argument(
        "--short-chars",
        type=int,
        default=None,
        help="短 chunk 阈值，默认使用 CHUNKING_CONFIG.min_chunk_chars。",
    )
    parser.add_argument(
        "--long-chars",
        type=int,
        default=None,
        help="长 chunk 阈值，默认使用 chunk_size_chars * 2。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    return parser.parse_args()


def build_thresholds(args: argparse.Namespace) -> ChunkQualityThresholds | None:
    if args.short_chars is None and args.long_chars is None:
        return None
    defaults = ChunkQualityThresholds()
    return ChunkQualityThresholds(
        short_chars=args.short_chars or defaults.short_chars,
        long_chars=args.long_chars or defaults.long_chars,
    )


def print_text_report(report: dict) -> None:
    print(f"doc_id: {report['doc_id']}")
    print(f"chunk_count: {report['chunk_count']}")
    print(f"total_chars: {report['total_chars']}")
    print(f"min_chars: {report['min_chars']}")
    print(f"max_chars: {report['max_chars']}")
    print(f"avg_chars: {report['avg_chars']}")
    print(f"median_chars: {report['median_chars']}")
    print(f"short_chunk_count: {report['short_chunk_count']}")
    print(f"long_chunk_count: {report['long_chunk_count']}")
    print(f"section_count: {report['section_count']}")
    print(f"warnings: {', '.join(report['warnings']) or '-'}")

    print("\nTop sections")
    print("------------")
    for section in report["top_sections"]:
        print(f"{section['chunk_count']:>4}  {section['section_title']}")

    print("\nSamples")
    print("-------")
    for sample in report["samples"]:
        print(
            f"[{sample['chunk_index']}] {sample['chunk_id']} "
            f"len={sample['chunk_char_len']} section={sample['section_title']!r}"
        )
        print(sample["preview"])
        print("")


def main() -> None:
    args = parse_args()
    report = inspect_document_chunks(
        args.doc_id,
        sample_limit=args.sample_limit,
        thresholds=build_thresholds(args),
    )
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_text_report(payload)


if __name__ == "__main__":
    main()
