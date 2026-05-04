"""预览知识库文档重新切片结果，不写索引。

示例：
  PYTHONPATH=. ./.venv/bin/python scripts/preview_rechunk.py \
    --doc-id doc-37102bfd03fd315d \
    --chunk-size 360 \
    --overlap 80 \
    --min-chars 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.config import CHUNKING_CONFIG
from app.constants.knowledge import RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT
from app.knowledge.rechunk_preview import RechunkPreviewParams, preview_rechunk_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览知识库文档重新切片结果。")
    parser.add_argument("--doc-id", required=True, help="要预览的文档 doc_id。")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNKING_CONFIG.chunk_size_chars,
        help="候选 chunk size。",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNKING_CONFIG.chunk_overlap_chars,
        help="候选 overlap。",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=CHUNKING_CONFIG.min_chunk_chars,
        help="候选最小 chunk 字符数。",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=RECHUNK_PREVIEW_DEFAULT_SAMPLE_LIMIT,
        help="展示样例 chunk 数。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    return parser.parse_args()


def print_text_report(report: dict) -> None:
    print(f"doc_id: {report['doc_id']}")
    print(f"title: {report['title']}")
    print(f"source: {report['source']}")
    print(f"source_mode: {report['source_mode']}")
    print(f"applied: {report['applied']}")
    print(f"params: {report['params']}")
    print(f"warnings: {', '.join(report['warnings']) or '-'}")

    current = report["current"]
    preview = report["preview"]
    print("\nCurrent -> Preview")
    print("------------------")
    print(f"chunk_count: {current['chunk_count']} -> {preview['chunk_count']}")
    print(f"avg_chars: {current['avg_chars']} -> {preview['avg_chars']}")
    print(f"median_chars: {current['median_chars']} -> {preview['median_chars']}")
    print(f"short_chunks: {current['short_chunk_count']} -> {preview['short_chunk_count']}")
    print(f"long_chunks: {current['long_chunk_count']} -> {preview['long_chunk_count']}")
    print(f"section_count: {current['section_count']} -> {preview['section_count']}")
    print(f"delta: {report['delta']}")

    print("\nPreview Samples")
    print("---------------")
    for sample in preview["samples"]:
        print(
            f"[{sample['chunk_index']}] {sample['chunk_id']} "
            f"len={sample['chunk_char_len']} section={sample['section_title']!r}"
        )
        print(sample["preview"])
        print("")


def main() -> None:
    args = parse_args()
    report = preview_rechunk_document(
        args.doc_id,
        params=RechunkPreviewParams(
            chunk_size_chars=args.chunk_size,
            chunk_overlap_chars=args.overlap,
            min_chunk_chars=args.min_chars,
            sample_limit=args.sample_limit,
        ),
    )
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_text_report(payload)


if __name__ == "__main__":
    main()
