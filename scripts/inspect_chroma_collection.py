"""查看 Chroma collection 中的原始记录结构。

常用示例：
- 查看 docs collection 前 5 条：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_chroma_collection.py --collection docs --limit 5
- 查看指定 chunk：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_chroma_collection.py --collection docs --ids 0::chunk::1
- 输出 JSON，方便脚本二次处理：
  PYTHONPATH=. ./.venv/bin/python scripts/inspect_chroma_collection.py --collection memory --limit 2 --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from app.config import VECTOR_STORE_CONFIG
from app.utils.logger import preview
from app.vector_store.chroma_client import get_chroma_client

DEFAULT_COLLECTION = VECTOR_STORE_CONFIG.doc_collection_name
DEFAULT_LIMIT = 5
DEFAULT_PREVIEW_CHARS = 120
CHROMA_GET_INCLUDE = ["documents", "metadatas"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看 Chroma collection 的 id、metadata 和 document preview。"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"collection 名称，默认 {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"最多展示多少条记录，默认 {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="从第几条开始读取，默认 0。",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="按 Chroma id 精确读取；指定后会忽略 offset。",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help=f"document preview 长度，默认 {DEFAULT_PREVIEW_CHARS}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出，方便和 jq 或脚本组合。",
    )
    return parser.parse_args()


def collection_exists(collection_name: str) -> bool:
    client = get_chroma_client()
    return any(item.name == collection_name for item in client.list_collections())


def load_records(
    *,
    collection_name: str,
    ids: list[str] | None,
    limit: int,
    offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    total_count = collection.count()

    if ids:
        raw = collection.get(ids=ids, include=CHROMA_GET_INCLUDE)
    else:
        raw = collection.get(
            limit=max(limit, 0),
            offset=max(offset, 0),
            include=CHROMA_GET_INCLUDE,
        )

    raw_ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    records: list[dict[str, Any]] = []

    for index, item_id in enumerate(raw_ids):
        records.append(
            {
                "id": item_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
        )

    return total_count, records


def build_output(
    records: list[dict[str, Any]], preview_chars: int
) -> list[dict[str, Any]]:
    output = []

    for record in records:
        document = record["document"]
        output.append(
            {
                "id": record["id"],
                "metadata": record["metadata"],
                "document_len": len(document),
                "preview": preview(document, preview_chars),
            }
        )

    return output


def print_text_output(
    *,
    collection_name: str,
    total_count: int,
    records: list[dict[str, Any]],
) -> None:
    print(f"collection: {collection_name}")
    print(f"total_count: {total_count}")
    print(f"returned_count: {len(records)}")

    for index, record in enumerate(records, start=1):
        print("\n--- record", index, "---")
        print("id:", record["id"])
        print("metadata:", json.dumps(record["metadata"], ensure_ascii=False))
        print("document_len:", record["document_len"])
        print("preview:", record["preview"])


def main() -> None:
    args = parse_args()
    collection_name = args.collection

    if not collection_exists(collection_name):
        raise SystemExit(f"collection 不存在：{collection_name}")

    total_count, raw_records = load_records(
        collection_name=collection_name,
        ids=args.ids,
        limit=args.limit,
        offset=args.offset,
    )
    records = build_output(raw_records, args.preview_chars)

    if args.json:
        print(
            json.dumps(
                {
                    "collection": collection_name,
                    "total_count": total_count,
                    "returned_count": len(records),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print_text_output(
        collection_name=collection_name,
        total_count=total_count,
        records=records,
    )


if __name__ == "__main__":
    main()
