"""把旧版 `data/memory_db.json` 批量迁移到 Chroma memory collection。

迁移策略：
1. 优先复用旧 JSON 中已经存在的 embedding，避免重复调用 embedding API
2. 只有当旧记录没有 embedding，或 embedding 结构异常时，才回退实时重算
3. memory 仍然按“整条 document”写入，不做 chunk
4. document id 继续沿用：
   `session_id::memory::memory_key`
   这样可以和当前线上 add_memory_item 的 upsert 语义保持一致
"""

import json
from pathlib import Path

from app.config import VECTOR_STORE_CONFIG
from app.llm import embed_text
from app.memory.vector_memory import (
    DEFAULT_MEMORY_CONFIDENCE,
    MEMORY_SCHEMA_VERSION,
    _build_memory_document_id,
    build_memory_metadata,
)
from app.utils.memory_key import MEMORY_TYPE_FACT
from app.vector_store import ChromaVectorStore

MEMORY_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory_db.json"
BATCH_SIZE = 32


def load_memory_rows() -> list[dict]:
    if not MEMORY_DB_PATH.exists():
        return []
    return json.loads(MEMORY_DB_PATH.read_text(encoding="utf-8"))


def is_valid_embedding(value: object) -> bool:
    """判断旧 JSON 里的 embedding 是否可直接复用。"""

    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, (int, float)) for item in value)


def build_migration_rows(rows: list[dict]) -> list[dict]:
    """把旧 JSON 记录转换成可写入 Chroma 的标准行。"""

    migrated: list[dict] = []

    for index, row in enumerate(rows):
        session_id = row.get("session_id", "default")
        memory_key = row.get("memory_key", f"legacy_{index}")
        content = row.get("content", "")
        tags = row.get("tags", [])
        embedding = row.get("embedding")

        if not content.strip():
            continue

        if not is_valid_embedding(embedding):
            embedding = embed_text(content, profile="memory_embedding")

        migrated.append(
            {
                "id": _build_memory_document_id(session_id, memory_key),
                "document": content,
                "embedding": embedding,
                "metadata": {
                    **build_memory_metadata(
                        session_id=session_id,
                        source=row.get("source", "conversation"),
                        source_route=row.get("source_route", ""),
                        timestamp=row.get("timestamp", 0),
                        tags=tags,
                        memory_key=memory_key,
                        memory_type=row.get("memory_type", MEMORY_TYPE_FACT),
                        rewritten_query=row.get("rewritten_query", ""),
                        confidence=float(
                            row.get("confidence", DEFAULT_MEMORY_CONFIDENCE)
                        ),
                        is_active=bool(row.get("is_active", True)),
                    ),
                    "schema_version": int(
                        row.get("schema_version", MEMORY_SCHEMA_VERSION)
                    ),
                },
            }
        )

    return migrated


def main() -> None:
    rows = load_memory_rows()
    migrated_rows = build_migration_rows(rows)
    store = ChromaVectorStore()

    # 迁移脚本默认直接重建 memory collection。
    # 这是“全量导入旧 JSON 历史数据”的语义，比增量追加更直观。
    store.reset_collection(
        VECTOR_STORE_CONFIG.memory_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for batch_start in range(0, len(migrated_rows), BATCH_SIZE):
        batch = migrated_rows[batch_start : batch_start + BATCH_SIZE]
        store.upsert(
            collection_name=VECTOR_STORE_CONFIG.memory_collection_name,
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        print(
            f"migrated memory rows: {batch_start + 1}-{batch_start + len(batch)} / {len(migrated_rows)}"
        )

    print(
        f"memory migration complete. collection={VECTOR_STORE_CONFIG.memory_collection_name}, rows={len(migrated_rows)}"
    )


if __name__ == "__main__":
    main()
