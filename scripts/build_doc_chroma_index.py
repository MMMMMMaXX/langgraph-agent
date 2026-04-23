# python -m scripts.build_doc_chroma_index
"""离线构建文档 Chroma 索引。

当前阶段的明确约束：
1. embedding 在项目外部计算，这里通过 `app.llm.embed_texts()` 先算向量
2. docs 做 chunk
3. memory 先不在这个脚本里处理

因此这个脚本只负责：
- 读取原始文档 `data/db.json`
- 按 chunk 规则切分
- 批量计算 embedding
- 写入 Chroma docs collection
"""

import json
from pathlib import Path

from app.chunking import chunk_document_text
from app.config import VECTOR_STORE_CONFIG
from app.llm import embed_texts
from app.vector_store import ChromaVectorStore

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db.json"
BATCH_SIZE = 16


def load_docs() -> list[dict]:
    return json.loads(DB_PATH.read_text(encoding="utf-8"))


def infer_doc_title(doc: dict, doc_id: str, content: str) -> str:
    """从原始文档里尽量提取可读标题。

    当前 `data/db.json` 只有 content 字段，但后续真实文档通常会带 title/source。
    这里先兼容两种情况：
    - 有 title/name 时直接使用
    - 没有标题时，用正文前 32 个字符作为调试标题
    """

    explicit_title = str(doc.get("title") or doc.get("name") or "").strip()
    if explicit_title:
        return explicit_title
    preview = content.strip().replace("\n", " ")[:32]
    return preview or f"doc-{doc_id}"


def build_doc_chunks(docs: list[dict]) -> list[dict]:
    """把原始文档转换成可写入 Chroma 的 chunk 记录。"""

    chunk_records: list[dict] = []

    for doc_index, doc in enumerate(docs):
        doc_id = str(doc.get("id", doc_index))
        content = doc.get("content", "")
        doc_title = infer_doc_title(doc, doc_id, content)
        source = str(doc.get("source") or doc.get("path") or DB_PATH.name)
        chunks = chunk_document_text(doc_id=doc_id, text=content)

        for chunk in chunks:
            chunk_records.append(
                {
                    "id": chunk.chunk_id,
                    "document": chunk.text,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "doc_title": doc_title,
                        "source": source,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "chunk_char_len": chunk.char_len,
                    },
                }
            )

    return chunk_records


def main() -> None:
    docs = load_docs()
    chunk_records = build_doc_chunks(docs)

    store = ChromaVectorStore()
    store.reset_collection(
        VECTOR_STORE_CONFIG.doc_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for batch_start in range(0, len(chunk_records), BATCH_SIZE):
        batch = chunk_records[batch_start : batch_start + BATCH_SIZE]
        texts = [item["document"] for item in batch]
        embeddings = embed_texts(texts, profile="doc_embedding")

        store.upsert(
            collection_name=VECTOR_STORE_CONFIG.doc_collection_name,
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )

        print(
            f"indexed doc chunks: {batch_start + 1}-{batch_start + len(batch)} / {len(chunk_records)}"
        )

    print(
        f"chroma doc index built. collection={VECTOR_STORE_CONFIG.doc_collection_name}, chunks={len(chunk_records)}"
    )


if __name__ == "__main__":
    main()
