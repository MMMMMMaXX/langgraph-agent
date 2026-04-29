# python -m scripts.build_knowledge_index
"""离线构建知识库索引。

这个脚本同时写两套索引：
1. SQLite knowledge catalog + FTS5：保存文档/chunk 真相源和 lexical retrieval
2. Chroma docs collection：保存 chunk embedding，服务 dense retrieval
"""

from __future__ import annotations

import json
from pathlib import Path

from app.chunking import chunk_document_text
from app.config import VECTOR_STORE_CONFIG
from app.knowledge import KnowledgeCatalog, KnowledgeChunkRecord
from app.llm import embed_texts
from app.vector_store import ChromaVectorStore

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db.json"
BATCH_SIZE = 16


def load_docs(path: Path = DB_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_doc_title(doc: dict, doc_id: str, content: str) -> str:
    """从原始文档里尽量提取可读标题。"""

    explicit_title = str(doc.get("title") or doc.get("name") or "").strip()
    if explicit_title:
        return explicit_title
    preview = content.strip().replace("\n", " ")[:32]
    return preview or f"doc-{doc_id}"


def infer_source(doc: dict, default_path: Path = DB_PATH) -> str:
    return str(doc.get("source") or doc.get("path") or default_path.name)


def infer_source_type(doc: dict, source: str) -> str:
    explicit = str(doc.get("source_type") or "").strip()
    if explicit:
        return explicit
    return Path(source).suffix.lower().lstrip(".") or "txt"


def build_doc_chunks(docs: list[dict]) -> list[dict]:
    """把原始文档转换成可写入 SQLite/Chroma 的 chunk 记录。"""

    chunk_records: list[dict] = []

    for doc_index, doc in enumerate(docs):
        doc_id = str(doc.get("id", doc_index))
        content = str(doc.get("content", ""))
        doc_title = infer_doc_title(doc, doc_id, content)
        source = infer_source(doc)
        source_type = infer_source_type(doc, source)
        chunks = chunk_document_text(
            doc_id=doc_id,
            text=content,
            source_type=source_type,
        )

        for chunk in chunks:
            metadata = {
                "doc_id": chunk.doc_id,
                "doc_title": doc_title,
                "source": source,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_char_len": chunk.char_len,
            }
            chunk_records.append(
                {
                    "id": chunk.chunk_id,
                    "document": chunk.text,
                    "metadata": metadata,
                    "catalog_record": KnowledgeChunkRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        doc_title=doc_title,
                        source=source,
                        section_title=chunk.section_title,
                        chunk_index=chunk.chunk_index,
                        content=chunk.text,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        chunk_char_len=chunk.char_len,
                    ),
                }
            )

    return chunk_records


def write_sqlite_catalog(docs: list[dict], chunk_records: list[dict]) -> None:
    catalog = KnowledgeCatalog()
    catalog.reset()

    for doc_index, doc in enumerate(docs):
        doc_id = str(doc.get("id", doc_index))
        content = str(doc.get("content", ""))
        catalog.upsert_document(
            doc_id=doc_id,
            title=infer_doc_title(doc, doc_id, content),
            source=infer_source(doc),
            content=content,
            source_type="json",
            metadata={key: value for key, value in doc.items() if key != "content"},
        )

    catalog.replace_chunks(item["catalog_record"] for item in chunk_records)


def write_chroma_index(chunk_records: list[dict]) -> None:
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


def main() -> None:
    docs = load_docs()
    chunk_records = build_doc_chunks(docs)
    write_sqlite_catalog(docs, chunk_records)
    write_chroma_index(chunk_records)
    print(
        "knowledge index built. "
        f"chroma_collection={VECTOR_STORE_CONFIG.doc_collection_name}, "
        f"chunks={len(chunk_records)}"
    )


if __name__ == "__main__":
    main()
