# python -m scripts.build_doc_chroma_index
"""兼容旧入口：实际构建逻辑已迁移到 build_knowledge_index。"""

from scripts.build_knowledge_index import (
    build_doc_chunks,
    infer_doc_title,
    load_docs,
    main,
)

__all__ = ["build_doc_chunks", "infer_doc_title", "load_docs", "main"]


if __name__ == "__main__":
    main()
