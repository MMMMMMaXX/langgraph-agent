from typing import Any

from app.vector_store.chroma_client import get_chroma_client


class ChromaVectorStore:
    """对 Chroma 做一层项目内抽象。

    设计目标：
    - 业务层不直接依赖 chromadb SDK 细节
    - embedding 继续由项目外部计算后再写入
    - docs / memory 共用一套基础接口
    """

    def __init__(self) -> None:
        self._client = get_chroma_client()

    def get_or_create_collection(self, name: str, metadata: dict | None = None) -> Any:
        # 获取或创建 collection，目前有 docs 和 memory 两类集合。
        return self._client.get_or_create_collection(name=name, metadata=metadata)

    def reset_collection(self, name: str, metadata: dict | None = None) -> None:
        try:
            # 重建索引时先删除旧 collection，再重新创建。用于离线构建 doc 索引和迁移 memory。
            self._client.delete_collection(name=name)
        except Exception:
            # collection 首次不存在是正常情况，不需要把重建流程打断。
            pass
        self._client.get_or_create_collection(name=name, metadata=metadata)

    # 写入或覆盖向量记录。docs 写 chunk；memory 写整条记忆。
    def upsert(
        self,
        *,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        collection = self.get_or_create_collection(
            collection_name,
            # collection 使用 cosine 距离，后续把 distance 转成项目内部的相似度分数。
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # 按 query embedding 做向量召回，返回 documents、metadatas、distances。
    def query(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> dict:
        collection = self.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    # 非向量查询读取记录，比如按 session_id 取最近 memory，或构建全局 memory index。
    def get(
        self,
        *,
        collection_name: str,
        where: dict | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """读取 collection 中已有记录。

        这个接口主要给 memory 场景使用：
        - 按 memory_key + session_id 查旧记录，做 upsert
        - 按 session 读取全量 metadata，构建全局记忆索引

        这里仍然保持最小抽象，不把 Chroma 的所有参数都暴露出来，
        避免业务层和底层 SDK 形成强耦合。
        """

        collection = self.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        kwargs = {
            "ids": ids,
            "where": where,
            "include": ["documents", "metadatas"],
        }
        if limit is not None:
            kwargs["limit"] = max(limit, 0)
        if offset is not None:
            kwargs["offset"] = max(offset, 0)
        return collection.get(**kwargs)

    def count(self, *, collection_name: str) -> int:
        """读取 collection 当前记录数，用于分页窗口控制。"""

        collection = self.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return int(collection.count())

    def delete(
        self,
        *,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> None:
        """删除 collection 中的记录。

        文档重新导入时，SQLite catalog 会先替换 chunk；Chroma 也需要按 doc_id
        删除旧 chunk，避免新文档 chunk 数变少时残留旧向量被召回。
        """

        collection = self.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        kwargs = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if kwargs:
            collection.delete(**kwargs)
