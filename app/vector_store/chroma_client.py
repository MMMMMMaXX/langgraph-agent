from pathlib import Path
from typing import Any

from app.config import VECTOR_STORE_CONFIG


def get_chroma_client() -> Any:
    """返回本地持久化 Chroma client。

    这里延迟 import `chromadb`，有两个目的：
    - 没装依赖时，错误提示更聚焦、更容易定位
    - 不会影响当前仍在使用 JSON 存储的旧路径启动
    """

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "缺少 chromadb 依赖，请先安装 requirements.txt 中的新依赖。"
        ) from exc

    persist_dir = Path(VECTOR_STORE_CONFIG.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    # 创建本地持久化 Chroma 客户端，数据落在 data/chroma，服务重启后索引仍在。
    return chromadb.PersistentClient(path=str(persist_dir))
