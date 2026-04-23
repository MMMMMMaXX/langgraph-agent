import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"


def _set_legacy_langchain_aliases() -> None:
    """兼容旧版 LangChain/LangSmith 环境变量名。

    新版推荐 LANGSMITH_*，部分旧依赖仍读取 LANGCHAIN_*。
    这里只在旧变量未显式设置时补齐别名，避免覆盖用户已有配置。
    """

    alias_pairs = (
        ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
        ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
        ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    )
    for source_name, target_name in alias_pairs:
        source_value = os.getenv(source_name, "").strip()
        if source_value and not os.getenv(target_name, "").strip():
            os.environ[target_name] = source_value


def load_project_env() -> None:
    """加载项目根目录的 `.env` 文件。

    `override=False` 保证命令行或部署平台显式传入的环境变量优先级最高。
    """

    load_dotenv(PROJECT_ENV_PATH, override=False)
    _set_legacy_langchain_aliases()
