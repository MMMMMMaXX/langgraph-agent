"""环境变量解析辅助（仅供 app.llm 包内部使用）。

与 app/env.py 的区别：
- app/env.py 负责加载 .env 文件（.env -> os.environ）
- 这里提供读取单个环境变量的类型化小工具，支持 fallback 链和类型容错
"""

import os


def _env(name: str, fallback: str | None = None) -> str | None:
    """读取环境变量并去除首尾空白。

    - 返回 None 表示未配置或为空字符串
    - fallback 指定另一个环境变量名，用于向下兼容老命名（如 DEEPSEEK_API_KEY -> API_KEY）
    """

    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return value.strip()
    if fallback is None:
        return None
    fallback_value = os.getenv(fallback)
    if fallback_value is not None and fallback_value.strip() != "":
        return fallback_value.strip()
    return None


def _env_float(name: str, default: float) -> float:
    """解析 float 型环境变量，解析失败回退到 default。"""

    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """解析 int 型环境变量，解析失败回退到 default。"""

    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
