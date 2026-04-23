from app.constants.model_profiles import PROFILE_DEFAULT_EMBEDDING
from app.llm import embed_text


def get_embedding(text: str, profile: str = PROFILE_DEFAULT_EMBEDDING) -> list[float]:
    """兼容旧调用入口。

    现在 embedding 的 provider / model 解析已经统一收敛到 `app.llm`。
    这里保留薄封装，只是为了避免一口气修改所有旧引用点。
    """

    return embed_text(text, profile=profile)
