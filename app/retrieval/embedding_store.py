import json
import math
from pathlib import Path
from typing import List, Dict

DOC_EMBED_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "doc_embeddings.json"
)
MEMORY_EMBED_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "memory_embeddings.json"
)


def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: List[Dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
