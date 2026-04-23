# python -m scripts.build_doc_embeddings
import json
from pathlib import Path

from app.llm import embed_texts
from app.retrieval.embedding_store import save_json

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "doc_embeddings.json"
BATCH_SIZE = 16


def main():
    docs = json.loads(DB_PATH.read_text(encoding="utf-8"))
    results = []
    contents = [doc["content"] for doc in docs]

    # 文档建索引通常是离线批处理任务，使用批量 embedding 更接近真实工程形态：
    # - 可以减少请求次数
    # - 后续更容易接入统一的 batching / retry / cache 策略
    for batch_start in range(0, len(contents), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(contents))
        batch_texts = contents[batch_start:batch_end]
        embeddings = embed_texts(batch_texts, profile="doc_embedding")

        for item_index, embedding in enumerate(embeddings, start=batch_start):
            results.append(
                {
                    "id": item_index,
                    "content": contents[item_index],
                    "embedding": embedding,
                }
            )
            print(f"done doc {item_index}")

    save_json(OUT_PATH, results)
    print("doc embeddings built.")


if __name__ == "__main__":
    main()
