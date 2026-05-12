"""Unit tests: app/retrieval/content_quality.py + run_content_quality_step."""

from __future__ import annotations

from app.agents.rag.doc_pipeline import (
    create_doc_pipeline_state,
    run_content_quality_step,
)
from app.agents.rag.types import DocRetrievalPipelineConfig
from app.retrieval.content_quality import looks_like_template_placeholder

# ----------------------------- 判定函数 -----------------------------


class TestPlaceholderDetection:
    def test_triggers_on_template_skeleton(self) -> None:
        # eval 里实际命中的原文，3 个中文占位符 → 应当判为模板骨架。
        content = (
            "## 故障排查\n\n错误：[常见错误消息]\n原因：[为什么会出现]\n"
            "解决方法：[怎么修]\n\n```\n**写指令的最佳实践"
        )
        assert looks_like_template_placeholder(content) is True

    def test_single_placeholder_not_enough(self) -> None:
        # 只有 1 个方括号标注：可能是正常修辞（"[注]"），不应误伤。
        content = "常见问题请参考 [附录一] 的说明。"
        assert looks_like_template_placeholder(content) is False

    def test_numeric_citation_refs_whitelisted(self) -> None:
        # RAG 回答里的 `[1]` / `[2]` 是 citation refs，不应被当作占位符计数。
        content = "根据资料[1][2][3]，这是一个有信息量的句子，不是模板骨架。"
        assert looks_like_template_placeholder(content) is False

    def test_empty_content(self) -> None:
        assert looks_like_template_placeholder("") is False

    def test_placeholders_mixed_with_real_content(self) -> None:
        # 模板+示例混排：出现两个占位符就已经足够判定（保守策略）。
        content = "步骤 1：调用接口 [接口名称] 并记录返回 [返回值说明]，然后再做处理。"
        assert looks_like_template_placeholder(content) is True


# ----------------------------- pipeline 集成 -----------------------------


def _make_state(docs: list[dict]):
    """最小 state 夹具：只关心 content_quality step 读写的字段。"""

    config = DocRetrievalPipelineConfig(
        query_type="factual",
        doc_top_k=5,
        doc_rerank_top_k=2,
        candidate_top_k=20,
        score_threshold=0.5,
        soft_match_threshold=0.35,
        hybrid_alpha=0.6,
        hybrid_beta=0.4,
    )
    state = create_doc_pipeline_state("任意问题", config)
    state.docs = list(docs)
    return state


class TestRunContentQualityStep:
    def test_drops_template_and_records_count(self) -> None:
        bad = {
            "id": "c1",
            "content": "错误：[常见错误消息] 原因：[原因] 解决方法：[怎么修]",
            "score": 0.9,
        }
        good = {
            "id": "c2",
            "content": "payment-service 超时是因为下游 bank-adapter 偶发 TLS 握手失败。",
            "score": 0.7,
        }
        state = _make_state([bad, good])
        result = run_content_quality_step(state)

        assert [doc["id"] for doc in result.docs] == ["c2"]
        assert result.retrieval_debug["template_placeholder_dropped"] == 1

    def test_reads_preview_when_content_missing(self) -> None:
        # 部分 retrieval 路径只放 preview 字段（content 为空）；过滤函数
        # 应当 fall back 到 preview，避免放行明显的骨架 chunk。
        doc = {
            "id": "c1",
            "content": "",
            "preview": "错误：[消息] 原因：[说明] 解法：[办法]",
        }
        state = _make_state([doc])
        result = run_content_quality_step(state)

        assert result.docs == []
        assert result.retrieval_debug["template_placeholder_dropped"] == 1

    def test_keeps_all_when_no_template(self) -> None:
        docs = [
            {"id": "c1", "content": "段落一。"},
            {"id": "c2", "content": "段落二。"},
        ]
        state = _make_state(docs)
        result = run_content_quality_step(state)

        assert [doc["id"] for doc in result.docs] == ["c1", "c2"]
        assert result.retrieval_debug["template_placeholder_dropped"] == 0
