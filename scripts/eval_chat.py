# DOC_SCORE_THRESHOLD=0.5 \
# MAX_DOC_CONTEXT_CHARS=360 \
# MAX_DOC_ANSWER_TOKENS=180 \
# uvicorn app.api:app --reload

# EVAL_BASE_URL=http://127.0.0.1:8000 \
# EVAL_CASE_IDS=aria_definition,virtual_list_definition \
# ./.venv/bin/python scripts/eval_chat.py

import ast
import contextlib
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Phase 1 Auth：eval 既有 case 不带 auth，需要显式启用匿名 fallback 才能走通；
# 使用 setdefault 不覆盖外部显式配置，方便想测 401 行为时从命令行关掉。
os.environ.setdefault("ALLOW_ANONYMOUS_AUTH", "true")
# tool_safety 用例需要 confirmation 签发/校验；给 eval 一个固定默认 secret，
# 便于多步 case 签发的 token 在同一个进程内再次被校验通过。
os.environ.setdefault("CONFIRMATION_SECRET", "eval-confirmation-secret")
# tool_executions / mock_tickets 默认写入 data/operations.sqlite3；eval 每次运行
# 使用独立临时路径，避免历史 idempotency_key 污染 dedup 判定。
if "OPERATIONS_SQLITE_PATH" not in os.environ:
    import tempfile as _tempfile

    _eval_ops_dir = _tempfile.mkdtemp(prefix="eval-ops-")
    os.environ["OPERATIONS_SQLITE_PATH"] = str(
        Path(_eval_ops_dir) / "operations.sqlite3"
    )

import app.api as api
from app.constants.eval import (
    EVAL_BOOL_FALSE,
    EVAL_BOOL_TRUE,
    EVAL_CASE_SETUP_IMPORTS_KEY,
    EVAL_CATEGORY_FALLBACK,
    EVAL_BASE_URL_ENV,
    EVAL_CASE_IDS_ENV,
    EVAL_CONVERSATION_HISTORY_PATH_ENV,
    EVAL_EXPECTED_FALLBACK_KEY,
    EVAL_EXPECTED_IMPORT_CHUNK_ALIAS_KEY,
    EVAL_EXPECTED_IMPORT_CHUNK_INDEX_KEY,
    EVAL_EXPECTED_IMPORT_CHUNKS_KEY,
    EVAL_FIELD_NOT_APPLICABLE,
    EVAL_HTTP_TIMEOUT_ENV,
    EVAL_HTTP_TIMEOUT_SECONDS,
    EVAL_IMPORT_ALIAS_KEY,
    EVAL_IMPORT_CONTENT_KEY,
    EVAL_IMPORT_CONTENT_PATH_KEY,
    EVAL_OUTPUT_CSV_ENV,
    EVAL_OUTPUT_JSON_ENV,
)
from app.constants.policies import INSUFFICIENT_KNOWLEDGE_ANSWER

CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
CITATION_REF_PATTERN = re.compile(r"\[(\d+)\]")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def filter_cases(cases: list[dict]) -> list[dict]:
    case_ids = os.getenv(EVAL_CASE_IDS_ENV, "").strip()
    if not case_ids:
        return cases

    selected_ids = {item.strip() for item in case_ids.split(",") if item.strip()}
    return [case for case in cases if case["id"] in selected_ids]


def extract_scalar(log_text: str, key: str) -> str:
    marker = f"{key} = "
    for line in log_text.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return ""


def extract_bool(log_text: str, key: str) -> str:
    value = extract_scalar(log_text, key)
    return value or "-"


def extract_node_timings(log_text: str) -> dict:
    value = extract_scalar(log_text, "nodeTimingsMs")
    if not value:
        return {}
    try:
        return ast.literal_eval(value)
    except Exception:
        return {}


def get_debug_payload(payload: dict) -> dict:
    return payload.get("debug") or {}


def format_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def parse_ms(value) -> float | None:
    if value in (None, "", "-"):
        return None
    return float(value)


def answer_quality(answer: str) -> str:
    """粗粒度答案质量信号，仅基于长度和兜底字符串，不做语义评分。

    - poor：空答案或知识不足兜底
    - ok：有内容但过短（< 20 字），可能是错误信息
    - good：正常长度回答（≥ 20 字），不代表内容正确
    """
    if not answer or answer == INSUFFICIENT_KNOWLEDGE_ANSWER:
        return "poor"
    if len(answer) < 20:
        return "ok"
    return "good"


def contains_all(text: str, expected_parts: list[str]) -> bool:
    return all(part in text for part in expected_parts)


def contains_any(text: str, blocked_parts: list[str]) -> bool:
    return any(part in text for part in blocked_parts)


def get_nested_value(data: dict, dotted_path: str):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def normalize_expected_ids(value) -> list[str]:
    """把 case 中的期望 doc/chunk id 统一成字符串列表。"""

    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def collect_hit_identifiers(hit: dict) -> set[str]:
    """收集一个 debug hit 可以代表的所有 id。

    merged_docs 可能把多个相邻 chunk 合成一段，所以这里同时看：
    - id：当前 hit id，可能是单 chunk，也可能是 "a+b" 形式
    - doc_id：原始文档 id，适合粗粒度评估
    - merged_chunk_ids：合并前的 chunk id 列表，适合精确 chunk 命中评估
    """

    identifiers = set()
    for key in ("id", "doc_id"):
        value = hit.get(key)
        if value not in (None, ""):
            identifiers.add(str(value))

    for value in hit.get("merged_chunk_ids") or []:
        if value not in (None, ""):
            identifiers.add(str(value))

    return identifiers


def ordered_unique(values: list[str]) -> list[str]:
    """保留顺序去重，便于 eval 输出稳定且可读。"""

    seen = set()
    result = []
    for value in values:
        if value in (None, "", EVAL_FIELD_NOT_APPLICABLE) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def collect_stage_doc_ids(hits: list[dict]) -> list[str]:
    """收集某个检索阶段返回过的 doc_id。"""

    return ordered_unique(
        [str(hit.get("doc_id", "")) for hit in hits if hit.get("doc_id")]
    )


def collect_stage_chunk_ids(hits: list[dict]) -> list[str]:
    """收集某个检索阶段返回过的 chunk id，兼容 merged_chunk_ids。"""

    chunk_ids: list[str] = []
    for hit in hits:
        value = hit.get("chunk_id") or hit.get("id")
        if value not in (None, ""):
            chunk_ids.append(str(value))
        for merged_chunk_id in hit.get("merged_chunk_ids") or []:
            if merged_chunk_id not in (None, ""):
                chunk_ids.append(str(merged_chunk_id))
    return ordered_unique(chunk_ids)


def join_ids(values: list[str]) -> str:
    return ",".join(ordered_unique(values)) or EVAL_FIELD_NOT_APPLICABLE


def hits_contain_expected(
    hits: list[dict],
    *,
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
) -> bool | str:
    """判断某个检索阶段是否命中预期文档/切片。

    没有配置 expected_* 时返回 "-"，表示该 case 不参与 retrieval hit 统计。
    如果配置了 expected_chunk_ids，则优先按精确 chunk 判断；这样不会因为同一
    doc_id 命中而掩盖“引用了错误段落”的问题。
    """

    expected_ids = set(expected_chunk_ids or expected_doc_ids)
    if not expected_ids:
        return "-"

    for hit in hits:
        if collect_hit_identifiers(hit) & expected_ids:
            return True
    return False


def bool_metric(value: bool | str) -> str:
    if value == EVAL_FIELD_NOT_APPLICABLE:
        return EVAL_FIELD_NOT_APPLICABLE
    return EVAL_BOOL_TRUE if value else EVAL_BOOL_FALSE


def extract_answer_citation_refs(answer: str) -> list[str]:
    """提取回答中实际出现的引用编号，如 [1]、[2]。"""

    return sorted(set(CITATION_REF_PATTERN.findall(answer)), key=int)


def collect_available_citation_refs(citations: list[dict]) -> set[str]:
    """收集 debug citations 中声明过的引用编号。"""

    refs = set()
    for citation in citations:
        index = citation.get("index")
        if index not in (None, ""):
            refs.add(str(index))
            continue

        ref = str(citation.get("ref", ""))
        match = CITATION_REF_PATTERN.fullmatch(ref.strip())
        if match:
            refs.add(match.group(1))
    return refs


def build_answer_citation_eval(answer: str, rag_debug: dict) -> dict:
    """评估回答是否正确使用 debug 中声明过的 citation。"""

    citations = rag_debug.get("citations") or []
    available_refs = collect_available_citation_refs(citations)
    answer_refs = extract_answer_citation_refs(answer)
    answer_ref_set = set(answer_refs)
    invalid_refs = sorted(answer_ref_set - available_refs, key=int)
    missing_refs = sorted(available_refs - answer_ref_set, key=int)
    doc_used = bool(rag_debug.get("doc_used"))

    return {
        "answer_citation_refs": ",".join(answer_refs) or EVAL_FIELD_NOT_APPLICABLE,
        "answer_citation_count": len(answer_refs),
        "answer_has_citation": bool_metric(bool(answer_refs)) if doc_used else "-",
        "citation_refs_valid": (
            bool_metric(not invalid_refs) if answer_refs or available_refs else "-"
        ),
        "invalid_citation_refs": ",".join(invalid_refs) or EVAL_FIELD_NOT_APPLICABLE,
        "unused_citation_refs": ",".join(missing_refs) or EVAL_FIELD_NOT_APPLICABLE,
    }


def calculate_expected_doc_coverage(
    *,
    expected_doc_ids: list[str],
    citation_doc_ids: list[str],
) -> tuple[str, str]:
    """计算 citation 对 expected_doc_ids 的覆盖情况。"""

    if not expected_doc_ids:
        return "-", "-"

    expected = set(expected_doc_ids)
    actual = set(citation_doc_ids)
    covered = len(expected & actual)
    total = len(expected)
    return f"{covered}/{total}", bool_metric(covered == total)


def infer_retrieval_failure_stage(metrics: dict) -> str:
    """根据各阶段命中状态推断正确文档最早在哪一步丢失。"""

    if metrics["top_k_hit"] == "-":
        return "-"
    if not metrics["top_k_hit"]:
        return "top_docs_miss"
    if not metrics["filtered_hit"]:
        return "threshold_miss"
    if not metrics["rerank_hit"]:
        return "rerank_miss"
    if not metrics["merged_hit"]:
        return "chunk_merge_miss"
    return ""


def build_retrieval_eval(case: dict, debug_nodes: dict, answer: str = "") -> dict:
    """从 API debug payload 计算分阶段 retrieval eval 指标。"""

    expected_doc_ids = normalize_expected_ids(case.get("expected_doc_ids"))
    expected_chunk_ids = normalize_expected_ids(case.get("expected_chunk_ids"))
    rag_debug = debug_nodes.get("rag_agent", {})
    doc_debug = (rag_debug.get("retrieval_debug") or {}).get("doc", {})
    top_docs = rag_debug.get("top_docs") or []
    filtered_docs = rag_debug.get("filtered_docs") or []
    post_rerank_docs = rag_debug.get("post_rerank_docs") or []
    merged_docs = rag_debug.get("merged_docs") or []
    citations = rag_debug.get("citations") or []
    citation_doc_ids = [
        str(citation.get("doc_id", ""))
        for citation in citations
        if citation.get("doc_id")
    ]
    citation_chunk_ids = collect_stage_chunk_ids(citations)
    citation_coverage, citation_all_hit = calculate_expected_doc_coverage(
        expected_doc_ids=expected_doc_ids,
        citation_doc_ids=citation_doc_ids,
    )
    expected_fallback = bool(case.get(EVAL_EXPECTED_FALLBACK_KEY)) or (
        case.get("category") == EVAL_CATEGORY_FALLBACK
    )

    metrics = {
        "expected_doc_ids": join_ids(expected_doc_ids),
        "expected_chunk_ids": join_ids(expected_chunk_ids),
        "citation_count": len(citations),
        "citation_doc_ids": join_ids(citation_doc_ids),
        "citation_chunk_ids": join_ids(citation_chunk_ids),
        "source_doc_ids": join_ids(citation_doc_ids),
        "used_chunk_ids": join_ids(citation_chunk_ids),
        "citation_expected_doc_coverage": citation_coverage,
        "citation_all_expected_docs_hit": citation_all_hit,
        "citation_hit": (
            "-"
            if not (expected_doc_ids or expected_chunk_ids)
            else hits_contain_expected(
                citations,
                expected_doc_ids=expected_doc_ids,
                expected_chunk_ids=expected_chunk_ids,
            )
        ),
        "top_k_hit": hits_contain_expected(
            top_docs,
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "filtered_hit": hits_contain_expected(
            filtered_docs,
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "rerank_hit": hits_contain_expected(
            post_rerank_docs,
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "merged_hit": hits_contain_expected(
            merged_docs,
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "top_k_doc_ids": join_ids(collect_stage_doc_ids(top_docs)),
        "filtered_doc_ids": join_ids(collect_stage_doc_ids(filtered_docs)),
        "rerank_doc_ids": join_ids(collect_stage_doc_ids(post_rerank_docs)),
        "merged_doc_ids": join_ids(collect_stage_doc_ids(merged_docs)),
        "top_k_chunk_ids": join_ids(collect_stage_chunk_ids(top_docs)),
        "filtered_chunk_ids": join_ids(collect_stage_chunk_ids(filtered_docs)),
        "rerank_chunk_ids": join_ids(collect_stage_chunk_ids(post_rerank_docs)),
        "merged_chunk_ids": join_ids(collect_stage_chunk_ids(merged_docs)),
        "fallback_accuracy": (
            bool_metric(answer == INSUFFICIENT_KNOWLEDGE_ANSWER)
            if expected_fallback
            else EVAL_FIELD_NOT_APPLICABLE
        ),
        "dense_count": doc_debug.get("dense_count", "-"),
        "lexical_count": doc_debug.get("lexical_count", "-"),
        "hybrid_count": doc_debug.get("hybrid_count", "-"),
        "filtered_count": doc_debug.get("filtered_count", "-"),
        "rerank_count": doc_debug.get("consumed_count", "-"),
        "merged_count": doc_debug.get("merged_count", "-"),
    }
    metrics.update(build_answer_citation_eval(answer, rag_debug))
    metrics["retrieval_failure_stage"] = infer_retrieval_failure_stage(metrics)

    return {
        key: bool_metric(value) if isinstance(value, bool) or value == "-" else value
        for key, value in metrics.items()
    }


def evaluate_case_assertions(
    case: dict,
    answer: str,
    actual_route: str,
    debug_nodes: dict | None = None,
) -> tuple[str, str]:
    problems = []
    debug_nodes = debug_nodes or {}

    expected_route = case.get("expected_route", "")
    if expected_route and expected_route not in actual_route.split(","):
        problems.append(
            f"route mismatch: expected {expected_route}, got {actual_route}"
        )

    must_include = case.get("must_include", [])
    if must_include and not contains_all(answer, must_include):
        problems.append(f"missing expected text: {must_include}")

    must_not_include = case.get("must_not_include", [])
    if must_not_include and contains_any(answer, must_not_include):
        problems.append(f"contains blocked text: {must_not_include}")

    for dotted_path, expected_value in case.get("debug_must_equal", {}).items():
        actual_value = get_nested_value(debug_nodes, dotted_path)
        if actual_value != expected_value:
            problems.append(
                f"debug mismatch: {dotted_path} expected {expected_value}, got {actual_value}"
            )

    if problems:
        return "fail", "; ".join(problems)
    return "pass", ""


def post_chat(
    client,
    session_id: str,
    message: str,
    *,
    auth: dict | None = None,
    confirmation_token: str = "",
) -> dict:
    payload = {
        "session_id": session_id,
        "message": message,
        "debug": True,
    }
    if auth:
        payload["auth"] = auth
    if confirmation_token:
        payload["confirmation_token"] = confirmation_token
    conversation_history_path = os.getenv(
        EVAL_CONVERSATION_HISTORY_PATH_ENV, ""
    ).strip()
    if conversation_history_path:
        payload["conversation_history_path"] = conversation_history_path

    response = client.post(
        "/chat",
        json=payload,
    )
    return {
        "status_code": response.status_code,
        "payload": response.json(),
    }


def post_knowledge_import(client, import_payload: dict) -> dict:
    """导入 eval case 需要的临时知识文档。"""

    response = client.post("/knowledge/import", json=import_payload)
    payload = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"knowledge import failed: {payload}")
    return payload


def resolve_import_content_path(raw_path: str) -> Path:
    """解析 eval fixture 路径。

    相对路径以仓库根目录为基准，便于 eval_cases.json 在任意 cwd 下运行。
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def build_knowledge_import_payload(import_config: dict) -> dict:
    """把 eval setup 配置转换成 /knowledge/import 请求体。"""

    payload = {
        key: value
        for key, value in import_config.items()
        if key not in {EVAL_IMPORT_ALIAS_KEY, EVAL_IMPORT_CONTENT_PATH_KEY}
    }
    content_path = str(import_config.get(EVAL_IMPORT_CONTENT_PATH_KEY) or "").strip()
    if content_path:
        payload[EVAL_IMPORT_CONTENT_KEY] = resolve_import_content_path(
            content_path
        ).read_text(encoding="utf-8")
    return payload


def setup_knowledge_imports(client, case: dict) -> dict[str, str]:
    """执行 case 级知识导入，并返回 import alias -> doc_id 映射。

    这样 eval 不需要把内容 hash 生成的 doc_id 写死在 eval_cases.json 里。
    """

    alias_to_doc_id: dict[str, str] = {}
    for index, import_config in enumerate(case.get(EVAL_CASE_SETUP_IMPORTS_KEY, [])):
        import_payload = build_knowledge_import_payload(import_config)
        payload = post_knowledge_import(client, import_payload)
        alias = str(import_config.get(EVAL_IMPORT_ALIAS_KEY) or f"import_{index}")
        alias_to_doc_id[alias] = str(payload.get("doc_id", ""))
    return alias_to_doc_id


def resolve_expected_doc_ids(case: dict, alias_to_doc_id: dict[str, str]) -> dict:
    """把导入 alias 解析成实际 doc/chunk id，返回 case 副本。

    eval 运行时导入文档的 doc_id 由内容 hash 生成，不能在 eval_cases.json
    里写死。这里支持两层解析：
    1. expected_import_aliases -> expected_doc_ids，用于“命中文档即可”的 case。
    2. expected_import_chunks -> expected_chunk_ids，用于要求命中特定章节/chunk
       的 case，避免只命中文档但引用了错误段落也被误判通过。
    """

    if not alias_to_doc_id:
        return case

    resolved = dict(case)
    expected_doc_ids = normalize_expected_ids(case.get("expected_doc_ids"))
    for alias in case.get("expected_import_aliases", []):
        doc_id = alias_to_doc_id.get(str(alias), "")
        if doc_id:
            expected_doc_ids.append(doc_id)

    if expected_doc_ids:
        resolved["expected_doc_ids"] = expected_doc_ids

    expected_chunk_ids = normalize_expected_ids(case.get("expected_chunk_ids"))
    for chunk_ref in case.get(EVAL_EXPECTED_IMPORT_CHUNKS_KEY, []):
        alias = str(chunk_ref.get(EVAL_EXPECTED_IMPORT_CHUNK_ALIAS_KEY, ""))
        doc_id = alias_to_doc_id.get(alias, "")
        if not doc_id:
            continue
        chunk_index = chunk_ref.get(EVAL_EXPECTED_IMPORT_CHUNK_INDEX_KEY)
        if chunk_index is None:
            continue
        expected_chunk_ids.append(f"{doc_id}::chunk::{chunk_index}")

    if expected_chunk_ids:
        resolved["expected_chunk_ids"] = expected_chunk_ids
    return resolved


def get_payload_value(payload: dict, dotted_path: str):
    """按 dotted path 从响应 payload 读值。

    支持顺序遍历 dict / list，适合 eval step 之间 capture 跨步骤字段，
    例如 `pending_confirmation.token` 或
    `debug.nodes.tool_agent.pending_confirmation.token`。
    """

    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def substitute_variables(value, context: dict):
    """在字符串 / dict / list 中把 `${name}` 替换为 context 里捕获的值。"""

    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            return context.get(value[2:-1], "")
        return value
    if isinstance(value, dict):
        return {k: substitute_variables(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_variables(v, context) for v in value]
    return value


def run_case(client, case: dict) -> dict:
    if case.get("steps"):
        return run_multistep_case(client, case)
    return run_single_step_case(client, case)


def run_multistep_case(client, case: dict) -> dict:
    """跑多步 case：前后步骤之间共享 session_id 和 capture 变量。"""

    capture_context: dict[str, Any] = {}
    step_results: list[dict] = []
    aggregated_problems: list[str] = []
    last_payload: dict = {}
    last_response: dict = {"status_code": 0, "payload": {}}
    total_duration_ms = 0.0
    captured_stdout = io.StringIO()
    session_id = case["session_id"]

    with contextlib.redirect_stdout(captured_stdout):
        alias_to_doc_id = setup_knowledge_imports(client, case)

        for index, raw_step in enumerate(case["steps"]):
            step = substitute_variables(raw_step, capture_context)
            message = step.get("message", "")
            auth = step.get("auth")
            token = step.get("confirmation_token", "") or ""
            started = time.perf_counter()
            response = post_chat(
                client,
                session_id,
                message,
                auth=auth,
                confirmation_token=token,
            )
            step_ms = (time.perf_counter() - started) * 1000
            total_duration_ms += step_ms
            last_response = response
            last_payload = response["payload"]

            expect = step.get("expect", {}) or {}
            # capture：pull values out of payload for later steps.
            for name, path in (expect.get("capture") or {}).items():
                capture_context[name] = get_payload_value(last_payload, path)

            problems = evaluate_step_expectations(step, last_payload)
            if problems:
                aggregated_problems.append(f"step[{index}]: " + "; ".join(problems))
            step_results.append(
                {
                    "index": index,
                    "message": message,
                    "status_code": response["status_code"],
                    "answer": last_payload.get("answer", ""),
                    "pending_confirmation": bool(
                        last_payload.get("pending_confirmation")
                    ),
                    "tool_executions": list(last_payload.get("tool_executions") or []),
                    "duration_ms": step_ms,
                    "problems": problems,
                }
            )

    case = resolve_expected_doc_ids(case, alias_to_doc_id)
    debug_payload = get_debug_payload(last_payload)
    debug_nodes = debug_payload.get("nodes") or {}
    answer = last_payload.get("answer", "")
    routes = last_payload.get("routes", [])
    actual_route = ",".join(routes) if routes else "-"

    # 顶层 must_include/must_not_include/debug_must_equal 作用于"最后一步"。
    final_problems = evaluate_case_assertions(case, answer, actual_route, debug_nodes)[
        1
    ]
    if final_problems:
        aggregated_problems.append(final_problems)
    assertion_status = "pass" if not aggregated_problems else "fail"

    retrieval_eval = build_retrieval_eval(case, debug_nodes, answer)
    tool_safety_metrics = collect_tool_safety_metrics(case, step_results, last_payload)
    workflow_metrics = collect_workflow_metrics(
        case, debug_nodes, last_payload, step_results
    )

    return {
        "id": case["id"],
        "category": case.get("category", "-"),
        "status_code": last_response["status_code"],
        "expected_route": case.get("expected_route", ""),
        "actual_route": actual_route,
        "doc_used": "-",
        "memory_used": "-",
        "request_ms": format_ms(total_duration_ms),
        "rag_ms": "-",
        "memory_ms": "-",
        "answer_len": len(answer),
        "quality": answer_quality(answer),
        "assertion": assertion_status,
        "assertion_detail": "; ".join(aggregated_problems),
        "steps": step_results,
        **retrieval_eval,
        **tool_safety_metrics,
        **workflow_metrics,
        "debug_nodes": debug_nodes,
        "answer": answer,
        "detail": last_payload.get("detail", ""),
    }


def evaluate_step_expectations(step: dict, payload: dict) -> list[str]:
    """校验 step-level 期望；返回失败原因列表，空列表表示通过。

    expect 支持：
    - must_include / must_not_include：作用于 answer
    - debug_must_equal：dotted path 基于 payload["debug"]["nodes"]
    - status_code：等值校验
    - pending_confirmation_present：bool，断言响应顶层 pending_confirmation 是否存在
    - payload_must_equal：dotted path 基于完整 payload
    """

    expect = step.get("expect", {}) or {}
    problems: list[str] = []

    answer = payload.get("answer", "")
    must_include = expect.get("must_include", [])
    if must_include and not contains_all(answer, must_include):
        problems.append(f"missing expected text: {must_include}")
    must_not_include = expect.get("must_not_include", [])
    if must_not_include and contains_any(answer, must_not_include):
        problems.append(f"contains blocked text: {must_not_include}")

    if "pending_confirmation_present" in expect:
        expected = bool(expect["pending_confirmation_present"])
        actual = bool(payload.get("pending_confirmation"))
        if expected != actual:
            problems.append(
                f"pending_confirmation presence mismatch: "
                f"expected {expected}, got {actual}"
            )

    debug_nodes = (payload.get("debug") or {}).get("nodes") or {}
    for dotted, expected in (expect.get("debug_must_equal") or {}).items():
        actual = get_nested_value(debug_nodes, dotted)
        if actual != expected:
            problems.append(
                f"debug mismatch: {dotted} expected {expected!r}, got {actual!r}"
            )
    for dotted, expected in (expect.get("payload_must_equal") or {}).items():
        actual = get_payload_value(payload, dotted)
        if actual != expected:
            problems.append(
                f"payload mismatch: {dotted} expected {expected!r}, got {actual!r}"
            )
    return problems


def _tool_executions_from_payload(payload: dict) -> list[dict]:
    """tool_executions 在 debug=True 场景下出现在响应顶层。"""

    return list(payload.get("tool_executions") or [])


def _step_debug_nodes(payload: dict) -> dict:
    return (payload.get("debug") or {}).get("nodes") or {}


def collect_tool_safety_metrics(
    case: dict,
    step_results: list[dict],
    last_payload: dict,
) -> dict[str, str]:
    """为 tool_safety 类 case 计算聚合指标。"""

    check = case.get("tool_safety_check", "")
    if not check:
        return {}

    # 聚合所有 step 的 tool_executions；单步 case 时 step_results 为空，退回到 last_payload。
    executions: list[dict] = []
    for sr in step_results:
        executions.extend(sr.get("tool_executions") or [])
    if not executions:
        executions = _tool_executions_from_payload(last_payload)
    statuses = [ex.get("status") for ex in executions]
    metrics: dict[str, str] = {"tool_safety_check": check}

    if check == "side_effect_requires_confirmation":
        metrics["side_effect_executed_without_confirmation"] = bool_metric(
            "succeeded" in statuses
        )
    elif check == "anonymous_side_effect_blocked":
        rejected = any(s == "rejected_anonymous" for s in statuses)
        # 匿名情况下 LLM 看不到 side_effect 工具；没执行也算"被阻止"。
        no_succeeded_side_effect = all(s != "succeeded" for s in statuses)
        metrics["anonymous_side_effect_blocked"] = bool_metric(
            rejected or no_succeeded_side_effect
        )
    elif check == "idempotency_dedup":
        # 同一个 idempotency_key 最多只有一条 succeeded 落库；重复调用应命中
        # EXISTING 分支（status 仍是 succeeded，但 tool_executions 里每条都
        # 共享同一 key）。规则：按 idempotency_key 去重后，succeeded 记录 ≤1。
        succeeded_keys = {
            ex.get("idempotency_key", "")
            for ex in executions
            if ex.get("status") == "succeeded"
        }
        # 至少要有一次 succeeded，否则说明根本没跑成功，不算"去重"。
        ok = bool(succeeded_keys) and len(succeeded_keys) == 1
        metrics["idempotency_dedup_rate"] = bool_metric(ok)
    return metrics


def collect_workflow_metrics(
    case: dict,
    debug_nodes: dict,
    last_payload: dict,
    step_results: list[dict] | None = None,
) -> dict[str, str]:
    """为 category=workflow case 计算 Phase 2 DoD 要求的三项指标。

    - plan_schema_pass_rate: Planner 是否成功产出合法 plan（debug.planner.status=="ok"）
    - workflow_success_rate: 最终 workflow_status 是否落在"闭环"区间
      （succeeded / need_confirmation 都视为正确闭环，failed/partial/need_clarification
      不算；assertion 失败直接降为 false，避免 status 写了 succeeded 但答案错）
    - confirmation_bridge_rate: 对出现 pending_confirmation 的 case，后续是否通过
      token 重放成功落库；单步 case 天然无重放机会，该字段为 N/A（不计入分母）

    放在 `category=workflow` 维度聚合，避免污染 tool_safety/retrieval 既有指标。
    """

    if case.get("category") != "workflow":
        return {}

    planner_debug = debug_nodes.get("planner", {}) or {}
    composer_debug = debug_nodes.get("composer", {}) or {}

    plan_ok = planner_debug.get("status") == "ok"
    workflow_status = composer_debug.get("workflow_status", "")
    closed_loop_statuses = {"succeeded", "need_confirmation"}
    workflow_ok = workflow_status in closed_loop_statuses

    metrics: dict[str, str] = {
        "plan_schema_pass_rate": bool_metric(plan_ok),
        "workflow_success_rate": bool_metric(workflow_ok),
    }

    # confirmation bridge：只对"出现过 pending_confirmation 的 case"统计。
    # 单步 case 无法自证桥接，返回占位 "-" 让分母绕开。
    steps = step_results or []
    had_pending = any(sr.get("pending_confirmation") for sr in steps) or bool(
        last_payload.get("pending_confirmation")
    )
    bridged = False
    # 多步 case：最后一步不带 pending_confirmation 且 tool_executions 里有
    # succeeded 记录，视为 token 桥接成功。
    if steps:
        last_step = steps[-1]
        last_pending = bool(last_step.get("pending_confirmation"))
        last_execs = last_step.get("tool_executions") or []
        bridged = (
            had_pending
            and not last_pending
            and any(ex.get("status") == "succeeded" for ex in last_execs)
        )

    if had_pending and steps:
        metrics["confirmation_bridge_rate"] = bool_metric(bridged)
    else:
        metrics["confirmation_bridge_rate"] = "-"

    return metrics


def run_single_step_case(client, case: dict) -> dict:
    capture = io.StringIO()
    started_at = time.perf_counter()
    alias_to_doc_id: dict[str, str] = {}
    with contextlib.redirect_stdout(capture):
        alias_to_doc_id = setup_knowledge_imports(client, case)
        case = resolve_expected_doc_ids(case, alias_to_doc_id)
        # setup 让单条 case 可以构造真实多轮上下文，
        # 比如“先问北京，再问上海，再做总结”，这样更接近真实 agent 行为验证。
        for setup_message in case.get("setup", []):
            post_chat(client, case["session_id"], setup_message)

        response_data = post_chat(
            client,
            case["session_id"],
            case["message"],
            auth=case.get("auth"),
            confirmation_token=case.get("confirmation_token", "") or "",
        )
    duration_ms = (time.perf_counter() - started_at) * 1000
    payload = response_data["payload"]
    log_text = capture.getvalue()
    debug_payload = get_debug_payload(payload)
    node_timings = debug_payload.get("node_timings") or extract_node_timings(log_text)
    debug_nodes = debug_payload.get("nodes") or {}
    routes = payload.get("routes", [])
    actual_route = ",".join(routes) if routes else "-"
    answer = payload.get("answer", "")
    assertion_status, assertion_detail = evaluate_case_assertions(
        case,
        answer,
        actual_route,
        debug_nodes,
    )
    # 兼容 tool_safety 单步 case：顶层可直接声明 pending_confirmation_present。
    extra_problems: list[str] = []
    if "pending_confirmation_present" in case:
        expected = bool(case["pending_confirmation_present"])
        actual = bool(payload.get("pending_confirmation"))
        if expected != actual:
            extra_problems.append(
                "pending_confirmation presence mismatch: "
                f"expected {expected}, got {actual}"
            )
    if extra_problems:
        assertion_status = "fail"
        assertion_detail = "; ".join(
            [assertion_detail, *extra_problems] if assertion_detail else extra_problems
        )
    retrieval_eval = build_retrieval_eval(case, debug_nodes, answer)
    tool_safety_metrics = collect_tool_safety_metrics(case, [], payload)
    workflow_metrics = collect_workflow_metrics(case, debug_nodes, payload, [])

    return {
        "id": case["id"],
        "category": case.get("category", "-"),
        "status_code": response_data["status_code"],
        "expected_route": case.get("expected_route", ""),
        "actual_route": actual_route,
        "doc_used": (
            str(debug_nodes.get("rag_agent", {}).get("doc_used"))
            if "rag_agent" in debug_nodes
            else extract_bool(log_text, "docUsed")
        ),
        "memory_used": (
            str(debug_nodes.get("rag_agent", {}).get("memory_used"))
            if "rag_agent" in debug_nodes
            else (
                str(debug_nodes.get("chat_agent", {}).get("used_memory"))
                if "chat_agent" in debug_nodes
                else extract_bool(log_text, "memoryUsed")
            )
        ),
        "request_ms": format_ms(duration_ms),
        "rag_ms": format_ms(node_timings.get("rag_agent")),
        "memory_ms": format_ms(node_timings.get("memory")),
        "answer_len": len(answer),
        "quality": answer_quality(answer),
        "assertion": assertion_status,
        "assertion_detail": assertion_detail,
        **retrieval_eval,
        **tool_safety_metrics,
        **workflow_metrics,
        "debug_nodes": debug_nodes,
        "answer": answer,
        "detail": payload.get("detail", ""),
    }


def print_table(results: list[dict]) -> None:
    headers = [
        "id",
        "status_code",
        "category",
        "expected_route",
        "actual_route",
        "doc_used",
        "memory_used",
        "request_ms",
        "rag_ms",
        "memory_ms",
        "answer_len",
        "quality",
        "citation_count",
        "citation_hit",
        "citation_all_expected_docs_hit",
        "answer_has_citation",
        "citation_refs_valid",
        "top_k_hit",
        "filtered_hit",
        "rerank_hit",
        "merged_hit",
        "assertion",
    ]
    rows = [[str(item.get(header, "")) for header in headers] for item in results]
    widths = []
    for idx, header in enumerate(headers):
        max_row_width = max((len(row[idx]) for row in rows), default=0)
        widths.append(max(len(header), max_row_width))

    header_line = " | ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    sep_line = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def print_answer_details(results: list[dict]) -> None:
    print("\nAnswers")
    print("-------")
    for item in results:
        answer = item["answer"] or item["detail"]
        print(f"[{item['id']}] {answer}")


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item.get("assertion") == "pass")
    failed_items = [item for item in results if item.get("assertion") != "pass"]

    category_stats: dict[str, dict] = {}
    for item in results:
        category = item.get("category", "-")
        stats = category_stats.setdefault(
            category,
            {
                "count": 0,
                "passed": 0,
                "request_ms_values": [],
            },
        )
        stats["count"] += 1
        if item.get("assertion") == "pass":
            stats["passed"] += 1

        request_ms = parse_ms(item.get("request_ms"))
        if request_ms is not None:
            stats["request_ms_values"].append(request_ms)

    slowest_cases = sorted(
        results,
        key=lambda item: parse_ms(item.get("request_ms")) or -1,
        reverse=True,
    )[:3]
    retrieval_cases = [
        item for item in results if item.get("top_k_hit") not in (None, "-")
    ]

    retrieval_stats = {}
    for field in (
        "top_k_hit",
        "filtered_hit",
        "rerank_hit",
        "merged_hit",
        "citation_hit",
        "citation_all_expected_docs_hit",
        "answer_has_citation",
        "citation_refs_valid",
    ):
        total_with_expected = len(retrieval_cases)
        hits = sum(1 for item in retrieval_cases if item.get(field) == "true")
        retrieval_stats[field] = {
            "hits": hits,
            "total": total_with_expected,
            "rate": (hits / total_with_expected * 100) if total_with_expected else 0.0,
        }
    fallback_cases = [
        item for item in results if item.get("fallback_accuracy") not in (None, "-")
    ]
    fallback_hits = sum(
        1 for item in fallback_cases if item.get("fallback_accuracy") == "true"
    )

    # Phase 1 tool_safety 指标：只聚合明确声明了 tool_safety_check 的 case。
    # 多步 end-to-end confirmation 闭环由 Scheme A（tool_agent 确定性重放）
    # 兜底，args_hash 必然匹配，因此 idempotency_dedup_rate 可以进入 eval 聚合。
    tool_safety_fields = (
        "side_effect_executed_without_confirmation",
        "anonymous_side_effect_blocked",
        "idempotency_dedup_rate",
    )
    tool_safety_stats: dict[str, dict] = {}
    for field in tool_safety_fields:
        field_cases = [item for item in results if item.get(field) not in (None, "-")]
        # side_effect_executed_without_confirmation：越低越好（true 代表异常执行）；
        # 这里统计"被正确拦截"的比例，所以取 false 命中。其余指标 true 是期望结果。
        if field == "side_effect_executed_without_confirmation":
            hits = sum(1 for item in field_cases if item.get(field) == "false")
        else:
            hits = sum(1 for item in field_cases if item.get(field) == "true")
        total_field = len(field_cases)
        tool_safety_stats[field] = {
            "hits": hits,
            "total": total_field,
            "rate": (hits / total_field * 100) if total_field else 0.0,
        }

    # Phase 2 workflow 三项 DoD 指标：只聚合 category=workflow 的 case。
    # plan_schema_pass_rate / workflow_success_rate 分母 = workflow 总数；
    # confirmation_bridge_rate 只统计带 pending_confirmation 的 case，否则分母为 0。
    workflow_fields = (
        "plan_schema_pass_rate",
        "workflow_success_rate",
        "confirmation_bridge_rate",
    )
    workflow_stats: dict[str, dict] = {}
    for field in workflow_fields:
        field_cases = [item for item in results if item.get(field) not in (None, "-")]
        hits = sum(1 for item in field_cases if item.get(field) == "true")
        total_field = len(field_cases)
        workflow_stats[field] = {
            "hits": hits,
            "total": total_field,
            "rate": (hits / total_field * 100) if total_field else 0.0,
        }

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total * 100) if total else 0.0,
        "failed_items": failed_items,
        "category_stats": category_stats,
        "slowest_cases": slowest_cases,
        "retrieval_stats": retrieval_stats,
        "fallback_stats": {
            "hits": fallback_hits,
            "total": len(fallback_cases),
            "rate": (
                fallback_hits / len(fallback_cases) * 100 if fallback_cases else 0.0
            ),
        },
        "tool_safety_stats": tool_safety_stats,
        "workflow_stats": workflow_stats,
    }


def print_summary(results: list[dict]) -> None:
    summary = summarize_results(results)

    print("\nSummary")
    print("-------")
    print(
        f"pass_rate={summary['pass_rate']:.1f}% "
        f"({summary['passed']}/{summary['total']})"
    )

    print("\nBy category")
    print("-----------")
    for category, stats in sorted(summary["category_stats"].items()):
        values = stats["request_ms_values"]
        avg_request_ms = sum(values) / len(values) if values else 0.0
        print(
            f"[{category}] "
            f"passed={stats['passed']}/{stats['count']} "
            f"avg_request_ms={avg_request_ms:.2f}"
        )

    print("\nSlowest cases")
    print("------------")
    for item in summary["slowest_cases"]:
        print(
            f"[{item['id']}] "
            f"category={item.get('category', '-')} "
            f"request_ms={item.get('request_ms', '-')}"
        )

    print("\nRetrieval")
    print("---------")
    if not any(stats["total"] for stats in summary["retrieval_stats"].values()):
        print("no expected_doc_ids / expected_chunk_ids configured")
    else:
        for field, stats in summary["retrieval_stats"].items():
            print(
                f"{field}={stats['rate']:.1f}% " f"({stats['hits']}/{stats['total']})"
            )
    fallback_stats = summary["fallback_stats"]
    if fallback_stats["total"]:
        print(
            f"fallback_accuracy={fallback_stats['rate']:.1f}% "
            f"({fallback_stats['hits']}/{fallback_stats['total']})"
        )

    tool_safety_stats = summary.get("tool_safety_stats") or {}
    if any(stats["total"] for stats in tool_safety_stats.values()):
        print("\nTool safety")
        print("-----------")
        for field, stats in tool_safety_stats.items():
            if not stats["total"]:
                continue
            print(
                f"{field}={stats['rate']:.1f}% " f"({stats['hits']}/{stats['total']})"
            )

    workflow_stats = summary.get("workflow_stats") or {}
    if any(stats["total"] for stats in workflow_stats.values()):
        print("\nWorkflow")
        print("--------")
        for field, stats in workflow_stats.items():
            if not stats["total"]:
                continue
            print(
                f"{field}={stats['rate']:.1f}% " f"({stats['hits']}/{stats['total']})"
            )

    print("\nFailures")
    print("--------")
    if not summary["failed_items"]:
        print("none")
        return

    for item in summary["failed_items"]:
        print(
            f"[{item['id']}] " f"assertion_detail={item.get('assertion_detail', '-')}"
        )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_output(results: list[dict], path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote JSON results to {path}")


def write_csv_output(results: list[dict], path: Path) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "id",
        "status_code",
        "category",
        "expected_route",
        "actual_route",
        "doc_used",
        "memory_used",
        "request_ms",
        "rag_ms",
        "memory_ms",
        "answer_len",
        "quality",
        "expected_doc_ids",
        "expected_chunk_ids",
        "citation_count",
        "citation_doc_ids",
        "citation_chunk_ids",
        "source_doc_ids",
        "used_chunk_ids",
        "citation_expected_doc_coverage",
        "citation_all_expected_docs_hit",
        "citation_hit",
        "answer_citation_refs",
        "answer_citation_count",
        "answer_has_citation",
        "citation_refs_valid",
        "invalid_citation_refs",
        "unused_citation_refs",
        "top_k_hit",
        "filtered_hit",
        "rerank_hit",
        "merged_hit",
        "retrieval_failure_stage",
        "top_k_doc_ids",
        "filtered_doc_ids",
        "rerank_doc_ids",
        "merged_doc_ids",
        "top_k_chunk_ids",
        "filtered_chunk_ids",
        "rerank_chunk_ids",
        "merged_chunk_ids",
        "fallback_accuracy",
        "dense_count",
        "lexical_count",
        "hybrid_count",
        "filtered_count",
        "rerank_count",
        "merged_count",
        "plan_schema_pass_rate",
        "workflow_success_rate",
        "confirmation_bridge_rate",
        "assertion",
        "assertion_detail",
        "answer",
        "detail",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"Wrote CSV results to {path}")


def maybe_write_outputs(results: list[dict]) -> None:
    json_path = os.getenv(EVAL_OUTPUT_JSON_ENV, "").strip()
    csv_path = os.getenv(EVAL_OUTPUT_CSV_ENV, "").strip()

    # 结果落盘是实验沉淀的基础，默认不强制输出文件，需要时用环境变量开启。
    if json_path:
        write_json_output(results, Path(json_path))
    if csv_path:
        write_csv_output(results, Path(csv_path))


def get_http_timeout() -> float:
    value = os.getenv(EVAL_HTTP_TIMEOUT_ENV, "").strip()
    if not value:
        return EVAL_HTTP_TIMEOUT_SECONDS
    return float(value)


def build_client():
    base_url = os.getenv(EVAL_BASE_URL_ENV, "").strip()
    if base_url:
        # 优先支持直接请求已启动的本地服务，适合长时评测和观察真实日志。
        # 创作型/长文本 case 往往明显慢于普通 QA，因此把超时做成可配置项。
        return httpx.Client(base_url=base_url, timeout=get_http_timeout())

    api.clear_session_store()
    return TestClient(api.app)


def main() -> None:
    cases = filter_cases(load_cases())
    client = build_client()
    results = []

    try:
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{total}] running {case['id']} ...", flush=True)
            results.append(run_case(client, case))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    print_table(results)
    print_summary(results)
    print_answer_details(results)
    maybe_write_outputs(results)


if __name__ == "__main__":
    main()
