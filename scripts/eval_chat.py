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

import app.api as api
from app.constants.policies import INSUFFICIENT_KNOWLEDGE_ANSWER

CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
EVAL_CONVERSATION_HISTORY_PATH = "EVAL_CONVERSATION_HISTORY_PATH"
CITATION_REF_PATTERN = re.compile(r"\[(\d+)\]")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def filter_cases(cases: list[dict]) -> list[dict]:
    case_ids = os.getenv("EVAL_CASE_IDS", "").strip()
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


def hits_contain_expected(
    hits: list[dict],
    *,
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
) -> bool | str:
    """判断某个检索阶段是否命中预期文档/切片。

    没有配置 expected_* 时返回 "-"，表示该 case 不参与 retrieval hit 统计。
    """

    expected_ids = set(expected_doc_ids + expected_chunk_ids)
    if not expected_ids:
        return "-"

    for hit in hits:
        if collect_hit_identifiers(hit) & expected_ids:
            return True
    return False


def bool_metric(value: bool | str) -> str:
    if value == "-":
        return "-"
    return "true" if value else "false"


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
        "answer_citation_refs": ",".join(answer_refs) or "-",
        "answer_citation_count": len(answer_refs),
        "answer_has_citation": bool_metric(bool(answer_refs)) if doc_used else "-",
        "citation_refs_valid": (
            bool_metric(not invalid_refs) if answer_refs or available_refs else "-"
        ),
        "invalid_citation_refs": ",".join(invalid_refs) or "-",
        "unused_citation_refs": ",".join(missing_refs) or "-",
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
    citations = rag_debug.get("citations") or []
    citation_doc_ids = [
        str(citation.get("doc_id", ""))
        for citation in citations
        if citation.get("doc_id")
    ]
    citation_coverage, citation_all_hit = calculate_expected_doc_coverage(
        expected_doc_ids=expected_doc_ids,
        citation_doc_ids=citation_doc_ids,
    )

    metrics = {
        "expected_doc_ids": ",".join(expected_doc_ids) or "-",
        "expected_chunk_ids": ",".join(expected_chunk_ids) or "-",
        "citation_count": len(citations),
        "citation_doc_ids": ",".join(citation_doc_ids) or "-",
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
            rag_debug.get("top_docs") or [],
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "filtered_hit": hits_contain_expected(
            rag_debug.get("filtered_docs") or [],
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "rerank_hit": hits_contain_expected(
            rag_debug.get("post_rerank_docs") or [],
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
        ),
        "merged_hit": hits_contain_expected(
            rag_debug.get("merged_docs") or [],
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
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


def post_chat(client, session_id: str, message: str) -> dict:
    payload = {
        "session_id": session_id,
        "message": message,
        "debug": True,
    }
    conversation_history_path = os.getenv(EVAL_CONVERSATION_HISTORY_PATH, "").strip()
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


def setup_knowledge_imports(client, case: dict) -> dict[str, str]:
    """执行 case 级知识导入，并返回 import alias -> doc_id 映射。

    这样 eval 不需要把内容 hash 生成的 doc_id 写死在 eval_cases.json 里。
    """

    alias_to_doc_id: dict[str, str] = {}
    for index, import_payload in enumerate(case.get("setup_knowledge_imports", [])):
        payload = post_knowledge_import(client, import_payload)
        alias = str(import_payload.get("alias") or f"import_{index}")
        alias_to_doc_id[alias] = str(payload.get("doc_id", ""))
    return alias_to_doc_id


def resolve_expected_doc_ids(case: dict, alias_to_doc_id: dict[str, str]) -> dict:
    """把 expected_import_aliases 解析成实际 doc_id，返回 case 副本。"""

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
    return resolved


def run_case(client, case: dict) -> dict:
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

        response_data = post_chat(client, case["session_id"], case["message"])
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
    retrieval_eval = build_retrieval_eval(case, debug_nodes, answer)

    return {
        "id": case["id"],
        "category": case.get("category", "-"),
        "status_code": response_data["status_code"],
        "expected_route": case["expected_route"],
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

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total * 100) if total else 0.0,
        "failed_items": failed_items,
        "category_stats": category_stats,
        "slowest_cases": slowest_cases,
        "retrieval_stats": retrieval_stats,
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
        "dense_count",
        "lexical_count",
        "hybrid_count",
        "filtered_count",
        "rerank_count",
        "merged_count",
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
    json_path = os.getenv("EVAL_OUTPUT_JSON", "").strip()
    csv_path = os.getenv("EVAL_OUTPUT_CSV", "").strip()

    # 结果落盘是实验沉淀的基础，默认不强制输出文件，需要时用环境变量开启。
    if json_path:
        write_json_output(results, Path(json_path))
    if csv_path:
        write_csv_output(results, Path(csv_path))


def get_http_timeout() -> float:
    value = os.getenv("EVAL_HTTP_TIMEOUT", "").strip()
    if not value:
        return 120.0
    return float(value)


def build_client():
    base_url = os.getenv("EVAL_BASE_URL", "").strip()
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
