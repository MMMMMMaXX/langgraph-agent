# ./.venv/bin/python scripts/compare_eval_runs.py \
#   outputs/eval_runs/<baseline>.json \
#   outputs/eval_runs/<candidate>.json \
#   --markdown-out outputs/reports/<report-name>.md

# ./.venv/bin/python scripts/compare_eval_runs.py \
#   outputs/eval_runs/20260416-174313-baseline.json \
#   outputs/eval_runs/20260416-175610-concise.json \
#   --markdown-out outputs/reports/history-summary-optimization-041601.md


import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_id(results: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in results}


def to_float(value) -> float | None:
    if value in (None, "", "-"):
        return None
    return float(value)


def diff_ms(candidate_value, baseline_value) -> str:
    candidate = to_float(candidate_value)
    baseline = to_float(baseline_value)
    if candidate is None or baseline is None:
        return "-"
    return f"{candidate - baseline:.2f}"


def diff_ms_float(candidate_value, baseline_value) -> float | None:
    candidate = to_float(candidate_value)
    baseline = to_float(baseline_value)
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def print_table(rows: list[dict]) -> None:
    headers = [
        "id",
        "baseline_quality",
        "candidate_quality",
        "baseline_ms",
        "candidate_ms",
        "delta_ms",
        "baseline_len",
        "candidate_len",
    ]
    values = [[str(row.get(header, "")) for header in headers] for row in rows]
    widths = []
    for index, header in enumerate(headers):
        max_value_width = max((len(row[index]) for row in values), default=0)
        widths.append(max(len(header), max_value_width))

    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in values:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item.get("assertion") == "pass")
    values = [
        to_float(item.get("request_ms"))
        for item in results
        if to_float(item.get("request_ms")) is not None
    ]
    avg_request_ms = sum(values) / len(values) if values else None
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total * 100) if total else 0.0,
        "avg_request_ms": avg_request_ms,
    }


def build_rows(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    baseline_index = index_by_id(baseline)
    candidate_index = index_by_id(candidate)
    case_ids = sorted(set(baseline_index) | set(candidate_index))
    rows = []

    for case_id in case_ids:
        baseline_case = baseline_index.get(case_id, {})
        candidate_case = candidate_index.get(case_id, {})
        rows.append(
            {
                "id": case_id,
                "category": candidate_case.get("category")
                or baseline_case.get("category", "-"),
                "baseline_assertion": baseline_case.get("assertion", "-"),
                "candidate_assertion": candidate_case.get("assertion", "-"),
                "baseline_quality": baseline_case.get("quality", "-"),
                "candidate_quality": candidate_case.get("quality", "-"),
                "baseline_ms": baseline_case.get("request_ms", "-"),
                "candidate_ms": candidate_case.get("request_ms", "-"),
                "delta_ms": diff_ms(
                    candidate_case.get("request_ms"),
                    baseline_case.get("request_ms"),
                ),
                "baseline_len": baseline_case.get("answer_len", "-"),
                "candidate_len": candidate_case.get("answer_len", "-"),
                "delta_ms_value": diff_ms_float(
                    candidate_case.get("request_ms"),
                    baseline_case.get("request_ms"),
                ),
            }
        )

    return rows


def print_answer_diff(
    baseline: list[dict],
    candidate: list[dict],
    show_all: bool,
) -> None:
    baseline_index = index_by_id(baseline)
    candidate_index = index_by_id(candidate)
    case_ids = sorted(set(baseline_index) | set(candidate_index))

    print("\nAnswer comparison")
    print("-----------------")
    for case_id in case_ids:
        baseline_answer = baseline_index.get(case_id, {}).get("answer", "")
        candidate_answer = candidate_index.get(case_id, {}).get("answer", "")

        if not show_all and baseline_answer == candidate_answer:
            continue

        print(f"[{case_id}]")
        print(f"baseline : {baseline_answer or '-'}")
        print(f"candidate: {candidate_answer or '-'}")
        print("")


def build_markdown_report(
    baseline_path: Path,
    candidate_path: Path,
    baseline: list[dict],
    candidate: list[dict],
    rows: list[dict],
) -> str:
    baseline_summary = summarize_results(baseline)
    candidate_summary = summarize_results(candidate)

    lines = [
        "# Eval Comparison Report",
        "",
        f"- Baseline: `{baseline_path}`",
        f"- Candidate: `{candidate_path}`",
        "",
        "## Overview",
        "",
        "| Metric | Baseline | Candidate |",
        "| --- | ---: | ---: |",
        f"| Pass rate | {baseline_summary['pass_rate']:.1f}% ({baseline_summary['passed']}/{baseline_summary['total']}) | {candidate_summary['pass_rate']:.1f}% ({candidate_summary['passed']}/{candidate_summary['total']}) |",
        f"| Avg request ms | {format_md_ms(baseline_summary['avg_request_ms'])} | {format_md_ms(candidate_summary['avg_request_ms'])} |",
        "",
        "## Case Comparison",
        "",
        "| Case | Category | Baseline Assertion | Candidate Assertion | Baseline ms | Candidate ms | Delta ms | Baseline Len | Candidate Len |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row.get("category", "-"),
                    row.get("baseline_assertion", "-"),
                    row.get("candidate_assertion", "-"),
                    str(row.get("baseline_ms", "-")),
                    str(row.get("candidate_ms", "-")),
                    str(row.get("delta_ms", "-")),
                    str(row.get("baseline_len", "-")),
                    str(row.get("candidate_len", "-")),
                ]
            )
            + " |"
        )

    changed_answers = build_changed_answers_section(baseline, candidate)
    if changed_answers:
        lines.extend(
            [
                "",
                "## Changed Answers",
                "",
            ]
        )
        lines.extend(changed_answers)

    return "\n".join(lines) + "\n"


def format_md_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def build_changed_answers_section(
    baseline: list[dict],
    candidate: list[dict],
) -> list[str]:
    baseline_index = index_by_id(baseline)
    candidate_index = index_by_id(candidate)
    case_ids = sorted(set(baseline_index) | set(candidate_index))
    lines: list[str] = []

    for case_id in case_ids:
        baseline_answer = baseline_index.get(case_id, {}).get("answer", "")
        candidate_answer = candidate_index.get(case_id, {}).get("answer", "")
        if baseline_answer == candidate_answer:
            continue

        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Baseline: {baseline_answer or '-'}",
                f"- Candidate: {candidate_answer or '-'}",
                "",
            ]
        )

    return lines


def maybe_write_report(report_markdown: str, output_path: str) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown, encoding="utf-8")
    print(f"\nWrote Markdown report to {path}")


def parse_case_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def evaluate_gate(
    baseline: list[dict],
    candidate: list[dict],
    rows: list[dict],
    max_pass_rate_drop: float,
    max_avg_ms_increase: float,
    max_case_ms_increase: float,
    required_case_ids: set[str],
) -> list[str]:
    failures: list[str] = []
    baseline_summary = summarize_results(baseline)
    candidate_summary = summarize_results(candidate)

    pass_rate_drop = baseline_summary["pass_rate"] - candidate_summary["pass_rate"]
    if pass_rate_drop > max_pass_rate_drop:
        failures.append(
            f"pass rate dropped by {pass_rate_drop:.2f} percentage points "
            f"(allowed {max_pass_rate_drop:.2f})"
        )

    baseline_avg = baseline_summary["avg_request_ms"]
    candidate_avg = candidate_summary["avg_request_ms"]
    if baseline_avg is not None and candidate_avg is not None:
        avg_increase = candidate_avg - baseline_avg
        if avg_increase > max_avg_ms_increase:
            failures.append(
                f"average request_ms increased by {avg_increase:.2f}ms "
                f"(allowed {max_avg_ms_increase:.2f}ms)"
            )

    for row in rows:
        delta_ms_value = row.get("delta_ms_value")
        if delta_ms_value is not None and delta_ms_value > max_case_ms_increase:
            failures.append(
                f"case {row['id']} regressed by {delta_ms_value:.2f}ms "
                f"(allowed {max_case_ms_increase:.2f}ms)"
            )

    required_rows = [row for row in rows if row["id"] in required_case_ids]
    missing_required = required_case_ids - {row["id"] for row in required_rows}
    for case_id in sorted(missing_required):
        failures.append(f"required gate case missing: {case_id}")

    for row in required_rows:
        if row.get("candidate_assertion") != "pass":
            failures.append(f"required gate case failed: {row['id']}")

    return failures


def print_gate_result(failures: list[str]) -> None:
    print("\nRegression Gate")
    print("---------------")
    if not failures:
        print("PASS")
        return

    print("FAIL")
    for item in failures:
        print(f"- {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two eval result json files generated by eval_chat.py."
    )
    parser.add_argument("baseline", help="baseline eval json path")
    parser.add_argument("candidate", help="candidate eval json path")
    parser.add_argument(
        "--show-all-answers",
        action="store_true",
        help="print all answers, not only changed ones",
    )
    parser.add_argument(
        "--markdown-out",
        default="",
        help="optional markdown report output path",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="enable regression gate checks and exit non-zero on failure",
    )
    parser.add_argument(
        "--max-pass-rate-drop",
        type=float,
        default=0.0,
        help="allowed pass rate drop in percentage points when gate is enabled",
    )
    parser.add_argument(
        "--max-avg-ms-increase",
        type=float,
        default=500.0,
        help="allowed average request_ms increase when gate is enabled",
    )
    parser.add_argument(
        "--max-case-ms-increase",
        type=float,
        default=1000.0,
        help="allowed single-case request_ms increase when gate is enabled",
    )
    parser.add_argument(
        "--required-case-ids",
        default="",
        help="comma-separated case ids that must remain assertion=pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_results(Path(args.baseline))
    candidate = load_results(Path(args.candidate))
    rows = build_rows(baseline, candidate)
    print_table(rows)
    print_answer_diff(baseline, candidate, show_all=args.show_all_answers)
    report_markdown = build_markdown_report(
        Path(args.baseline),
        Path(args.candidate),
        baseline,
        candidate,
        rows,
    )
    maybe_write_report(report_markdown, args.markdown_out)

    if not args.gate:
        return

    failures = evaluate_gate(
        baseline,
        candidate,
        rows,
        max_pass_rate_drop=args.max_pass_rate_drop,
        max_avg_ms_increase=args.max_avg_ms_increase,
        max_case_ms_increase=args.max_case_ms_increase,
        required_case_ids=parse_case_ids(args.required_case_ids),
    )
    print_gate_result(failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
