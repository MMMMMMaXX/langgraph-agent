# ./.venv/bin/python scripts/run_eval_profile.py --profile baseline --base-url http://127.0.0.1:8000 --case-ids aria_definition,virtual_list_definition
# ./.venv/bin/python scripts/run_eval_profile.py --profile concise --base-url http://127.0.0.1:8000 --case-ids aria_definition,virtual_list_definition
# ./.venv/bin/python scripts/compare_eval_runs.py outputs/eval_runs/<baseline>.json outputs/eval_runs/<concise>.json

# EVAL_BASE_URL=http://127.0.0.1:8000 ./.venv/bin/python scripts/run_eval_profile.py --profile baseline

# EVAL_BASE_URL=http://127.0.0.1:8000 \
# EVAL_CASE_IDS=history_summary_after_two_turns \
# ./.venv/bin/python scripts/run_eval_profile.py --profile baseline

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = Path(__file__).resolve().parent / "eval_profiles.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "eval_runs"


def load_profiles() -> list[dict]:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def get_profile(profile_name: str) -> dict:
    for profile in load_profiles():
        if profile["name"] == profile_name:
            return profile
    raise ValueError(f"unknown profile: {profile_name}")


def build_output_paths(profile_name: str, output_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"{timestamp}-{profile_name}.json"
    csv_path = output_dir / f"{timestamp}-{profile_name}.csv"
    return json_path, csv_path


def build_manifest_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.manifest.json")


def build_history_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.conversation_history.sqlite3")


def build_env(profile: dict, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    # profile 内定义的是一组“可复现的实验参数”，这里统一注入子进程，
    # 避免每次手敲一长串环境变量，后续也便于沉淀成标准实验档案。
    for key, value in profile.get("env", {}).items():
        env[key] = str(value)

    if args.base_url:
        env["EVAL_BASE_URL"] = args.base_url
    if args.case_ids:
        env["EVAL_CASE_IDS"] = args.case_ids

    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval_chat.py with a named experiment profile."
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        help="profile name defined in scripts/eval_profiles.json",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("EVAL_BASE_URL", "").strip(),
        help="optional live API base url, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--case-ids",
        default=os.getenv("EVAL_CASE_IDS", "").strip(),
        help="optional comma-separated case ids",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated json/csv results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = build_output_paths(args.profile, output_dir)
    manifest_path = build_manifest_path(json_path)
    history_path = build_history_path(json_path)

    env = build_env(profile, args)
    env["EVAL_OUTPUT_JSON"] = str(json_path)
    env["EVAL_OUTPUT_CSV"] = str(csv_path)
    env["EVAL_CONVERSATION_HISTORY_PATH"] = str(history_path)
    env["CONVERSATION_HISTORY_PATH"] = str(history_path)
    env["CONVERSATION_HISTORY_BACKEND"] = "sqlite"
    env["CONVERSATION_HISTORY_SQLITE_PATH"] = str(history_path)

    manifest = {
        "profile": profile["name"],
        "description": profile.get("description", ""),
        "env": profile.get("env", {}),
        "case_ids": args.case_ids,
        "base_url": args.base_url,
        "json_output": str(json_path),
        "csv_output": str(csv_path),
        "conversation_history_output": str(history_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Profile: {profile['name']}")
    print(f"Description: {profile.get('description', '-')}")
    print("Applied env:")
    for key, value in profile.get("env", {}).items():
        print(f"  {key}={value}")
    if args.case_ids:
        print(f"Case filter: {args.case_ids}")
    if args.base_url:
        print(f"Base URL: {args.base_url}")
    print(f"JSON output: {json_path}")
    print(f"CSV output: {csv_path}")
    print(f"Conversation history: {history_path}")
    print(f"Manifest: {manifest_path}")
    print("", flush=True)

    command = [sys.executable, str(Path(__file__).resolve().parent / "eval_chat.py")]
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
