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
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.constants.eval import (
    CHROMA_PERSIST_DIR_ENV,
    CONVERSATION_HISTORY_BACKEND_ENV,
    CONVERSATION_HISTORY_BACKEND_SQLITE,
    CONVERSATION_HISTORY_PATH_ENV,
    CONVERSATION_HISTORY_SQLITE_PATH_ENV,
    EVAL_BASE_URL_ENV,
    EVAL_CASE_IDS_ENV,
    EVAL_CHROMA_DIR_SUFFIX,
    EVAL_CHROMA_KEEP_REASON_EXTERNAL,
    EVAL_CHROMA_KEEP_REASON_FAILURE,
    EVAL_CHROMA_KEEP_REASON_FLAG,
    EVAL_CONVERSATION_HISTORY_PATH_ENV,
    EVAL_CONVERSATION_HISTORY_SUFFIX,
    EVAL_KEEP_CHROMA_ENV,
    EVAL_KNOWLEDGE_SQLITE_SUFFIX,
    EVAL_MANIFEST_CHROMA_AUTO_CREATED_KEY,
    EVAL_MANIFEST_CHROMA_CLEANED_KEY,
    EVAL_MANIFEST_CHROMA_KEEP_REASON_KEY,
    EVAL_MANIFEST_CHROMA_PERSIST_DIR_KEY,
    EVAL_MANIFEST_CHROMA_SIZE_AFTER_KEY,
    EVAL_MANIFEST_CHROMA_SIZE_BEFORE_KEY,
    EVAL_MANIFEST_CHROMA_SIZE_BYTES_KEY,
    EVAL_MANIFEST_RUN_STATUS_KEY,
    EVAL_MANIFEST_SUFFIX,
    EVAL_OUTPUT_CSV_ENV,
    EVAL_OUTPUT_JSON_ENV,
    KNOWLEDGE_BASE_SQLITE_PATH_ENV,
)

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
    # 与 eval_chat.py 写出的 manifest 文件名保持一致：subprocess 退出后
    # wrapper 会把自身的 profile/env 元信息合并进 eval_chat 写出的 chroma
    # 生命周期记录，最终输出统一的 manifest，避免出现两套字段。
    return json_path.with_name(f"{json_path.stem}{EVAL_MANIFEST_SUFFIX}")


def _compute_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def cleanup_isolated_chroma_dir(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        return False


def build_history_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}{EVAL_CONVERSATION_HISTORY_SUFFIX}")


def build_knowledge_sqlite_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}{EVAL_KNOWLEDGE_SQLITE_SUFFIX}")


def build_chroma_dir(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}{EVAL_CHROMA_DIR_SUFFIX}")


def build_env(profile: dict, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    # profile 内定义的是一组“可复现的实验参数”，这里统一注入子进程，
    # 避免每次手敲一长串环境变量，后续也便于沉淀成标准实验档案。
    for key, value in profile.get("env", {}).items():
        env[key] = str(value)

    if args.base_url:
        env[EVAL_BASE_URL_ENV] = args.base_url
    if args.case_ids:
        env[EVAL_CASE_IDS_ENV] = args.case_ids

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
        default=os.getenv(EVAL_BASE_URL_ENV, "").strip(),
        help="optional live API base url, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--case-ids",
        default=os.getenv(EVAL_CASE_IDS_ENV, "").strip(),
        help="optional comma-separated case ids",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated json/csv results",
    )
    parser.add_argument(
        "--keep-chroma",
        action="store_true",
        default=os.getenv(EVAL_KEEP_CHROMA_ENV, "").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "保留隔离模式下 wrapper 创建的 Chroma 目录（默认成功后清理）。"
            "失败 / 异常退出会自动保留，不依赖该 flag。"
        ),
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
    knowledge_sqlite_path = build_knowledge_sqlite_path(json_path)
    chroma_dir = build_chroma_dir(json_path)
    knowledge_isolated = not bool(args.base_url)

    env = build_env(profile, args)
    env[EVAL_OUTPUT_JSON_ENV] = str(json_path)
    env[EVAL_OUTPUT_CSV_ENV] = str(csv_path)
    env[EVAL_CONVERSATION_HISTORY_PATH_ENV] = str(history_path)
    env[CONVERSATION_HISTORY_PATH_ENV] = str(history_path)
    env[CONVERSATION_HISTORY_BACKEND_ENV] = CONVERSATION_HISTORY_BACKEND_SQLITE
    env[CONVERSATION_HISTORY_SQLITE_PATH_ENV] = str(history_path)

    if knowledge_isolated:
        # 只在进程内 eval 模式注入隔离知识库路径。若连接已启动 API，
        # 子进程环境变量无法改变那个服务已加载的知识库配置，强行打印
        # “已隔离”反而会误导排查。
        env[KNOWLEDGE_BASE_SQLITE_PATH_ENV] = str(knowledge_sqlite_path)
        env[CHROMA_PERSIST_DIR_ENV] = str(chroma_dir)

    if args.keep_chroma:
        # 把 keep-chroma 透传给 eval_chat：在 base-url 模式或 eval_chat
        # 自行创建 chroma 目录的场景下也能保留，便于排查。
        env[EVAL_KEEP_CHROMA_ENV] = "1"

    profile_manifest_fields = {
        "profile": profile["name"],
        "description": profile.get("description", ""),
        "env": profile.get("env", {}),
        "case_ids": args.case_ids,
        "base_url": args.base_url,
        "json_output": str(json_path),
        "csv_output": str(csv_path),
        "conversation_history_output": str(history_path),
        "knowledge_isolated": knowledge_isolated,
        "knowledge_sqlite_output": (
            str(knowledge_sqlite_path) if knowledge_isolated else ""
        ),
        "chroma_output": str(chroma_dir) if knowledge_isolated else "",
        "knowledge_isolation_note": (
            "enabled for in-process eval"
            if knowledge_isolated
            else "disabled because base_url mode uses an already running API process"
        ),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(
        json.dumps(profile_manifest_fields, ensure_ascii=False, indent=2),
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
    if knowledge_isolated:
        print(f"Knowledge SQLite: {knowledge_sqlite_path}")
        print(f"Chroma dir: {chroma_dir}")
    else:
        print("Knowledge isolation: disabled for base-url mode")
    print(f"Manifest: {manifest_path}")
    print("", flush=True)

    command = [sys.executable, str(Path(__file__).resolve().parent / "eval_chat.py")]
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)

    # eval_chat 在 finally 中会把 chroma 生命周期 + run_status 写到同一个
    # manifest 文件，覆盖了 wrapper 之前写入的 profile 字段。这里读回来再
    # 把 wrapper 的字段合并进去，输出统一的最终 manifest。
    eval_chat_manifest: dict = {}
    if manifest_path.exists():
        try:
            eval_chat_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            eval_chat_manifest = {}

    # eval_chat 退出码 0 仅代表进程正常结束；case 断言失败也算 returncode==0。
    # 真正的 run-level 成功要看 eval_chat 自己写出的 run_status 字段。
    run_succeeded = (
        completed.returncode == 0
        and eval_chat_manifest.get(EVAL_MANIFEST_RUN_STATUS_KEY) == "success"
    )
    wrapper_chroma_cleaned = False
    wrapper_chroma_keep_reason: str | None = None
    chroma_size_before = (
        _compute_dir_size_bytes(chroma_dir) if knowledge_isolated else 0
    )
    if knowledge_isolated:
        # wrapper 自己创建并管理隔离 chroma 目录，eval_chat 只会把它标成
        # "external_persist_dir" 不动。这里按照成功 + 未指定保留 -> 清理的
        # 策略由 wrapper 兜底，避免重复运行后磁盘被吃满（参见 PR-6 设计）。
        if not run_succeeded:
            wrapper_chroma_keep_reason = EVAL_CHROMA_KEEP_REASON_FAILURE
        elif args.keep_chroma:
            wrapper_chroma_keep_reason = EVAL_CHROMA_KEEP_REASON_FLAG
        else:
            wrapper_chroma_cleaned = cleanup_isolated_chroma_dir(chroma_dir)
        if (
            wrapper_chroma_keep_reason is None
            and not wrapper_chroma_cleaned
            and chroma_dir.exists()
        ):
            # 清理失败也保留，标注外部持久化目录便于人工介入。
            wrapper_chroma_keep_reason = EVAL_CHROMA_KEEP_REASON_EXTERNAL
    chroma_size_after = _compute_dir_size_bytes(chroma_dir) if knowledge_isolated else 0

    final_manifest = {**profile_manifest_fields, **eval_chat_manifest}
    final_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    final_manifest["return_code"] = completed.returncode
    final_manifest["keep_chroma_flag"] = bool(args.keep_chroma)
    if knowledge_isolated:
        final_manifest["wrapper_chroma_cleaned"] = wrapper_chroma_cleaned
        final_manifest["wrapper_chroma_keep_reason"] = wrapper_chroma_keep_reason
        # eval_chat 只看到一个外部目录；用 wrapper 视角覆写关键字段，保证
        # 最终 manifest 反映 wrapper 真实做的清理动作。
        final_manifest[EVAL_MANIFEST_CHROMA_PERSIST_DIR_KEY] = str(chroma_dir)
        final_manifest[EVAL_MANIFEST_CHROMA_AUTO_CREATED_KEY] = False
        final_manifest[EVAL_MANIFEST_CHROMA_CLEANED_KEY] = wrapper_chroma_cleaned
        final_manifest[EVAL_MANIFEST_CHROMA_KEEP_REASON_KEY] = (
            wrapper_chroma_keep_reason
        )
        # 同时记录清理前/后大小：清理成功时 after=0，但 before 仍能反映
        # 这次 run 真实写入的 Chroma 体量，方便事后复核 chunks↔磁盘比例。
        final_manifest[EVAL_MANIFEST_CHROMA_SIZE_BEFORE_KEY] = chroma_size_before
        final_manifest[EVAL_MANIFEST_CHROMA_SIZE_AFTER_KEY] = chroma_size_after
        final_manifest[EVAL_MANIFEST_CHROMA_SIZE_BYTES_KEY] = chroma_size_before

    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if knowledge_isolated:
        if wrapper_chroma_cleaned:
            print(f"Wrapper cleaned chroma dir: {chroma_dir}")
        else:
            print(
                "Wrapper kept chroma dir "
                f"({wrapper_chroma_keep_reason or 'unknown'}): {chroma_dir}"
            )
    chroma_status = final_manifest.get(EVAL_MANIFEST_RUN_STATUS_KEY)
    if chroma_status:
        print(f"Run status: {chroma_status}")
    print(f"Manifest finalized: {manifest_path}")

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
