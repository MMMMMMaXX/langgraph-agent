from pathlib import Path

from scripts.run_eval_profile import (
    build_chroma_dir,
    build_history_path,
    build_knowledge_sqlite_path,
    build_manifest_path,
)


def test_eval_profile_derives_isolated_storage_paths_from_json_output() -> None:
    json_path = Path("outputs/eval_runs/20260506-120000-baseline.json")

    assert build_manifest_path(json_path) == Path(
        "outputs/eval_runs/20260506-120000-baseline.manifest.json"
    )
    assert build_history_path(json_path) == Path(
        "outputs/eval_runs/20260506-120000-baseline.conversation_history.sqlite3"
    )
    assert build_knowledge_sqlite_path(json_path) == Path(
        "outputs/eval_runs/20260506-120000-baseline.knowledge.sqlite3"
    )
    assert build_chroma_dir(json_path) == Path(
        "outputs/eval_runs/20260506-120000-baseline.chroma"
    )
