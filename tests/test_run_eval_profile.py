from pathlib import Path

from scripts.run_eval_profile import (
    build_chroma_dir,
    build_history_path,
    build_knowledge_sqlite_path,
    build_manifest_path,
    cleanup_isolated_chroma_dir,
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


def test_cleanup_isolated_chroma_dir_removes_directory(tmp_path) -> None:
    target = tmp_path / "iso.chroma"
    target.mkdir()
    (target / "data_level0.bin").write_bytes(b"x" * 8)

    assert cleanup_isolated_chroma_dir(target) is True
    assert not target.exists()


def test_cleanup_isolated_chroma_dir_returns_false_when_missing(tmp_path) -> None:
    assert cleanup_isolated_chroma_dir(tmp_path / "missing") is False
