"""Matrix entrypoint protection; children are mocked and never launched."""
import json

import pytest

from scripts import run_matrix


@pytest.mark.parametrize("tag", ("full", "memory-alignment"))
@pytest.mark.parametrize("skip_artifacts", (False, True))
def test_aligned_matrix_rejected_before_any_side_effect(tmp_path, monkeypatch, capsys, tag, skip_artifacts):
    root = tmp_path / "repository"
    config = root / "configs" / "aligned.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("memory_frame: agent_aligned\n", encoding="utf-8")
    # Reproduce the dangerous precondition: completed legacy runs and results.
    for seed in (11, 22, 33):
        path = root / "runs" / f"main-{seed}" / "final.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"training_steps": 5000000}), encoding="utf-8")
    protected = root / "results" / "full" / "summary.json"
    protected.parent.mkdir(parents=True)
    protected.write_text('{"original": true}', encoding="utf-8")
    before = {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}
    monkeypatch.setattr(run_matrix, "ROOT", root)

    def forbidden_command(*args, **kwargs):
        pytest.fail("Rejected alignment config must not launch training, evaluation, or export")

    monkeypatch.setattr(run_matrix, "command", forbidden_command)
    args = ["--config", "configs/aligned.yaml", "--tag", tag]
    if skip_artifacts:
        args.append("--skip-artifacts")
    with pytest.raises(SystemExit) as exc:
        run_matrix.main(args)
    assert exc.value.code == 2
    assert "run each seed directly" in capsys.readouterr().err
    after = {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}
    assert after == before


def test_legacy_matrix_still_reaches_training_dispatch(tmp_path, monkeypatch):
    config = tmp_path / "configs" / "legacy.yaml"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run_matrix, "ROOT", tmp_path)
    calls = []

    class DispatchReached(Exception):
        pass

    def stop_at_dispatch(args, logfile):
        calls.append(args)
        raise DispatchReached

    monkeypatch.setattr(run_matrix, "command", stop_at_dispatch)
    with pytest.raises(DispatchReached):
        run_matrix.main(["--config", "configs/legacy.yaml", "--variants", "main", "--seeds", "11"])
    assert len(calls) == 1
    assert calls[0][:3] == ["-m", "exploration", "train"]
