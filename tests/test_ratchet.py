"""SPEC3 D: the ratchet never lets the skill get worse without saying why."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals"))
import ratchet  # noqa: E402


def _runs(tmp: Path, results: dict, judge: dict | None = None, driver: str = "lfm3") -> Path:
    d = tmp / "runs"
    for i, (spec, ok) in enumerate(results.items()):
        r = d / f"{spec}__s{i}"
        r.mkdir(parents=True)
        (r / "log.json").write_text(json.dumps({"spec": spec, "suite": "tuning", "card": "S", "driver": driver,
                                                "pass": ok, "run_id": r.name}))
        if judge and spec in judge:
            (r / "judge.json").write_text(json.dumps({"score": judge[spec], "faults": []}))
    return d


def _floor(tmp: Path, floor: dict) -> Path:
    p = tmp / "floor.json"
    p.write_text(json.dumps(floor))
    return p


def test_a_run_below_the_floor_exits_1_and_names_the_spec(tmp_path, capsys):
    floor = _floor(tmp_path, {"tuning/S": {"rate": 1.0, "judge": None, "total": 2, "passing": ["t1", "t2"]}})
    before = floor.read_text()
    assert ratchet.check_runs([_runs(tmp_path, {"t1": True, "t2": False})], floor) == 1
    out = capsys.readouterr().out
    assert "BELOW THE FLOOR" in out and "t2 passed at the floor and fails now" in out
    assert floor.read_text() == before  # a floor is never lowered


def test_a_lower_judge_score_is_below_the_floor(tmp_path, capsys):
    floor = _floor(tmp_path, {"tuning/S": {"rate": 1.0, "judge": 8.0, "total": 1, "passing": ["t1"]}})
    assert ratchet.check_runs([_runs(tmp_path, {"t1": True}, judge={"t1": 6})], floor) == 1
    assert "judge score 6.00 is below the floor 8.00" in capsys.readouterr().out


def test_a_judge_score_a_point_under_the_floor_is_noise_not_a_regression(tmp_path, capsys):
    """One vote scoring 8 on a build that once scored 9 is the judge's noise: it must not go red,
    and it must not lower the floor either."""
    floor = _floor(tmp_path, {"tuning/S": {"rate": 1.0, "judge": 9.0, "total": 1, "passing": ["t1"]}})
    assert ratchet.check_runs([_runs(tmp_path, {"t1": True}, judge={"t1": 8})], floor) == 0
    assert json.loads(floor.read_text())["tuning/S"]["judge"] == 9.0


def test_a_run_above_the_floor_raises_it(tmp_path, capsys):
    floor = _floor(tmp_path, {"tuning/S": {"rate": 0.5, "judge": 6.0, "total": 2, "passing": ["t1"]}})
    assert ratchet.check_runs([_runs(tmp_path, {"t1": True, "t2": True}, judge={"t1": 9, "t2": 8})], floor) == 0
    assert "floor raised" in capsys.readouterr().out
    f = json.loads(floor.read_text())["tuning/S"]
    assert f["rate"] == 1.0 and f["judge"] == 8.5 and f["passing"] == ["t1", "t2"]


def test_the_next_driver_has_its_own_floor(tmp_path):
    floor = _floor(tmp_path, {"tuning/S": {"rate": 1.0, "judge": None, "total": 1, "passing": ["t1"]}})
    assert ratchet.check_runs([_runs(tmp_path, {"t1": False}, driver="next")], floor) == 0
    f = json.loads(floor.read_text())
    assert f["tuning/S"]["rate"] == 1.0 and f["tuning/next"]["rate"] == 0.0


def _git(root, *a):
    subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)


def _repo(tmp: Path) -> Path:
    root = tmp / "repo"
    (root / "evals" / "tuning").mkdir(parents=True)
    (root / "evals" / "tuning" / "t1.json").write_text("{}")
    (root / "evals" / "floor.json").write_text(json.dumps({"tuning/S": {"rate": 1.0, "judge": 8.0, "total": 1,
                                                                        "passing": ["t1"]}}))
    (root / "LOG.md").write_text("# Log\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return root


def test_check_git_fails_on_a_lowered_floor_without_a_log_line(tmp_path, capsys):
    root = _repo(tmp_path)
    assert ratchet.check_git("HEAD", root) == 0
    (root / "evals" / "floor.json").write_text(json.dumps({"tuning/S": {"rate": 0.5, "judge": 8.0, "total": 1,
                                                                        "passing": ["t1"]}}))
    assert ratchet.check_git("HEAD", root) == 1
    assert "floor tuning/S rate lowered from 1.0 to 0.5" in capsys.readouterr().out
    with (root / "LOG.md").open("a") as f:
        f.write("- tuning/S floor lowered: t1 now needs a picture the runner lacks\n")
    assert ratchet.check_git("HEAD", root) == 0
    assert "allowed by LOG.md" in capsys.readouterr().out


def test_check_git_fails_on_a_deleted_spec_without_a_log_line(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "evals" / "tuning" / "t1.json").unlink()
    assert ratchet.check_git("HEAD", root) == 1
    assert "spec evals/tuning/t1.json deleted" in capsys.readouterr().out
    with (root / "LOG.md").open("a") as f:
        f.write("- t1 retired: the recipe it tested is gone\n")
    assert ratchet.check_git("HEAD", root) == 0


def test_the_repo_floor_holds_every_guarded_group():
    f = json.loads((REPO / "evals" / "floor.json").read_text())
    assert f["tuning/S"]["rate"] == 1.0 and len(f["tuning/S"]["passing"]) == 7
    for d in ("tuning", "figure", "regressions"):
        assert list((REPO / "evals" / d).glob("*.json")), d


def test_run_appends_a_score_line_and_never_rewrites(tmp_path, monkeypatch):
    import run as evrun

    p = tmp_path / "scores.jsonl"
    p.write_text('{"old": 1}\n')
    monkeypatch.setattr(evrun, "SCORES", p)
    log = {"spec": "t1", "suite": "tuning", "card": "S", "driver": "next", "pass": True, "run_id": "r"}
    evrun.append_score(log)
    evrun.append_score({**log, "pass": False, "judge": 7})
    lines = p.read_text().splitlines()
    assert lines[0] == '{"old": 1}' and len(lines) == 3
    a, b = json.loads(lines[1]), json.loads(lines[2])
    assert (a["suite"], a["pass"], a["total"], a["judge"]) == ("tuning", 1, 1, None)
    assert b["pass"] == 0 and b["judge"] == 7 and b["sha"]
