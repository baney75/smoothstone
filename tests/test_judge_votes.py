"""The judge vote: one noisy judge run must not flip a pass or a fail."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import judge_views  # noqa: E402


def _judge(seq):
    """A stand-in judge that returns the next verdict of seq: True names a boundaries fault."""
    it = iter(seq)
    calls = []

    def one(build, request, model):
        bad = next(it)
        calls.append(bad)
        faults = [{"panel": "boundaries row 3, column 7", "what": "notch"}] if bad else []
        return {"score": 6 if bad else 9, "faults": faults}

    return one, calls


def test_one_noisy_fault_among_three_does_not_fail_the_part():
    one, calls = _judge([True, False, False])
    res = judge_views.votes(Path("."), "req", one=one)
    assert res["pass"] and res["faults"] == [] and res["votes"] == "1 of 3 failed"
    assert len(calls) == 3 and res["score"] == 9


def test_two_runs_that_name_a_fault_fail_the_part():
    one, calls = _judge([False, True, True])
    res = judge_views.votes(Path("."), "req", one=one)
    assert not res["pass"] and len(res["faults"]) == 2 and res["votes"] == "2 of 3 failed"


def test_two_runs_that_agree_settle_it_without_a_third():
    one, calls = _judge([True, True, False])
    res = judge_views.votes(Path("."), "req", one=one)
    assert not res["pass"] and len(calls) == 2
    one, calls = _judge([False, False, True])
    assert judge_views.votes(Path("."), "req", one=one)["pass"] and len(calls) == 2


def test_a_fault_outside_the_named_pictures_does_not_count():
    def one(build, request, model):
        return {"score": 8, "faults": [{"panel": "views front", "what": "rough base"}]}

    res = judge_views.votes(Path("."), "req", ["boundaries"], one=one)
    assert res["pass"] and res["votes"] == "0 of 2 failed"


def test_a_judge_that_fails_still_fails_and_records_the_run(tmp_path, monkeypatch):
    """No claude, a missing picture or a reply with no JSON: the run is graded failed, with the
    reason, and judge.json says so; nothing escapes to skip log.json and the score line."""
    import run

    def broken(*a, **k):
        raise judge_views.JudgeFailed("claude exited 1: not signed in")

    monkeypatch.setattr(judge_views, "votes", broken)
    grade = {"pass": True, "why": ["all checks pass"]}
    out = run.judge_run(tmp_path, tmp_path / "job", "req", {"faults_in": ["boundaries"]}, grade)
    assert not out["pass"] and out["judge"] is None
    assert out["why"] == ["judge failed: claude exited 1: not signed in"]
    assert "not signed in" in (tmp_path / "judge.json").read_text()


def test_a_reply_with_no_verdict_is_a_judge_failure_not_an_exit(tmp_path, monkeypatch):
    import subprocess

    for name in ("views.png", "boundaries.png"):
        (tmp_path / name).write_bytes(b"png")

    class P:
        returncode, stdout, stderr = 0, '{"result": "I cannot see the pictures."}', ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: P())
    import pytest

    with pytest.raises(judge_views.JudgeFailed):
        judge_views.judge(tmp_path, "req")
