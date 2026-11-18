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


def test_one_judge_run_that_errors_is_replaced_not_fatal():
    """PR 8 review first-F3: [JudgeFailed, pass, pass] is a pass, not "judge failed"."""
    seq = iter([judge_views.JudgeFailed("rate limited"), False, False])

    def one(build, request, model):
        x = next(seq)
        if isinstance(x, Exception):
            raise x
        return {"score": 9, "faults": []}

    res = judge_views.votes(Path("."), "req", one=one)
    assert res["pass"] and res["votes"] == "0 of 2 failed" and res["errors"] == ["rate limited"]


def test_too_many_judge_errors_is_a_judge_failure():
    import pytest

    def one(build, request, model):
        raise judge_views.JudgeFailed("not signed in")

    with pytest.raises(judge_views.JudgeFailed):
        judge_views.votes(Path("."), "req", one=one)


def test_an_empty_faults_in_counts_no_picture():
    """PR 8 review first-F2: "faults_in": [] (or a judge spec with none) only scores."""
    def one(build, request, model):
        return {"score": 7, "faults": [{"panel": "boundaries row 1", "what": "notch"}]}

    assert judge_views.votes(Path("."), "req", [], one=one)["pass"]


def test_a_reply_of_the_wrong_shape_is_a_judge_failure():
    """PR 8 review first-F1: valid JSON with a fault missing its panel, a plain string fault, or a
    score like "7/10" must not escape as a KeyError later."""
    import pytest

    for reply in ('{"score": 7, "faults": [{"where": "boundaries", "what": "notch"}]}',
                  '{"score": 7, "faults": ["a notch"]}', '{"score": "7/10", "faults": []}'):
        with pytest.raises(ValueError):
            judge_views.parse(reply)
    assert judge_views.parse('{"score": 8, "faults": [{"panel": "views front", "what": "x"}]}')["score"] == 8


def test_a_judge_run_that_times_out_is_a_judge_failure(tmp_path, monkeypatch):
    """PR 8 review r2 first-F2: a slow Opus reply is replaced like any failed run."""
    import subprocess

    import pytest

    for name in ("views.png", "boundaries.png"):
        (tmp_path / name).write_bytes(b"png")

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("claude", 900)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(judge_views.JudgeFailed):
        judge_views.judge(tmp_path, "req")
