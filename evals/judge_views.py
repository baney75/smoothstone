"""A blind judge of a painted part: Opus reads views.png and boundaries.png and scores them.

uv run python evals/judge_views.py <run dir or job dir> [--request TEXT]

The judge sees the request text and the two pictures, nothing else: no spec, no log, no grade.
It runs as `claude -p --model opus` on Donovan's sign-in (ANTHROPIC_API_KEY unset, so no API
billing), with the Read tool only, in a folder that holds copies of the two pictures.

One judge run can name a fault on one pass and not the next, so a verdict is a vote: two runs,
and a third only when they disagree (`votes`). A fault counts when it is in a picture named in
`faults_in` (default: boundaries). The majority decides; the score is the median. judge.json next
to the run holds {"score", "faults" (those of the runs that failed it, when the majority did),
"pass", "votes": "k of n failed", "runs": [each run]}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUBRIC = """You judge a 3D printed part painted in a few filament colors, from two rendered pictures.

The request the part was made for:
{request}

Read both pictures with the Read tool:
  {views}: the whole part from several sides.
  {boundaries}: close-ups of each color edge, and of the collar from level and from 30 degrees up.

Score how well the paint does what the request asks, as a careful owner would before printing:
- 10: every part has the color asked for, every color edge is a clean line where the shape changes
  (collar, cuffs, hem), and nothing needs fixing by hand.
- 7 to 9: right colors; small flaws a person might not notice in the print (a thin rim, a slight
  wave in an edge).
- 4 to 6: right colors but a flaw anyone would see: a jagged or sawtooth edge, a strip of the wrong
  color on a ledge or under a collar, a stray patch.
- 1 to 3: a part has the wrong color, a part asked for is missing, or the paint is a mess.
A part the request does not name takes the rest color, so the neck, the underside of the hair at the
nape and the wrist below a cuff are correct in the rest color even where they show between painted
parts; a light line there is that part catching the light. A fault is paint on the wrong side of
where the shape changes: a strip of one color on the other part's surface, a step in the edge, a
stray patch. List in faults only what you judge to be wrong. Something you looked at and judge
correct (such as the neck showing as a light line above the collar) is not a fault: leave it out of
faults, or put it in an optional "notes" list of strings.
Judge only what you see in the pictures. Name each fault with the picture and the panel it is in,
starting with the picture's name (such as "boundaries row 2, column 3" or "views front"), and what
is wrong in a few words.

Reply with one JSON object and nothing else:
{{"score": <1-10>, "faults": [{{"panel": "...", "what": "..."}}]}}"""


class JudgeFailed(Exception):
    """The judge could not give a verdict (no claude, a picture missing, a reply with no JSON)."""


def find_request(d: Path) -> str:
    for p in (d / "log.json", d.parent / "log.json"):
        if p.exists():
            log = json.loads(p.read_text())
            for m in log.get("messages", []):
                if m.get("role") == "user":
                    return m["content"].split("\n")[0]
    for p in (d / "job" / "brief.md", d / "brief.md"):
        if p.exists():
            return p.read_text().strip().splitlines()[0]
    raise SystemExit("no request found: pass --request")


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in the judge's reply")
    out = json.loads(m.group(0))
    if not isinstance(out, dict) or isinstance(out.get("score"), bool) or not isinstance(out.get("score"), (int, float)):
        raise ValueError(f"no number score in {m.group(0)[:200]}")
    score = int(out["score"])
    if not 1 <= score <= 10:
        raise ValueError(f"score {score} outside 1..10")
    raw = out.get("faults", [])
    if not isinstance(raw, list) or not all(isinstance(f, dict) and isinstance(f.get("panel"), str)
                                            and isinstance(f.get("what", ""), str) for f in raw):
        raise ValueError(f"faults are not a list of {{panel, what}}: {str(raw)[:200]}")
    return {"score": score, "faults": [{"panel": f["panel"], "what": f.get("what", "")} for f in raw]}


def judge(build: Path, request: str, model: str = "opus") -> dict:
    with tempfile.TemporaryDirectory(prefix="judge-") as tmp:
        t = Path(tmp)
        for name in ("views.png", "boundaries.png"):
            src = build / name
            if not src.exists():
                raise JudgeFailed(f"missing {src}")
            shutil.copy(src, t / name)
        prompt = RUBRIC.format(request=request, views=t / "views.png", boundaries=t / "boundaries.png")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        try:
            p = subprocess.run(["claude", "-p", "--model", model, "--tools", "Read", "--allowedTools", "Read",
                                "--strict-mcp-config", "--no-session-persistence", "--add-dir", str(t),
                                "--output-format", "json", prompt],
                               capture_output=True, text=True, env=env, cwd=t, timeout=900)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise JudgeFailed(f"claude did not answer: {e}") from e
        if p.returncode != 0:
            raise JudgeFailed(f"claude exited {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
        try:
            return {**parse(json.loads(p.stdout).get("result", "")), "model": model, "request": request}
        except (ValueError, KeyError, TypeError) as e:
            raise JudgeFailed(f"the judge's reply had no verdict: {e}") from e


def named(res: dict, faults_in: list[str]) -> list[dict]:
    """The faults of one judge run that are in the pictures `faults_in` names."""
    return [f for f in res["faults"] if any(f["panel"].lower().startswith(p) for p in faults_in)]


def votes(build: Path, request: str, faults_in: list[str] | None = None, n: int = 3, model: str = "opus",
          one=judge) -> dict:
    """The majority of up to n blind judge runs: the first ceil(n/2) run together, and more run
    only while neither side has a majority. A run fails the part when it names a fault in a
    picture in `faults_in` (default: boundaries; [] counts no picture). A judge run that fails
    (JudgeFailed) is replaced, up to n tries in all; JudgeFailed only when no majority is left."""
    if faults_in is None:
        faults_in = ["boundaries"]
    need = n // 2 + 1
    runs: list[dict] = []
    errors: list[str] = []

    def try_one(_=None):
        try:
            return one(build, request, model)
        except JudgeFailed as e:
            errors.append(str(e))
            return None

    with ThreadPoolExecutor(need) as pool:
        runs += [r for r in pool.map(try_one, range(need)) if r is not None]
    tries = need
    while tries < n:
        fails = sum(bool(named(r, faults_in)) for r in runs)
        if fails >= need or len(runs) - fails >= need:
            break
        tries += 1
        r = try_one()
        if r is not None:
            runs.append(r)
    fails = [r for r in runs if named(r, faults_in)]
    if len(fails) < need and len(runs) - len(fails) < need:
        raise JudgeFailed(f"no majority from {len(runs)} verdicts in {tries} tries: " + "; ".join(errors))
    failed = len(fails) >= need
    return {"score": statistics.median(r["score"] for r in runs),
            "faults": [f for r in fails for f in named(r, faults_in)] if failed else [],
            "pass": not failed, "votes": f"{len(fails)} of {len(runs)} failed", "runs": runs,
            "errors": errors, "model": model, "request": request}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a run folder (with job/) or a job folder")
    ap.add_argument("--request")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--once", action="store_true", help="one judge run, no vote")
    ap.add_argument("--faults-in", default="boundaries", help="comma list of pictures whose faults fail the part")
    a = ap.parse_args()
    d = Path(a.run).resolve()
    job = d / "job" if (d / "job" / "build").exists() else d
    req = a.request or find_request(d)
    res = judge(job / "build", req, a.model) if a.once else votes(job / "build", req, [x for x in a.faults_in.split(",") if x], model=a.model)
    (d / "judge.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
