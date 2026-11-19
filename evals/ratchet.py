"""The ratchet: the skill only gets better.

uv run python evals/ratchet.py <runs dir> [<runs dir> ...] [--floor evals/floor.json]
uv run python evals/ratchet.py --check-git origin/main

With runs: groups the runs by (suite, card) (the next driver counts as card "next"), and compares
each group to evals/floor.json. Below the floor it names the specs that regressed and exits 1.
Above it, or for a group the floor does not have yet, it raises the floor in the file and says
so. It never lowers a floor.

--check-git BASE: fails when the working tree, against BASE, lowers any number in floor.json,
drops a group or a passing spec from it, or deletes a spec from evals/tuning, figure or
regressions, unless a line added to LOG.md names that group or spec and says why.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "evals" / "floor.json"
GUARDED = ("evals/tuning", "evals/figure", "evals/regressions")
# The judge is an LLM: the same build can score a point apart from one vote to the next. A group
# is below its judge floor only when its mean is more than this many points under it.
JUDGE_SLACK = 1.0


def _suite_of(spec_id: str, evals: Path) -> str:
    for p in evals.glob(f"*/{spec_id}.json"):
        return p.parent.name
    return "unknown"


def load_runs(dirs: list[Path], evals: Path = ROOT / "evals") -> list[dict]:
    """One entry per run folder under the dirs: {"key", "spec", "pass", "judge", "dir"}."""
    runs = []
    for d in dirs:
        logs = [d / "log.json"] if (d / "log.json").exists() else sorted(d.glob("*/log.json"))
        for p in logs:
            log = json.loads(p.read_text())
            suite = log.get("suite") or _suite_of(log["spec"], evals)
            card = "next" if log.get("driver") == "next" else str(log.get("card"))
            jp = p.parent / "judge.json"
            judge = json.loads(jp.read_text())["score"] if jp.exists() else log.get("judge")
            runs.append({"key": f"{suite}/{card}", "spec": log["spec"], "pass": bool(log.get("pass")),
                         "judge": judge, "dir": str(p.parent)})
    return runs


def measure(runs: list[dict]) -> dict:
    """Per key: {"rate", "judge" (mean, or None), "total", "passing" (specs that passed every run),
    "failing" (the other specs run)}."""
    out: dict = {}
    for key in sorted({r["key"] for r in runs}):
        rs = [r for r in runs if r["key"] == key]
        js = [r["judge"] for r in rs if r["judge"] is not None]
        specs = sorted({r["spec"] for r in rs})
        out[key] = {"rate": round(sum(r["pass"] for r in rs) / len(rs), 4),
                    "judge": round(sum(js) / len(js), 2) if js else None,
                    "total": len(rs),
                    "passing": [s for s in specs if all(r["pass"] for r in rs if r["spec"] == s)]}
        out[key]["failing"] = [s for s in specs if s not in out[key]["passing"]]
    return out


def compare(now: dict, floor: dict) -> tuple[list[str], dict]:
    """(regressions, new floor). A group is below its floor when a spec that passed at the floor
    fails now, its pass rate is lower, or its mean judge score is lower."""
    bad: list[str] = []
    new = json.loads(json.dumps(floor))
    for key, m in now.items():
        f = floor.get(key)
        if f is None:
            new[key] = {k: v for k, v in m.items() if k != "failing"}
            continue
        lost = [s for s in f.get("passing", []) if s in m["failing"]]
        below = False
        if lost:
            bad += [f"{key}: {s} passed at the floor and fails now" for s in lost]
            below = True
        if m["total"] >= f.get("total", 0) and m["rate"] < f["rate"]:
            bad.append(f"{key}: pass rate {m['rate']:.2f} is below the floor {f['rate']:.2f}")
            below = True
        if m["judge"] is not None and f.get("judge") is not None and m["judge"] < f["judge"] - JUDGE_SLACK:
            bad.append(f"{key}: judge score {m['judge']:.2f} is below the floor {f['judge']:.2f}"
                       f" by more than {JUDGE_SLACK:g}")
            below = True
        if below:
            continue
        g = new[key]
        if m["total"] >= f.get("total", 0) and m["rate"] > f["rate"]:
            g["rate"], g["total"] = m["rate"], m["total"]
        if m["judge"] is not None and (f.get("judge") is None or m["judge"] > f["judge"]):
            g["judge"] = m["judge"]
        g["passing"] = sorted(set(f.get("passing", [])) | set(m["passing"]))
    return bad, new


def check_runs(dirs: list[Path], floor_path: Path = FLOOR, evals: Path = ROOT / "evals") -> int:
    runs = load_runs(dirs, evals)
    if not runs:
        print("no runs found")
        return 1
    now = measure(runs)
    floor = json.loads(floor_path.read_text()) if floor_path.exists() else {}
    bad, new = compare(now, floor)
    for key, m in now.items():
        j = "" if m["judge"] is None else f", judge {m['judge']:.2f}"
        f = floor.get(key)
        fl = "no floor yet" if f is None else f"floor {f['rate']:.2f}" + ("" if f.get("judge") is None else f", judge {f['judge']:.2f}")
        print(f"{key}: {m['rate']:.2f} of {m['total']} runs{j} ({fl})" + (f"; failing: {', '.join(m['failing'])}" if m["failing"] else ""))
    if bad:
        print("BELOW THE FLOOR:")
        for b in bad:
            print("  " + b)
        return 1
    if new != floor:
        floor_path.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        print(f"floor raised in {floor_path}")
    else:
        print("at the floor")
    return 0


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def check_git(base: str, root: Path = ROOT) -> int:
    """Violations of the ratchet in the working tree against `base`; 0 when there are none."""
    rel = "evals/floor.json"
    try:
        old = json.loads(_git(root, "show", f"{base}:{rel}"))
    except subprocess.CalledProcessError:
        old = {}
    p = root / rel
    cur = json.loads(p.read_text()) if p.exists() else {}
    found: list[tuple[str, str]] = []  # (name a LOG.md line must hold, what happened)
    for key, f in old.items():
        g = cur.get(key)
        if g is None:
            found.append((key, f"floor {key} removed"))
            continue
        for field in ("rate", "judge"):
            if f.get(field) is not None and (g.get(field) is None or g[field] < f[field]):
                found.append((key, f"floor {key} {field} lowered from {f[field]} to {g.get(field)}"))
        for s in sorted(set(f.get("passing", [])) - set(g.get("passing", []))):
            found.append((s, f"floor {key} no longer lists {s} as passing"))
    gone = _git(root, "diff", "--name-only", "--diff-filter=D", base, "--", *GUARDED).split()
    for path in gone:
        found.append((Path(path).stem, f"spec {path} deleted"))
    added = [ln[1:] for ln in _git(root, "diff", base, "--", "LOG.md").splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    bad = [what for name, what in found if not any(name in ln for ln in added)]
    for name, what in found:
        if what not in bad:
            print(f"allowed by LOG.md: {what}")
    if bad:
        print("THE RATCHET SLIPPED (add a LOG.md line naming it and why, if it is on purpose):")
        for b in bad:
            print("  " + b)
        return 1
    print(f"ratchet holds against {base}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--floor", type=Path, default=FLOOR)
    ap.add_argument("--check-git", metavar="BASE")
    a = ap.parse_args()
    if a.check_git:
        return check_git(a.check_git)
    if not a.runs:
        ap.error("give a runs dir or --check-git BASE")
    return check_runs(a.runs, a.floor)


if __name__ == "__main__":
    sys.exit(main())
