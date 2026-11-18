"""Round 2 harness: one model, one tool (stone), one job folder per run.

uv run python evals/run.py <spec.json> --seed N [--card S|M|L|none] [--driver lfm3|next]
                          [--compact-at K] [--out DIR] [--server URL]

Fixed for every condition (same as lfm3-3d-skill harness v3): 10 steps, 2048 new tokens per
step, the model card's sampling, a cut step continues the same thought on the next step. Only
the card and the compaction change.

Drivers:
  lfm3  a llama.cpp server (raw chat template through /apply-template, then /completion)
  next  no model: runs stone status, then every NEXT line stone prints, until it stops. This
        is the baseline that shows what the tool does without any model.

--compact-at K: after step K, every earlier turn is dropped. The model keeps only the system
prompt and the request, plus a note that its context was compacted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoothstone import check as checks  # noqa: E402
from smoothstone import recipes as rc  # noqa: E402

HARNESS = "r2.1 (v3 settings; one stone tool; lenient call parse; optional forced compaction)"
MAX_STEPS = 10
MAX_NEW = 2048
SAMPLING = {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1}
STONE = str(ROOT / ".venv" / "bin" / "stone")

SYSTEM = ("You are an agent that makes 3D printable parts. Use the stone tool to do the task. "
          "When the part is ready, reply with a short final message and no tool call.")
COMPACTED = ("Your context was compacted, so your earlier steps are gone. The job is still on disk. "
             "Run stone status to see where you are.")
TOOLS = [{"type": "function", "function": {
    "name": "stone",
    "description": "Run one stone command, for example: stone status",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the stone command line"}},
                   "required": ["command"]}}}]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_stone(command: str, env: dict, cwd: Path) -> str:
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"FIX: could not read the command ({e}). Write it like: stone status"
    if argv and argv[0] == "stone":
        argv = argv[1:]
    if not argv:
        return "FIX: empty command.\nNEXT: stone status"
    # its own process group, so a timeout also stops the models it started (mflux, the lift)
    p = subprocess.Popen([STONE, *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         cwd=cwd, start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=int(os.environ.get("STONE_CMD_TIMEOUT", "900")))
    except subprocess.TimeoutExpired:  # a figure build takes about 430 s; its style and lift are kept
        stop_group(p)
        return f"FIX: the command ran too long, and what it finished is kept.\nNEXT: stone {shlex.join(argv)}"
    out = (stdout or "") + (("\n" + stderr) if stderr and not stdout else "")
    return out.strip()[:3000]


def stop_group(p: subprocess.Popen) -> None:
    """SIGTERM stone's whole process group, give it 5 s, then SIGKILL whatever is left of it."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:  # the group is gone
            break
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for f in (p.stdout, p.stderr):
        if f:
            f.close()


def last_next(text: str) -> str | None:
    m = re.findall(r"^NEXT: (.+)$", text, re.M)
    return m[-1].strip() if m else None


# --- llama.cpp driver -------------------------------------------------------------------------
def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_turn(text):
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
    else:
        reasoning, rest = text, ""
    calls = []
    for i, m in enumerate(TOOL_RE.finditer(rest)):
        name, body = m.group(1).strip(), m.group(2)
        args = {k: v for k, v in PARAM_RE.findall(body)}
        # same rule for every condition: <function=stone status> carries the command in the name,
        # and a bare body with no <parameter> is the command
        if name.split()[0] == "stone" and " " in name and "command" not in args:
            args = {"command": name}
        elif not args and body.strip():
            args = {"command": body.strip()}
        calls.append({"id": f"call{i}", "type": "function",
                      "function": {"name": name.split()[0], "arguments": json.dumps(args)}})
    return reasoning.strip(), TOOL_RE.sub("", rest).strip(), calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--card", default="S", choices=["S", "M", "L", "none"])
    ap.add_argument("--driver", default="lfm3", choices=["lfm3", "next"])
    ap.add_argument("--compact-at", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT.parent / "runs2"))
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--model-name", default="LFM3-1.7B Q4_K_M (LiquidAI/LFM3-1.7B-GGUF)")
    a = ap.parse_args()

    spec_path = Path(a.spec).resolve()
    spec = json.loads(spec_path.read_text())
    cond = a.driver if a.driver == "next" else f"card{a.card}" + (f"_compact{a.compact_at}" if a.compact_at else "")
    run_id = f"{spec['id']}__{cond}__s{a.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
    rdir = Path(a.out) / run_id
    job = rdir / "job"
    rdir.mkdir(parents=True)
    env = {**os.environ, "STONE_JOB": str(job), "STONE_TIER": a.card if a.card != "none" else "S"}

    # the environment, set up before the model starts: the job with the request (and picture)
    new = ["new", spec["request"]]
    if spec.get("image"):
        new += ["--image", str((spec_path.parent / spec["image"]).resolve())]
    setup = subprocess.run([STONE, *new], capture_output=True, text=True, env=env, cwd=rdir).stdout

    system = SYSTEM
    card_sha = None
    if a.card != "none":
        card = (ROOT / "skill" / "tiers" / f"{a.card}.md").read_text()
        card_sha = sha(card.encode())
        system += "\n\nA skill is loaded for this task. Follow it.\n\n" + card
    user = spec["request"] + ("\nThe picture is already in the job." if spec.get("image") else "") + \
        "\nThe job is ready; use the stone tool."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    log = {"run_id": run_id, "spec": spec["id"], "suite": spec_path.parent.name, "spec_sha256": sha(spec_path.read_bytes()), "condition": cond,
           "seed": a.seed, "driver": a.driver, "card": a.card, "card_sha256": card_sha, "compact_at": a.compact_at,
           "model": a.model_name if a.driver == "lfm3" else None, "sampling": SAMPLING if a.driver == "lfm3" else None,
           "harness": HARNESS, "max_steps": MAX_STEPS, "max_new_tokens": MAX_NEW, "setup": setup,
           "system_sha256": sha(system.encode()), "steps": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    t0 = time.time()
    pending = ""
    nxt = "stone status"
    print(f"{spec['id']}  {cond}  seed {a.seed}", flush=True)

    # a follow-up is the person asking for a change after the model stopped: the same chat goes on
    says = [None] + [f["say"] for f in spec.get("followups", [])]
    first = None  # an "asks" spec with follow-ups: the first phase must end in a question, not a part
    for phase, say in enumerate(says):
        if phase == 1 and spec["expect"].get("asks"):
            first = grade_job(job, spec, messages)
        if say:
            messages.append({"role": "user", "content": say})
            nxt = "stone status"
            print(f"   person: {say}", flush=True)
        for step in range(1, MAX_STEPS + 1):
            st = {"step": step, "phase": phase, "t": round(time.time() - t0, 2)}
            if a.compact_at and step == a.compact_at + 1 and a.driver == "lfm3":
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\n" + COMPACTED}]
                pending = ""
                st["compacted"] = True
                print(f"{step:>2} (context compacted)", flush=True)
            if a.driver == "next":
                if not nxt or "<" in nxt or nxt.startswith(("none", "ask the person")):
                    st["final"] = f"stopped: NEXT is {nxt!r}"
                    log["steps"].append(st)
                    break
                calls = [{"id": f"n{step}", "type": "function", "function": {"name": "stone", "arguments": json.dumps({"command": nxt})}}]
                am = {"role": "assistant", "content": "", "tool_calls": calls}
            else:
                prompt = _post(a.server, "/apply-template", {"messages": messages, "tools": TOOLS})["prompt"] + pending
                r = _post(a.server, "/completion", {"prompt": prompt, "n_predict": MAX_NEW, "seed": a.seed, "stop": ["<|im_end|>"],
                                                    "cache_prompt": True, **SAMPLING})
                text, cut = r["content"], r.get("stop_type") == "limit"
                st.update({"generated": text, "tokens_predicted": r.get("tokens_predicted"), "stop_type": r.get("stop_type"),
                           "continued_from_cut": bool(pending)})
                if cut:
                    pending += text
                    st["tool_calls"] = []
                    log["steps"].append(st)
                    print(f"{step:>2} (cut at {MAX_NEW} tokens, continues)", flush=True)
                    continue
                reasoning, content, calls = parse_turn(pending + text)
                pending = ""
                am = {"role": "assistant", "content": content}
                if reasoning:
                    am["reasoning_content"] = reasoning
                if calls:
                    am["tool_calls"] = calls
            messages.append(am)
            st["tool_calls"] = []
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if c["function"]["name"] != "stone":
                    result = "FIX: the only tool is stone. Call it with a command like: stone status"
                else:
                    result = run_stone(str(args.get("command", "")), env, rdir)
                nxt = last_next(result) or nxt
                st["tool_calls"].append({"name": c["function"]["name"], "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
                print(f"{step:>2} {args.get('command', '?')[:70]}  ->  {(last_next(result) or result.splitlines()[-1] if result else '')[:60]}", flush=True)
            log["steps"].append(st)
            if not calls:
                print(f"{step:>2} (stopped, no tool call)", flush=True)
                break

    exp = spec["followups"][-1]["expect"] if spec.get("followups") else spec["expect"]
    grade = grade_job(job, {**spec, "expect": exp}, messages)
    if first is not None and not first["pass"]:
        grade.update({"pass": False, "why": ["before the answer: " + w for w in first["why"]] + grade["why"]})
    if exp.get("judge") and grade["pass"]:
        grade = judge_run(rdir, job, spec["request"], exp["judge"], grade)
    if exp.get("recipe") == "figure" and grade["pass"]:
        grade = likeness_run(rdir, job, spec["request"], grade, (exp.get("likeness") or {}).get("min"))
    log.update({"seconds": round(time.time() - t0, 1), **grade, "messages": messages})
    (rdir / "log.json").write_text(json.dumps(log, indent=1))
    append_score(log)
    print(("PASS  " if grade["pass"] else "FAIL  ") + "; ".join(grade["why"])[:160], flush=True)


def grade_job(job: Path, spec: dict, messages: list | None = None) -> dict:
    """Outside the model's reach: did stone done write a part that passes the checks and the spec?"""
    why = []
    built = json.loads((job / "build" / "params.json").read_text()) if (job / "build" / "params.json").exists() else {}
    recipe = built.get("recipe")
    params = built.get("params", {})
    final = job / f"{recipe}.stl" if recipe else None
    done = bool(final and final.exists())
    if not done:
        why.append("no finished part")
    exp = spec["expect"]
    if exp.get("asks"):
        # a part that must fit something with no size given: the right move is a question, not a part
        last = next((m.get("content") or "" for m in reversed(messages or []) if m["role"] == "assistant"), "")
        why = [] if not done and "?" in last else (["built a part with a made-up size"] if done else ["ended without asking"])
        return {"pass": not why, "why": why or ["asked for the size"], "recipe": recipe, "params": params,
                "checks": [], "stl": None, "stl_sha256": None}
    if recipe != exp["recipe"]:
        why.append(f"recipe {recipe} (want {exp['recipe']})")
    for k, want in exp.get("params", {}).items():
        got = params.get(k)
        ok = (isinstance(want, list) and isinstance(got, (int, float)) and want[0] <= got <= want[1]) or \
             (not isinstance(want, list) and (got == want or (isinstance(want, (int, float)) and isinstance(got, (int, float)) and abs(got - want) < 1e-6)))
        if not ok:
            why.append(f"{k}={got!r} (want {want})")
    for s in exp.get("sum", []):
        tot = sum(params.get(k, 0) or 0 for k in s["params"])
        if tot != s["eq"]:
            why.append(f"{'+'.join(s['params'])}={tot} (want {s['eq']})")
    if exp.get("image_used") and not params.get(exp["image_used"]):
        why.append("the picture was not used")
    res = []
    if done and recipe:
        r = rc.load(recipe)
        res = [x.to_dict() for x in checks.check_stl(final, recipe=r, params=params)]  # the file that ships
        bad = [x["name"] for x in res if not x["ok"]]
        if bad:
            why.append("print checks fail: " + ", ".join(bad))
    col = None
    if exp.get("colors"):
        col = grade_colors(job, recipe, exp["colors"])
        why += col.pop("why")
    stl_sha = sha(final.read_bytes()) if done else None
    out = {"pass": not why, "why": why or ["all checks pass"], "recipe": recipe, "params": params,
           "checks": res, "stl": str(final) if done else None, "stl_sha256": stl_sha}
    if col is not None:
        out["colors"] = col
    return out


def judge_run(rdir: Path, job: Path, request: str, want: dict, grade: dict) -> dict:
    """The blind judge on the run's pictures (judge_views.py), saved as judge.json next to the run.
    want {"faults_in": ["boundaries"]}: the part fails when most judge runs name a fault in one of
    those pictures (a vote of up to 3, so one noisy run does not flip it)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import judge_views

    why = [w for w in grade["why"] if w != "all checks pass"]
    try:
        res = judge_views.votes(job / "build", request, want.get("faults_in", ["boundaries"]))
    except (judge_views.JudgeFailed, subprocess.TimeoutExpired, OSError) as e:
        # a run the judge could not grade still fails, and still lands in log.json and scores.jsonl
        (rdir / "judge.json").write_text(json.dumps({"score": None, "error": str(e)}, indent=1))
        return {**grade, "pass": False, "why": why + [f"judge failed: {e}"], "judge": None}
    (rdir / "judge.json").write_text(json.dumps(res, indent=1))
    why += [f"judge: {f['panel']}: {f['what']}" for f in res["faults"]]
    return {**grade, "pass": not why, "why": why or grade["why"], "judge": res["score"], "judge_votes": res["votes"]}


def likeness_run(rdir: Path, job: Path, request: str, grade: dict, least: int | None = None) -> dict:
    """Every figure: is it the subject asked for? (judge_likeness.py, saved as likeness.json).
    The checks above pass any well made figure, even of the wrong person; a judge that cannot
    answer fails the run with the reason, never passes it. A spec's expect.likeness.min raises
    the bar above the judge's own."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import judge_likeness

    try:
        res = judge_likeness.judge(job, request)
    except Exception as e:  # no render, no reference, no reply: a run is never passed unjudged
        res = {"score": None, "pass": False, "conflicts": [], "error": f"{type(e).__name__}: {e}"}
    if least and res.get("score") is not None and res["score"] < least:
        res = {**res, "pass": False, "least": least}
    (rdir / "likeness.json").write_text(json.dumps(res, indent=1))
    if res["pass"]:
        return {**grade, "likeness": res["score"]}
    why = [w for w in grade["why"] if w != "all checks pass"]
    if res.get("error"):
        why.append(f"likeness: the judge could not answer ({res['error'][:120]})")
    else:
        bar = f", the spec asks {least}" if res.get("least") else ""
        why.append(f"likeness {res['score']}/10{bar}: " + "; ".join(f"{c['feature']}: {c['what']}"
                                                            for c in res["conflicts"])[:240])
    return {**grade, "pass": False, "why": why, "likeness": res["score"]}


SCORES = Path(os.environ.get("STONE_SCORES", str(ROOT / "evals" / "scores.jsonl")))


def append_score(log: dict) -> None:
    """One more line in evals/scores.jsonl; only rejudge.py replaces a run's line."""
    with SCORES.open("a") as f:
        f.write(json.dumps(score_line(log)) + "\n")


def score_line(log: dict) -> dict:
    """A run's line in evals/scores.jsonl."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        head = None
    line = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sha": head, "suite": log.get("suite"),
            "spec": log["spec"], "card": log["card"], "driver": log["driver"], "pass": int(bool(log["pass"])),
            "total": 1, "judge": log.get("judge"), "likeness": log.get("likeness"), "run": log["run_id"]}
    return line


def _part_key(name: str) -> str:
    """A part word to its group's default name: jacket and shirt are both "top"."""
    from smoothstone import parts

    name = name.lower()
    for key, words in parts.PART_WORDS.items():
        if name in words:
            return key
    return name


PART_OFF_MAX = 0.02  # share of a part's area that may take another color (0.0 measured on real f2 builds)


def grade_colors(job: Path, recipe: str | None, want: dict) -> dict:
    """The painted file against {"<color>": [parts], "rest": "<color>"}: the 3MF must be there and
    read back clean, its color edges smooth, and each part must print in the color asked for (every
    part not named takes the rest color). Returns {"why", "shares", "parts"}."""
    import numpy as np

    from smoothstone import colors as C

    why: list[str] = []
    f3 = job / f"{recipe}.3mf" if recipe else None
    if not f3 or not f3.exists():
        return {"why": [f"no {recipe}.3mf"], "shares": {}, "parts": {}}
    st = json.loads((job / "status.json").read_text()).get("colors") if (job / "status.json").exists() else None
    if not st:
        return {"why": ["stone status has no colors"], "shares": {}, "parts": {}}
    fil = [str(x).lower() for x in st["filaments"]]
    V, F, codes, main = C.read_3mf(f3)
    face_fil = np.array([C.decode_paint(c) if c else main for c in codes])
    stl = job / f"{recipe}.stl"
    vol = None
    if stl.exists():
        import trimesh

        vol = float(trimesh.load(stl, process=False).volume)
    why += C.check_3mf(f3, len(F), len(fil), volume=vol)
    worst, lines = C.color_edges(V, F, face_fil)
    pn, rj = job / "build" / "paint.npy", job / "build" / "regions.json"
    if pn.exists() and rj.exists():
        # the outline of the name raised on the plinth is the shape's own crease: not judged
        regs = json.loads(rj.read_text())["regions"]
        num0 = np.load(pn)
        if len(num0) == len(F):
            C.mark_name_edges(lines, F, num0, next((d["n"] for d in regs if d.get("part") == "name"), None),
                              next((d["n"] for d in regs if d.get("base")), None))
            worst = C.worst_ratio(lines)
    if worst > C.JAGGED:
        why.append(f"color edges jagged ({worst:.2f}, max {C.JAGGED})")
    if (st.get("wobble") or 0) > C.WOBBLE:
        why.append(f"color edges wavy ({st['wobble']:.2f} mm, max {C.WOBBLE})")
    m = C.load_stl_mesh(V, F)
    area = m.area_faces
    shares = {fil[n - 1]: round(float(area[face_fil == n].sum() / area.sum()), 3) for n in np.unique(face_fil)}
    got: dict[str, str] = {}
    pn, rj = job / "build" / "paint.npy", job / "build" / "regions.json"
    if not pn.exists() or not rj.exists():
        why.append("no paint.npy or regions.json to tell the parts")
        return {"why": why, "shares": shares, "parts": got}
    num = np.load(pn)
    if len(num) != len(F):
        why.append(f"paint.npy has {len(num)} triangles, the 3MF {len(F)}")
        return {"why": why, "shares": shares, "parts": got}
    tally: dict[str, np.ndarray] = {}
    regs = json.loads(rj.read_text())["regions"]
    # a sliver boxed in by a part (a jacket's lip under the hem line) is that part's own edge
    sj = job / "build" / "slivers.json"
    host = {int(k): v for k, v in json.loads(sj.read_text()).items()} if sj.exists() else {}
    part_of = {d["n"]: d.get("part") or "other" for d in regs}
    for d in regs:
        sel = num == d["n"]
        part = part_of.get(host.get(d["n"]), part_of[d["n"]])
        t = tally.setdefault(part, np.zeros(len(fil) + 1))
        t += np.bincount(face_fil[sel], weights=area[sel], minlength=len(fil) + 1)
    for part, t in tally.items():
        got[part] = fil[int(t[1:].argmax())]
    from smoothstone import parts as P

    named = {q: str(c).lower() for c, ps in want.items() if c != "rest" for p in ps
             for q in (P.parts_of(p) or (_part_key(p),))}
    rest = str(want.get("rest", "")).lower()
    plinth = got.get("base")
    for part, c in sorted(got.items()):
        if part == "name" and "name" not in named:
            # the name on the plinth takes a color that stands out from it, unless the request names one
            if c == plinth:
                why.append(f"name is {c}, the plinth's own color, so it does not read")
            continue
        need = named.get(_part_key(part), rest)
        if need and c != need:
            why.append(f"{part} is {c} (want {need})")
        elif need and need in fil:
            # the majority is right; the rest of the part must be right too, but for the band where
            # the paint follows the shape's groove instead of the part's own outline
            t = tally[part]
            off = 1 - float(t[fil.index(need) + 1] / t[1:].sum()) if t[1:].sum() else 0.0
            if off > PART_OFF_MAX:
                why.append(f"{part} is {off:.0%} not {need} (at most {PART_OFF_MAX:.0%})")
    for p in named:
        if not any(_part_key(q) == p for q in got):
            why.append(f"no part {p} in the listing")
    return {"why": why, "shares": shares, "parts": got}


if __name__ == "__main__":
    main()
