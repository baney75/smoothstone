"""Photos -> styled render -> lifted mesh -> one clean printable body.

Local models do the work, none of them the model that drives stone:
- describe: LFM3-VL-2B (4 bit, about 3 GB) writes one sentence on how an animal or a thing looks,
  and answers yes or no on a face, glasses, a beard and a hat to check photos and renders.
- face: tools/figure/face.py (on the lift venv's opencv) crops a person's face out of the lead
  photo, for the style model to see first.
- style: FLUX.2 klein 4B edit through mflux (Apache 2.0, about 5 GB at 4 bit) turns the face crop
  and the photo into a clean front view of a collectible figure on white.
- lift: tools/lift/lift.py turns that render into a mesh (single-image 3D).
- repair: trimesh keeps the main body, closes it and scales it to unit height, feet on z = 0.

Every step writes its file into the job's images/figure folder and is skipped when the file is
already there and its inputs have not changed, so a build that dies half way picks up again.
A finished figure is also kept in a shared cache (STONE_FIGURE_CACHE, default
~/.cache/stone-figure; "0" turns it off) under the same key, so a new job with the same photos and
settings reuses it instead of running the models again. A machine without the style model (Linux
has no mflux) can build a figure that was made once elsewhere, by copying that key folder in.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from . import config

STYLES = ("chibi", "statue", "bust", "object")

LOOK = {
    "chibi": (
        "a collectible chibi vinyl figure of this exact {who}: big head, small body, a friendly "
        "smile, the same clothes, arms close to the body, feet together, standing upright"
    ),
    "statue": (
        "a detailed collectible statue of this exact {who} with natural proportions: the same "
        "clothes, arms close to the body, feet together, standing upright, sculpted folds and "
        "textures"
    ),
    "bust": (
        "a sculpted bust of this exact {who}: head, neck and shoulders on a short round stand, "
        "the same clothes at the shoulders"
    ),
    "object": (
        "this exact {who} alone, the same shape, proportions and details, as a solid object "
        "that stands on its own"
    ),
}
# a person's chibi is a likeness figure: a chibi face is a generic cartoon face, and the blind
# likeness judge gave Elon 5 as a chibi and 9 on both seeds with this look and the face crop
PERSON_LOOK = {
    "chibi": (
        "a premium collectible figure of this exact {who} with a large head, about one third of the "
        "figure's height, on a slightly small body, the real face from the photo sculpted in fine detail: "
        "the same jaw, chin, nose, eyes, eyebrows, mouth and expression, the same hairstyle and hairline, "
        "the same clothes, arms close to the body, feet together, standing upright"
    ),
}
# for a render that came back with lettering or logos on the clothes (DHH's race suit): a figure
# cannot print or paint them, and they are other people's marks. Only then, as the first try it
# turned Jensen's leather jacket into a sweater.
PLAIN = ("the same clothes,", "the same kind of clothes in plain solid colors,")
LIKENESS = {
    "person": ("Keep the likeness from the first photo exactly: the same hair color, hairline and "
               "hairstyle, the same face shape, expression and skin tone. "),
    "animal": "Keep the likeness from the first photo exactly: the same fur colors, markings, ears and face. ",
    "character": ("Keep the likeness from the first picture exactly: the same face, colors, costume, "
                  "markings and proportions of this character. "),
    "object": "",
}
# only for a person the description says wears glasses: a style model draws whatever a prompt
# names, so "the same glasses if any" or "no glasses" both put glasses on a face without them
GLASSES = ("The same glasses exactly: the same frame shape, the same thin frame thickness and color, "
           "never thicker or bolder than in the photo. ")
WEARS_GLASSES = re.compile(r"\b(?:glasses|spectacles|eyeglasses|sunglasses)\b", re.I)
# from a denial to the end of its clause: "no glasses", "not wearing a hat", "without a tie"
DENIAL = re.compile(r"\b(?:no|not|without|never|neither|nor)\b[^,.;]*|\b\w+n't\b[^,.;]*", re.I)


def positive(text: str) -> str:
    """The description with every denial cut out: what is there, never what is not."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        clauses = [DENIAL.sub("", c) for c in re.split(r"(,|;)", sentence)]
        kept = "".join(clauses)
        kept = re.sub(r"\b(?:and|with|but|or|has|have|is|are)\s*(?=[,.;!?]|$)", "", kept)
        kept = re.sub(r"(?:\s*[,;]\s*)+(?=[,;.!?]|$)", "", kept)  # empty clauses
        kept = re.sub(r"\s*[,;]\s*[,;]", ",", kept)
        kept = re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s{2,}", " ", kept)).strip(" ,;")
        if len(re.findall(r"[A-Za-z]{2,}", kept)) >= 2:  # "He" alone is what a denial left
            out.append(kept if kept[-1] in ".!?" else kept + ".")
    return " ".join(out)


FRAME = (
    "{look}. {likeness}Full figure in frame, centered, facing the camera, front view, plain pure white "
    "background, soft studio product lighting, crisp sculpted details, matte finish, no base, "
    "no text, no shadow on the background."
)


class FigureProblem(Exception):
    pass


def prompt(style: str, who: str = "person", extra: str = "", glasses: bool | None = None,
           plain: bool = False) -> str:
    """The style prompt. For a person the photo carries the look and the description stays out:
    its "light brown hair" drew Elon's hair too light (judge 8 with it, 9 without). Glasses come
    from the photo check when it ran, else from the description."""
    extra = positive(extra)
    likeness = LIKENESS.get(who, "")
    look = (PERSON_LOOK if who == "person" else LOOK).get(style, LOOK[style])
    if plain:
        look = look.replace(*PLAIN)
    if who == "person":
        if glasses if glasses is not None else WEARS_GLASSES.search(extra):
            likeness += GLASSES
        extra = ""
    p = FRAME.format(look=look.format(who=who), likeness=likeness)
    return f"{p} {extra.strip()}".strip()


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            h.update(p.read_bytes())
        else:
            h.update(str(p).encode())
    return h.hexdigest()[:16]


def _mflux_bin() -> Path:
    env = os.environ.get("STONE_MFLUX")
    base = Path(env).expanduser() if env else Path.home() / ".cache" / "stone-style" / ".venv" / "bin"
    exe = base / "mflux-generate-flux2-edit"
    if not exe.exists():
        raise FigureProblem("the style model is not set up, run tools/figure/setup.sh once")
    return exe


def _style_model() -> list[str]:
    """The local 4 bit klein weights (mflux-community/flux2-klein-4b-mflux-q4), else the Hub copy."""
    env = os.environ.get("STONE_STYLE_MODEL")
    local = Path(env).expanduser() if env else Path.home() / ".cache/stone-style/models/flux2-klein-4b-q4"
    if (local / "transformer").is_dir():
        return ["--model", str(local), "--base-model", "flux2-klein-4b"]
    return ["--model", "flux2-klein-4b", "--quantize", "4"]


def _describe_model() -> Path | None:
    env = os.environ.get("STONE_DESCRIBE_MODEL")
    local = Path(env).expanduser() if env else Path.home() / ".cache/stone-style/models/lfm3-vl-2b-4bit"
    return local if (local / "config.json").exists() else None


def describe(photo: Path, out: Path, who: str) -> str:
    """One sentence on how the subject looks, written to out. Empty when the model is not set up."""
    if out.exists():
        return out.read_text().strip()
    model = _describe_model()
    if model is None or os.environ.get("STONE_DESCRIBE") == "0":
        out.write_text("")
        return ""
    py = _mflux_bin().parent / "python"
    script = config.REPO_ROOT / "tools" / "figure" / "describe.py"
    log = out.with_name("describe.log")
    with log.open("a") as f:
        try:
            r = subprocess.run([str(py), "-I", str(script), str(photo), who, str(model)],
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as e:
            raise FigureProblem("the describe model took over 600 s") from e
        f.write(r.stderr)
    text = " ".join(r.stdout.split())
    if r.returncode != 0 or not text:
        raise FigureProblem(f"the describe model failed, see {log}")
    out.write_text(text + "\n")
    return text


# what the render must keep from the photo; a style model adds glasses or a beard on its own
KEEP = {"glasses": "glasses", "beard": "a beard", "hat": "a hat"}
STYLE_TRIES = 3


def traits(images: list[Path], who: str, log: Path) -> list[dict] | None:
    """Yes or no for each image: a face, glasses, a beard, a hat (tools/figure/describe.py --traits).

    None when the describe model is not set up or the check fails: the figure is then made
    unchecked, as before, never stopped by its own check."""
    model = _describe_model()
    if model is None or os.environ.get("STONE_DESCRIBE") == "0" or who not in ("person", "animal"):
        return None
    script = config.REPO_ROOT / "tools" / "figure" / "describe.py"
    try:
        py = _mflux_bin().parent / "python"
        r = subprocess.run([str(py), "-I", str(script), "--traits", who, str(model), *map(str, images)],
                           capture_output=True, text=True, timeout=600)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(r.stderr[-4000:])
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (FigureProblem, OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    return got if isinstance(got, list) and len(got) == len(images) else None


def screen(photos: list[Path], who: str, folder: Path, say=print) -> tuple[list[Path], dict | None]:
    """The photos that show the subject's face, a bare head first, and what the first one shows.

    Only the first photo sets the likeness, so a photo of a car or a protein that came back for a
    name must not lead. The answers are kept in folder, so a rebuild asks nothing."""
    if who not in ("person", "animal"):
        return photos, None
    memo = folder / f"screen-{_digest(*photos, who)}.json"
    if memo.exists():
        seen = json.loads(memo.read_text())
    else:
        seen = traits(photos, who, folder / "screen.log")
        if seen is None:
            return photos, None
        folder.mkdir(parents=True, exist_ok=True)
        memo.write_text(json.dumps(seen))
    # a photo the owner gave is theirs to trust: a small or side-on face can read as none
    keep = [(p, t) for p, t in zip(photos, seen) if t.get("face") or p.name.startswith("source.")]
    if not keep:
        raise FigureProblem(f"none of the {len(photos)} photos shows the {who}'s face; "
                            "give one with stone image <path to photo>")
    if len(keep) < len(photos):
        n = len(photos) - len(keep)
        say(f"Left out {n} photo{'s' if n > 1 else ''} with no face in {'them' if n > 1 else 'it'}.")
    keep.sort(key=lambda pt: bool(pt[1].get("hat")))  # hair shows best on a bare head
    return [p for p, _ in keep], keep[0][1]


def style_checked(photos: list[Path], out: Path, style_name: str, who: str, seed: int, extra: str,
                  want: dict | None, say=print) -> Path:
    """style(), then the render checked against the photo: one that added or left out glasses,
    a beard or a hat is styled again with the next seed, and the closest of STYLE_TRIES is kept."""
    best, plain = None, False
    check = bool(want) and any(k in want for k in KEEP)  # nothing to hold a render to otherwise
    for i in range(STYLE_TRIES if check else 1):
        s = seed + i
        o = out.with_name(f"style-{s}.png") if check else out
        style(photos, o, style_name, who, s, extra, glasses=(want or {}).get("glasses"), plain=plain)
        if not check:
            return o
        got = (traits([o], who, out.with_name("check.log")) or [None])[0]
        if got is None:  # the check could not run: keep this one
            best = (o, [])
            break
        miss = [k for k in KEEP if k in want and k in got and bool(got[k]) != bool(want[k])]
        miss += ["lettering"] if got.get("lettering") else []  # never wanted, whatever the photo has
        if best is None or len(miss) < len(best[1]):
            best = (o, miss)
        if not miss:
            break
        how = " and ".join(("added " if got[k] else "left out ") + KEEP[k] for k in miss if k in KEEP)
        if "lettering" in miss:
            plain = True
            how = " and ".join(x for x in (how, "put lettering or a logo on the clothes, which a figure cannot "
                                                "print") if x)
        again = " in plain colors" if plain else ""
        if i + 1 < STYLE_TRIES:
            say(f"The render {how}, unlike the photo; styling again{again} with seed {s + 1}.")
        else:
            say(f"The render {how}, unlike the photo, in all {STYLE_TRIES} tries; keeping the closest.")
    _landed(out, lambda tmp: shutil.copyfile(best[0], tmp))
    out.with_name("check.json").write_text(json.dumps({"want": want, "kept": best[0].name,
                                                       "differs": best[1]}, indent=1))
    return out


def face_crop(photo: Path, out: Path) -> Path | None:
    """The face and hair of the photo's biggest face (tools/figure/face.py), or None when there is
    no face or the style venv is missing: the figure is then styled from the whole photo."""
    if out.exists():
        return out
    try:  # the lift venv pins opencv 4, whose Haar face finder opencv 5 (in the style venv) dropped
        lift_py = config.REPO_ROOT / "tools" / "lift" / ".venv" / "bin" / "python"
        py = lift_py if lift_py.exists() else _mflux_bin().parent / "python"
        r = subprocess.run([str(py), "-I", str(config.REPO_ROOT / "tools" / "figure" / "face.py"),
                            str(photo), str(out)], capture_output=True, text=True, timeout=120)
    except (FigureProblem, OSError, subprocess.TimeoutExpired):
        return None
    return out if r.returncode == 0 and out.exists() else None


def _lift_cmd() -> list[str]:
    env = os.environ.get("STONE_LIFT")
    if env:
        return shlex.split(env)
    lift = config.REPO_ROOT / "tools" / "lift"
    if not (lift / "lift.py").exists():
        raise FigureProblem("the lift model is not set up, run tools/figure/setup.sh once")
    uv = shutil.which("uv") or "uv"
    return [uv, "run", "--project", str(lift), "python", str(lift / "lift.py")]


def _landed(out: Path, write) -> Path:
    """write(tmp) makes the step's output under a temporary name in out's folder (same suffix, so
    a writer that goes by it still can); only a write that returns moves it onto out. The harness
    kills a slow build mid-write, and the next run reuses out as done, so out is never half a file."""
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    tmp.unlink(missing_ok=True)
    try:
        write(tmp)
        if tmp.exists():
            os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _run(cmd: list[str], log: Path, timeout: int) -> None:
    t = time.time()
    with log.open("a") as f:
        f.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise FigureProblem(f"{Path(cmd[0]).name} took over {timeout} s") from e
        f.write(f"exit {r.returncode} after {time.time() - t:.0f} s\n")
    if r.returncode != 0:
        raise FigureProblem(f"{Path(cmd[-1] if 'lift' in cmd[-1] else cmd[0]).name} failed, see {log}")


def style(photos: list[Path], out: Path, style_name: str, who: str, seed: int, extra: str = "",
          glasses: bool | None = None, plain: bool = False) -> Path:
    if out.exists():
        return out
    cmd = [str(_mflux_bin()), *_style_model(),
           "--image-paths", *map(str, photos),
           "--prompt", prompt(style_name, who, extra, glasses, plain),
           "--width", "768", "--height", "1024", "--steps", "4", "--seed", str(seed),
           "--low-ram",
           "--output"]
    _landed(out, lambda tmp: _run(cmd + [str(tmp)], out.with_name("style.log"), timeout=1800))
    if not out.exists():
        raise FigureProblem("the style model wrote no picture")
    return out


def lift(render: Path, out: Path, seed: int) -> Path:
    if out.exists():
        return out
    _landed(out, lambda tmp: _run(_lift_cmd() + [str(render), str(tmp), "--seed", str(seed),
                                                 "--octree", os.environ.get("STONE_LIFT_OCTREE", "256"),
                                                 "--chunks", os.environ.get("STONE_LIFT_CHUNKS", "4000")],
                                  out.with_name("lift.log"), timeout=3600))
    if not out.exists():
        raise FigureProblem("the lift model wrote no mesh")
    return out


def repair(src: Path, out: Path, max_faces: int = 250_000) -> dict:
    """Largest closed body, Z up, front to -Y, unit height, centered on x/y, feet on z = 0."""
    import numpy as np
    import trimesh

    m = trimesh.load(src, force="mesh")
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: abs(p.volume) if p.is_watertight else p.area * 1e-3)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight:
        m = _remesh(m)
    if len(m.faces) > max_faces:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    trimesh.repair.fix_normals(m)
    # lift models write Y up with the face toward +Z; OpenSCAD wants Z up with the face toward -Y
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    lo, hi = m.bounds
    h = hi[2] - lo[2]
    # center on the feet: the middle of the lowest 5% of the body
    z = m.vertices[:, 2]
    feet = m.vertices[z < lo[2] + 0.05 * h]
    cx, cy = feet[:, 0].mean(), feet[:, 1].mean()
    m.apply_translation([-cx, -cy, -lo[2]])
    m.apply_scale(1.0 / h)
    lo, hi = m.bounds
    info = {"faces": len(m.faces), "watertight": bool(m.is_watertight),
            "width": float(hi[0] - lo[0]), "depth": float(hi[1] - lo[1]),
            "feet_w": float(np.ptp(feet[:, 0]) / h), "feet_d": float(np.ptp(feet[:, 1]) / h)}

    def write(tmp: Path) -> None:  # the mesh lands last: make() takes figure.stl as the step done
        m.export(tmp)
        _landed(out.with_suffix(".json"), lambda t: t.write_text(json.dumps(info, indent=1)))
    _landed(out, write)
    return info


def _remesh(m):
    """Voxel remesh for a mesh that will not close: solid fill, then marching cubes."""
    pitch = max(m.extents) / 220
    v = m.voxelized(pitch).fill()
    return v.marching_cubes.apply_transform(v.transform)


CACHED = ("look.txt", "style.png", "lift.glb", "figure.stl", "figure.json")


def cache_dir() -> Path | None:
    env = os.environ.get("STONE_FIGURE_CACHE")
    if env == "0":
        return None
    return Path(env).expanduser() if env else Path.home() / ".cache" / "stone-figure"


def _from_cache(key: str, folder: Path) -> bool:
    c = cache_dir()
    if c is None or not (c / key / "figure.stl").exists():
        return False
    for name in CACHED:
        if (c / key / name).exists() and not (folder / name).exists():
            shutil.copyfile(c / key / name, folder / name)
    return True


def _to_cache(key: str, folder: Path) -> None:
    c = cache_dir()
    if c is None or (c / key / "figure.stl").exists():
        return
    try:
        tmp = c / f".{key}.tmp{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        for name in CACHED:
            if (folder / name).exists():
                shutil.copyfile(folder / name, tmp / name)
        tmp.rename(c / key)
    except OSError:
        shutil.rmtree(c / f".{key}.tmp{os.getpid()}", ignore_errors=True)


def make(photos: list[Path], folder: Path, style_name: str = "chibi", who: str = "person",
         seed: int = 7, extra: str = "", say=print) -> Path:
    """Run every step that is not done yet and return the unit-height figure.stl."""
    if style_name not in STYLES:
        raise FigureProblem(f"style must be one of {', '.join(STYLES)}")
    if not photos:
        raise FigureProblem("no photos, run stone find or stone image first")
    photos, want = screen(list(photos), who, folder, say)
    key = _digest(*photos, style_name, who, seed, extra,
                  prompt(style_name, who, extra, (want or {}).get("glasses")))
    folder = folder / key
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "figure.stl").exists() and _from_cache(key, folder):
        say("Using the figure made before from these photos (shared cache).")
    render = folder / "style.png"
    if not render.exists():
        desc = ""
        if who != "person" or want is None:  # a person's look comes from the photo, glasses from the check
            look = folder / "look.txt"
            if not look.exists():
                say("Describing the photo (LFM3-VL-2B, under a minute)...")
            desc = describe(photos[0], look, who)
        refs = photos[:int(os.environ.get("STONE_STYLE_REFS", "1"))]
        face = face_crop(photos[0], folder / "face.png") if who == "person" else None
        if face:
            refs = [face, *refs]
        say("Styling the photo (FLUX.2 klein 4B, about 2 minutes)...")
        style_checked(refs, render, style_name, who, seed, " ".join(x for x in (desc, extra) if x),
                      want, say)
    mesh = folder / "lift.glb"
    if not mesh.exists():
        say("Lifting the render to 3D (Hunyuan3D-2mini, about 4 minutes)...")
        lift(render, mesh, seed)
    fig = folder / "figure.stl"
    if not fig.exists():
        repair(mesh, fig)
    _to_cache(key, folder)
    latest = folder.parent / "latest.txt"
    latest.write_text(str(fig) + "\n")
    return fig
