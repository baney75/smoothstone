"""The figure recipe without the style and lift models: a stand-in mesh goes through repair,
the plinth recipe and every print check, and the CLI points at stone find when photos are missing."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh

from smoothstone import cli, figure

REPO = Path(__file__).resolve().parent.parent


def stone(*argv):
    text, code = cli.run(list(argv))
    assert text.splitlines()[-1].startswith("NEXT: "), text
    return text, code, text.splitlines()[-1][6:]


def standin(path: Path) -> Path:
    """A little person, Y up and facing +Z like a lift model writes it: body, head, two feet."""
    body = trimesh.creation.capsule(height=0.9, radius=0.35)  # along Z
    body.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    body.apply_translation([0, 0.8, 0])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    head.apply_translation([0, 1.85, 0.05])
    feet = [trimesh.creation.box([0.25, 0.2, 0.4]) for _ in range(2)]
    feet[0].apply_translation([-0.15, 0.1, 0.05])
    feet[1].apply_translation([0.15, 0.1, 0.05])
    m = trimesh.boolean.union([body, head, *feet], engine="manifold")
    m.export(path)
    return path


def test_repair_orients_and_scales(tmp_path):
    info = figure.repair(standin(tmp_path / "lift.glb"), tmp_path / "figure.stl")
    m = trimesh.load(tmp_path / "figure.stl")
    lo, hi = m.bounds
    assert info["watertight"] and m.is_watertight
    assert abs((hi[2] - lo[2]) - 1) < 1e-6 and abs(lo[2]) < 1e-6  # unit height, feet on z = 0
    top = m.vertices[m.vertices[:, 2] > 0.8]
    assert top[:, 1].mean() < 0.02  # the face that pointed +Z now points -Y


def test_prompt_names_the_style():
    assert "chibi" in figure.prompt("chibi", "animal") and "this exact person" in figure.prompt("chibi")
    assert "large head" in figure.prompt("chibi")  # a person's chibi is the likeness figure
    assert "this exact animal" in figure.prompt("statue", "animal")
    with pytest.raises(figure.FigureProblem):
        figure.make([], Path("."), "nope")


def test_build_without_photos_asks_for_find(jobenv, monkeypatch):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    stone("new", "A mini figure of Ada Lovelace, 80 mm tall")
    stone("set", "figure", "height=80")
    text, code, nxt = stone("build")
    assert code == 1 and nxt == 'stone find "Ada Lovelace"'
    assert stone("status")[2] == 'stone find "Ada Lovelace"'


def test_figure_on_plinth_passes_checks(jobenv, monkeypatch, openscad, tmp_path):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    fig = tmp_path / "figure.stl"
    figure.repair(standin(tmp_path / "lift.glb"), fig)
    stone("new", 'A figure 70 mm tall with "ADA" on the base')
    stone("set", "figure", f"figure={fig}", "height=70", "name=ADA")
    text, code, nxt = stone("build")
    assert code == 0, text
    assert "Part ready" in text and "70.0 mm" in text


@pytest.fixture(autouse=True)
def figure_cache(tmp_path, monkeypatch):
    """Each test gets its own empty shared figure cache, never the machine's."""
    monkeypatch.setenv("STONE_FIGURE_CACHE", str(tmp_path / "figure-cache"))


def test_a_new_job_reuses_a_figure_made_before_without_any_model(tmp_path, monkeypatch):
    """The same photos and settings in a new job folder: no model runs, the same mesh comes back.
    This is how a Linux host (no style model) builds a figure made once on the Mac."""
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"jpeg")

    def fake_style(photos, out, *a, **k):
        shutil.copyfile(photo, out)
        return out

    monkeypatch.setattr(figure, "describe", lambda p, out, who: out.write_text("") and "")
    monkeypatch.setattr(figure, "style", fake_style)
    monkeypatch.setattr(figure, "lift", lambda render, out, seed: standin(out))
    a = figure.make([photo], tmp_path / "job1" / "fig", say=lambda s: None)

    def no_model(*a, **k):
        raise figure.FigureProblem("the style model is not set up")

    monkeypatch.setattr(figure, "describe", no_model)
    monkeypatch.setattr(figure, "style", no_model)
    monkeypatch.setattr(figure, "lift", no_model)
    b = figure.make([photo], tmp_path / "job2" / "fig", say=lambda s: None)
    assert b != a and b.read_bytes() == a.read_bytes()
    # only the finished mesh copied in (no style.png, no lift.glb): still no model
    c = figure.cache_dir() / figure.cache_key(a.parent.name)
    for name in ("style.png", "lift.glb", "look.txt"):
        (c / name).unlink()
    assert figure.make([photo], tmp_path / "job4" / "fig", say=lambda s: None).read_bytes() == a.read_bytes()
    monkeypatch.setenv("STONE_FIGURE_CACHE", "0")
    with pytest.raises(figure.FigureProblem):
        figure.make([photo], tmp_path / "job3" / "fig", say=lambda s: None)


def test_a_setting_that_changes_the_mesh_misses_the_cache(tmp_path, monkeypatch):
    """PR 8 review second-S1: a different lift octree must run the lift again, and a stand-in lift
    (STONE_LIFT) must never write a mesh into the shared cache."""
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"jpeg")
    calls = []

    def fake_style(photos, out, *a, **k):
        shutil.copyfile(photo, out)
        return out

    def fake_lift(render, out, seed):
        calls.append("lift")
        return standin(out)

    monkeypatch.setattr(figure, "describe", lambda p, out, who: out.write_text("") and "")
    monkeypatch.setattr(figure, "style", fake_style)
    monkeypatch.setattr(figure, "lift", fake_lift)
    figure.make([photo], tmp_path / "job1" / "fig", say=lambda s: None)
    figure.make([photo], tmp_path / "job2" / "fig", say=lambda s: None)
    assert calls == ["lift"]
    monkeypatch.setenv("STONE_LIFT_OCTREE", "384")
    figure.make([photo], tmp_path / "job3" / "fig", say=lambda s: None)
    assert calls == ["lift", "lift"]
    monkeypatch.setenv("STONE_LIFT", "python standin.py")
    monkeypatch.setenv("STONE_LIFT_OCTREE", "512")
    figure.make([photo], tmp_path / "job4" / "fig", say=lambda s: None)
    monkeypatch.delenv("STONE_LIFT")
    figure.make([photo], tmp_path / "job5" / "fig", say=lambda s: None)
    assert calls == ["lift", "lift", "lift", "lift"], "a stand-in lift's mesh was cached"


def test_cached_steps_skip_models(tmp_path, monkeypatch):
    """A second build with the same photos and settings reuses every file and runs no model."""
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"jpeg")
    calls = []

    def fake_style(photos, out, *a, **k):
        calls.append("style")
        shutil.copyfile(photo, out)
        return out

    def fake_lift(render, out, seed):
        calls.append("lift")
        standin(out)
        return out

    def fake_describe(photo, out, who):
        calls.append("describe")
        out.write_text("short silver hair\n")
        return "short silver hair"

    monkeypatch.setattr(figure, "describe", fake_describe)
    monkeypatch.setattr(figure, "style", fake_style)
    monkeypatch.setattr(figure, "lift", fake_lift)
    monkeypatch.setattr(figure, "traits", lambda *a, **k: None)
    a = figure.make([photo], tmp_path / "fig", say=lambda s: None)
    b = figure.make([photo], tmp_path / "fig", say=lambda s: None)
    assert a == b and a.exists() and calls == ["describe", "style", "lift"]


def test_find_leaves_other_plans_alone(jobenv, monkeypatch):
    """stone find on a fan guard job downloads nothing and keeps the job's NEXT."""
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    from smoothstone import find as findtool

    def no_fetch(*a, **k):
        raise AssertionError("find downloaded photos for a fan guard")

    monkeypatch.setattr(findtool, "fetch", no_fetch)
    stone("new", "A fan guard for a 120 mm fan")
    stone("set", "fan_guard", "size=120")
    before = stone("status")[2]
    text, code, nxt = stone("find", "fan guard")
    assert code == 1 and "figure" in text
    assert stone("status")[2] == before


def test_mini_alone_does_not_pick_figure(monkeypatch):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    from smoothstone import recipes as rc
    top = rc.rank(list(rc.load_all().values()), "a mini planter for a succulent")[0][1]
    assert top.name != "figure"


def test_find_before_a_plan_leaves_a_fan_guard_alone(jobenv, monkeypatch):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    from smoothstone import find as findtool

    def no_fetch(*a, **k):
        raise AssertionError("find downloaded photos for a fan guard")

    monkeypatch.setattr(findtool, "fetch", no_fetch)
    stone("new", "A guard for a 120 mm fan")
    text, code, nxt = stone("find", "fan guard")
    assert code == 1 and "figure" in text and nxt == "stone show fan_guard"


def test_own_photo_on_a_figure_plan_goes_to_build(jobenv, monkeypatch):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    stone("new", "A mini figure of my dog, 80 mm tall")
    stone("set", "figure", "height=80")
    assert stone("build")[2] == "stone image <path to photo>"
    photo = REPO / "examples" / "dog.png"
    if not photo.exists():
        photo = REPO / "evals" / "heldout" / "dog.png"
    text, code, nxt = stone("image", str(photo))
    assert nxt == "stone build", text
    assert stone("status")[2] == "stone build"


@pytest.mark.parametrize("request_text", ["A guard for a 120 mm fan", "A mini Abraham Lincoln, 80 mm tall",
                                          "A minifigure of Ada Lovelace", "a tiny Einstein"])
def test_find_never_loops_back_to_find(jobenv, monkeypatch, request_text):
    """A bare find, then a named find: the second NEXT is never another find that refuses."""
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    from smoothstone import find as findtool
    got = []
    monkeypatch.setattr(findtool, "fetch", lambda q, d, n=4: got.append(q) or [])
    stone("new", request_text)
    stone("find")
    text, code, nxt = stone("find", "fan guard" if "fan" in request_text else "Abraham Lincoln")
    if "fan" in request_text:
        assert not got and code == 1 and not nxt.startswith("stone find")
    else:
        assert got, text  # a subject with no part named is a figure


def _style_picture(path: Path) -> None:
    """A stand-in style picture: a skin-colored head over a black body on white."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (256, 512), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 170, 196, 470], fill=(20, 20, 20))
    d.rectangle([70, 470, 186, 505], fill=(20, 20, 20))
    d.ellipse([70, 20, 186, 180], fill=(225, 180, 150))
    im.save(path)


def test_stone_colors_end_to_end_on_a_figure_with_a_style_picture(jobenv, monkeypatch, openscad, tmp_path):
    """Review medium (cli.py:1299): cmd_colors on a real figure job, listing and then painting."""
    from smoothstone import colors

    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    d = tmp_path / "abc123"
    d.mkdir()
    fig = d / "figure.stl"
    figure.repair(standin(tmp_path / "lift.glb"), fig)
    _style_picture(d / "style.png")
    stone("new", 'A figure 70 mm tall with "ADA" on the base, all white but the body, which is black')
    stone("set", "figure", f"figure={fig}", "height=70", "name=ADA")
    text, code, nxt = stone("build")
    assert code == 0 and "Part ready" in text, text
    assert nxt.startswith("stone colors"), text

    text, code, nxt = stone("colors")
    assert code == 0, text
    assert "Color regions of figure.stl" in text and "base" in text
    assert "% of surface" in text
    assert nxt.startswith("stone colors ") and "rest=" in nxt, text
    regions = json.loads((jobenv / "stone-job" / "build" / "regions.json").read_text())
    assert regions["anatomy"]["top"] > 0

    text, code, nxt = stone(*nxt.split()[1:])
    assert code == 0, text
    assert "PASS color_edges" in text, text
    out = jobenv / "stone-job" / "figure.3mf"
    assert out.is_file(), text
    n = len(colors.read_3mf(out)[1])
    assert colors.check_3mf(out, n, len(status_json_colors(jobenv)["filaments"])) == []
    assert nxt == cli.DONE_NEXT
    assert (jobenv / "stone-job" / "build" / "views.png").is_file()

    # SPEC3 C: the eval grader reads the parts' colors back from the 3MF
    import sys

    sys.path.insert(0, str(REPO / "evals"))
    import run as evrun

    job = jobenv / "stone-job"
    got = evrun.grade_colors(job, "figure", {"rest": "none"})["parts"]
    assert "base" in got and len(set(got.values())) >= 2, got
    rest = got["base"]
    want: dict = {"rest": rest}
    for part, c in got.items():
        if c != rest:
            want.setdefault(c, []).append(part)
    g = evrun.grade_colors(job, "figure", want)
    assert g["why"] == [] and abs(sum(g["shares"].values()) - 1) < 0.01, g
    odd = next(p for p, c in got.items() if c != rest)
    wrong = {"rest": rest, rest: [odd]}
    assert f"{odd} is {got[odd]} (want {rest})" in evrun.grade_colors(job, "figure", wrong)["why"]

    # PR 6 review r2 sol-sol-F4: a part painted 60% right and 40% wrong must fail, though its
    # majority is right. Repaint 40% of the odd part's area (by region) in the rest color.
    import numpy as np

    V, F, codes, main = colors.read_3mf(job / "figure.3mf")
    fil = status_json_colors(jobenv)["filaments"]
    face = np.array([colors.decode_paint(c) if c else main for c in codes])
    num = np.load(job / "build" / "paint.npy")
    regs = json.loads((job / "build" / "regions.json").read_text())["regions"]
    sel = np.isin(num, [d["n"] for d in regs if (d.get("part") or "other") == odd])
    area = colors.load_stl_mesh(V, F).area_faces
    idx = np.where(sel & (face == fil.index(got[odd]) + 1))[0]
    flip = idx[np.cumsum(area[idx]) <= 0.4 * area[sel].sum()]
    face[flip] = fil.index(rest) + 1
    colors.write_3mf(job / "figure.3mf", V, F, face, "figure")
    why = evrun.grade_colors(job, "figure", want)["why"]
    assert any(w.startswith(f"{odd} is ") and f"not {got[odd]}" in w for w in why), why


def test_a_failing_color_edge_never_points_back_at_the_same_mapping(jobenv, monkeypatch, openscad, tmp_path):
    """Review first-F1: a ledge fault must give a NEXT that changes the mapping (the region on the
    failing edge takes the rest color, or the part prints in one color), so following NEXT ends in
    a written 3MF instead of a loop."""
    import numpy as np

    from smoothstone import colors

    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    d = tmp_path / "abc123"
    d.mkdir()
    fig = d / "figure.stl"
    figure.repair(standin(tmp_path / "lift.glb"), fig)
    _style_picture(d / "style.png")
    stone("new", 'A figure 70 mm tall with "ADA" on the base, all white but the body, which is black')
    stone("set", "figure", f"figure={fig}", "height=70", "name=ADA")
    text, code, nxt = stone("build")
    assert code == 0 and nxt.startswith("stone colors"), text

    real = colors.ledges

    def always_a_ledge(V, F, face_fil, lines):  # every two-color paint climbs a ledge
        out = real(V, F, face_fil, lines)
        if len(np.unique(face_fil)) > 1 and out:
            out[0] = {**out[0], "area": 1.0, "angle": 15.0, "fil": int(face_fil.max()), "faces": 9}
        return out

    monkeypatch.setattr(colors, "ledges", always_a_ledge)
    if nxt == "stone colors":
        text, code, nxt = stone("colors")
    tried = []
    for _ in range(6):
        mapping = nxt
        text, code, nxt = stone(*mapping.split()[1:])
        tried.append(mapping)
        if nxt == cli.DONE_NEXT:
            break
        assert "climbs onto a ledge" in text, text
        assert nxt != mapping and nxt != "stone colors", text  # never the mapping that just failed
    assert nxt == cli.DONE_NEXT, tried
    assert len(tried) > 1, tried
    out = jobenv / "stone-job" / "figure.3mf"
    assert out.is_file()
    assert len(status_json_colors(jobenv)["shares"].split(";")) == 1  # the one-color print


def status_json_colors(work: Path) -> dict:
    return json.loads((work / "stone-job" / "status.json").read_text())["colors"]


def test_a_build_after_stone_colors_with_nothing_changed_keeps_the_paint(jobenv, monkeypatch, openscad, tmp_path):
    """Review r2 second-N2: a small model that reads a FIX from stone colors as "fix, then stone
    build again" must not undo the paint: the rebuild wiped the 3MF and re-suggested the first
    mapping, so a retry looped forever. With the plan unchanged the build is skipped."""
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    d = tmp_path / "abc123"
    d.mkdir()
    fig = d / "figure.stl"
    figure.repair(standin(tmp_path / "lift.glb"), fig)
    _style_picture(d / "style.png")
    stone("new", 'A figure 70 mm tall with "ADA" on the base, all white but the body, which is black')
    stone("set", "figure", f"figure={fig}", "height=70", "name=ADA")
    text, code, nxt = stone("build")
    assert code == 0 and nxt.startswith("stone colors"), text
    text, code, nxt = stone("colors")
    text, code, nxt = stone(*nxt.split()[1:])
    assert nxt == cli.DONE_NEXT, text
    painted = jobenv / "stone-job" / "figure.3mf"
    before = painted.read_bytes()
    text, code, nxt = stone("build")
    assert code == 0, text
    assert painted.exists() and painted.read_bytes() == before, text
    assert nxt == cli.DONE_NEXT and "Nothing changed" in text, text
    assert status_json_colors(jobenv), "the colors were dropped from status"
    # a real change still rebuilds, and drops the old paint
    stone("set", "figure", f"figure={fig}", "height=72", "name=ADA")
    text, code, nxt = stone("build")
    assert "Nothing changed" not in text and not painted.exists(), text


def _stub_figure(monkeypatch, fig: Path) -> list:
    """figure_step stand-in for the tier-S flow (no figure= given): records the figure as made."""
    calls = []

    def made(job, r, params, out):
        calls.append(1)
        d = job.image_dir / "figure"
        d.mkdir(parents=True, exist_ok=True)
        (d / "latest.txt").write_text(str(fig))
        return fig
    monkeypatch.setattr(cli, "figure_step", made)
    return calls


def _painted_job(jobenv, monkeypatch, tmp_path):
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    d = tmp_path / "abc123"
    d.mkdir()
    fig = d / "figure.stl"
    figure.repair(standin(tmp_path / "lift.glb"), fig)
    _style_picture(d / "style.png")
    calls = _stub_figure(monkeypatch, fig)
    stone("new", 'A figure 70 mm tall with "ADA" on the base, all white but the body, which is black')
    stone("set", "figure", "height=70", "name=ADA")
    text, code, nxt = stone("build")
    assert code == 0 and nxt.startswith("stone colors"), text
    text, code, nxt = stone("colors")
    text, code, nxt = stone(*nxt.split()[1:])
    assert nxt == cli.DONE_NEXT, text
    return fig, calls, jobenv / "stone-job" / "figure.3mf"


def test_the_tier_s_flow_skips_a_no_op_build_before_the_figure_step(jobenv, monkeypatch, openscad, tmp_path):
    """Review PR 9 second-F4: S.md sets only height and name, so the figure comes from the figure
    step. A build after stone colors must skip without running that step again."""
    fig, calls, painted = _painted_job(jobenv, monkeypatch, tmp_path)
    before = painted.read_bytes()
    text, code, nxt = stone("build")
    assert "Nothing changed" in text and nxt == cli.DONE_NEXT, text
    assert painted.read_bytes() == before
    assert len(calls) == 1, "the figure step ran again on a build that changed nothing"


def test_a_new_figure_at_the_same_path_rebuilds(jobenv, monkeypatch, openscad, tmp_path):
    """Review PR 9 first-F2: the skip must key on what the build reads, not on a path string."""
    fig, calls, painted = _painted_job(jobenv, monkeypatch, tmp_path)
    m = trimesh.load(fig)
    m.apply_scale([1.0, 1.0, 0.9])
    m.export(fig)
    text, code, nxt = stone("build")
    assert "Nothing changed" not in text, text


def test_an_edited_recipe_source_rebuilds(jobenv, monkeypatch, openscad, tmp_path):
    fig, calls, painted = _painted_job(jobenv, monkeypatch, tmp_path)
    own = tmp_path / "edited_recipes"
    shutil.copytree(REPO / "recipes", own)
    monkeypatch.setenv("STONE_RECIPES", str(own))
    text, code, nxt = stone("build")
    assert "Nothing changed" in text, text  # a copy with the same source is the same recipe
    with open(own / "figure" / "recipe.scad", "a") as f:
        f.write("\n// edited\n")
    text, code, nxt = stone("build")
    assert "Nothing changed" not in text, text


def test_the_style_prompt_takes_the_kind_wikidata_gave_the_found_subject(jobenv, monkeypatch):
    """An elephant or a golden retriever is an animal and a tower is an object, though neither
    word is in the animal list; only a photo the owner gives falls back to the request's words."""
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    stone("new", "A figurine of an african elephant, 80 mm tall")
    job = cli.Job(jobenv / "stone-job")
    found = job.image_dir / "found"
    found.mkdir(parents=True)
    (found / "found.json").write_text(json.dumps({"query": "african elephant", "kind": "animal"}))
    assert cli.figure_who(job, "chibi") == "animal"
    (found / "found.json").write_text(json.dumps({"query": "Eiffel Tower", "kind": "object"}))
    assert cli.figure_who(job, "chibi") == "object"
    assert cli.figure_who(job, "object") == "object"


ELON_LOOK = ("The person has light brown, wavy hair, no glasses, no facial hair, appears to be middle-aged, "
             "and is of fair complexion. They are wearing a black suit jacket, a white shirt, and a black tie.")


def test_a_person_without_glasses_gets_no_glasses_in_the_style_prompt():
    """The Elon render came back with thick black glasses: the prompt named glasses twice, once in
    the likeness line and once as 'no glasses' from the description, and naming them draws them."""
    p = figure.prompt("chibi", "person", ELON_LOOK)
    assert "glasses" not in p.lower() and "facial hair" not in p.lower()


def test_a_person_is_styled_from_the_photo_not_from_the_description():
    """Owner: "Those things look nothing like elon". The description's "light brown hair" made his
    hair too light: the blind likeness judge gave 8 with the description in the prompt and 9 on
    both seeds without it, so for a person the photo alone carries the look."""
    p = figure.prompt("chibi", "person", ELON_LOOK)
    assert "light brown" not in p and "middle-aged" not in p
    assert "the real face from the photo" in p and "the same jaw" in p and "large head" in p
    assert "expression" in p and "smile" not in p  # a smile he does not have was drawn on
    assert "friendly smile" in figure.prompt("chibi", "animal")  # animals keep the chibi look
    assert "light brown" in figure.prompt("chibi", "animal", "light brown fur.")


def test_a_person_with_glasses_keeps_them_exactly():
    p = figure.prompt("chibi", "person", "The person has short grey hair and thin round glasses. They wear a jacket "
                      "with no tie.")
    assert "same frame shape" in p and "tie" not in p


def test_the_photo_check_decides_the_glasses_over_the_description():
    """The traits check looked at the photo; the description is one model's sentence about it."""
    assert "same frame shape" in figure.prompt("chibi", "person", "", glasses=True)
    assert "glasses" not in figure.prompt("chibi", "person", "He wears glasses.", glasses=False).lower()


def test_the_positive_words_of_a_description():
    assert figure.positive("short hair, no beard, without a hat, a red shirt.") == "short hair, a red shirt."
    assert figure.positive("He is not wearing glasses. He has a black jacket.") == "He has a black jacket."


def _models(tmp_path, monkeypatch, answers):
    """Fake describe, style and lift; traits answers by file name. Returns the style calls."""
    calls = []

    def fake_style(photos, out, style_name, who, seed, extra="", glasses=None, plain=False):
        calls.append((photos[0].name, seed) + (("plain",) if plain else ()))
        out.write_bytes(f"render {seed}".encode())
        return out

    def fake_traits(images, who, log):
        return [answers(Path(i).name) for i in images]

    monkeypatch.setattr(figure, "describe", lambda photo, out, who: (out.write_text("hair\n"), "hair")[1])
    monkeypatch.setattr(figure, "style", fake_style)
    monkeypatch.setattr(figure, "lift", lambda render, out, seed: standin(out))
    monkeypatch.setattr(figure, "traits", fake_traits)
    monkeypatch.setattr(figure, "face_crop", lambda photo, out: None)
    return calls


BARE = {"face": True, "glasses": False, "beard": False, "hat": False}


def test_a_photo_with_no_face_never_sets_the_likeness(tmp_path, monkeypatch):
    """For "DHH" the finder once put a protein first, and the describe model then wrote a man with
    gold glasses from it. Only a photo that shows the face may lead; a cap goes behind a bare head."""
    photos = []
    for name in ("protein.png", "cap.jpg", "talk.jpg"):
        photos.append(tmp_path / name)
        photos[-1].write_bytes(name.encode())
    said = []
    calls = _models(tmp_path, monkeypatch, lambda n: {"protein.png": {"face": False},
                                                      "cap.jpg": {**BARE, "hat": True}}.get(n, BARE))
    figure.make(photos, tmp_path / "fig", say=said.append)
    assert calls[0][0] == "talk.jpg"
    assert any("Left out 1 photo with no face" in x for x in said)


def test_no_photo_with_a_face_stops_the_figure_with_a_way_on(tmp_path, monkeypatch):
    photo = tmp_path / "car.jpg"
    photo.write_bytes(b"car")
    _models(tmp_path, monkeypatch, lambda n: {"face": False})
    with pytest.raises(figure.FigureProblem, match="stone image"):
        figure.make([photo], tmp_path / "fig", say=lambda s: None)


def test_a_render_that_adds_glasses_is_styled_again_with_the_next_seed(tmp_path, monkeypatch):
    """The Elon render came back with thick black glasses he does not wear."""
    photo = tmp_path / "elon.jpg"
    photo.write_bytes(b"elon")
    calls = _models(tmp_path, monkeypatch,
                    lambda n: {**BARE, "glasses": n == "style-7.png"})
    said = []
    fig = figure.make([photo], tmp_path / "fig", seed=7, say=said.append)
    assert [c[1] for c in calls] == [7, 8]
    assert (fig.parent / "style.png").read_bytes() == b"render 8"
    assert json.loads((fig.parent / "check.json").read_text())["differs"] == []
    assert any("added glasses" in x and "seed 8" in x for x in said)


def test_a_render_with_lettering_on_the_clothes_is_styled_again_in_plain_colors(tmp_path, monkeypatch):
    """DHH's lead photo is him in a race suit, and the render copied its sponsor logos and an
    "Abu Dhabi Racing" shirt: lettering a figure cannot print or paint, and other people's marks."""
    photo = tmp_path / "dhh.jpg"
    photo.write_bytes(b"dhh")
    calls = _models(tmp_path, monkeypatch, lambda n: {**BARE, "lettering": n in ("dhh.jpg", "style-7.png")})
    said = []
    fig = figure.make([photo], tmp_path / "fig", seed=7, say=said.append)
    assert calls == [("dhh.jpg", 7), ("dhh.jpg", 8, "plain")]
    assert (fig.parent / "style.png").read_bytes() == b"render 8"
    assert any("lettering" in x and "plain" in x and "seed 8" in x for x in said)


def test_a_render_with_plain_clothes_keeps_the_clothes_it_has():
    """The plain look is only for a render that came back with lettering: as the first try it
    turned Jensen's leather jacket into a sweater."""
    same, plain = figure.prompt("chibi", "person"), figure.prompt("chibi", "person", plain=True)
    assert "the same clothes," in same and "plain solid colors" not in same
    assert "plain solid colors" in plain and "the same clothes," not in plain


def test_the_render_check_asks_about_lettering_on_the_clothes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("describe", Path(__file__).resolve().parents[1] / "tools" / "figure" / "describe.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    assert "logo" in d.TRAITS["person"]["lettering"]


def test_a_render_that_never_matches_keeps_the_closest(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"p")
    beard = {"style-7.png": {**BARE, "glasses": True, "beard": True}, "style-8.png": {**BARE, "glasses": True}}
    calls = _models(tmp_path, monkeypatch, lambda n: beard.get(n, {**BARE, "hat": True, "beard": True}
                                                               if n.startswith("style") else BARE))
    fig = figure.make([photo], tmp_path / "fig", seed=7, say=lambda s: None)
    assert [c[1] for c in calls] == [7, 8, 9]
    assert (fig.parent / "style.png").read_bytes() == b"render 8"


def test_the_face_crop_leads_the_style_references(tmp_path, monkeypatch):
    """With the whole photo alone the face is a few pixels of a 768 px render and comes out generic;
    the face crop first and the photo second took the likeness judge from 6 to 9 on Elon."""
    photo = tmp_path / "elon.jpg"
    photo.write_bytes(b"elon")
    refs = []
    _models(tmp_path, monkeypatch, lambda n: BARE)
    monkeypatch.setattr(figure, "style", lambda photos, out, *a, **k: (refs.append([p.name for p in photos]),
                                                                      out.write_bytes(b"r"))[1] and out)

    def crop(src, out):
        out.write_bytes(b"face")
        return out
    monkeypatch.setattr(figure, "face_crop", crop)
    figure.make([photo], tmp_path / "fig", say=lambda s: None)
    assert refs[0] == ["face.png", "elon.jpg"]


def test_no_face_crop_for_an_animal_or_when_it_fails(tmp_path, monkeypatch):
    photo = tmp_path / "dog.jpg"
    photo.write_bytes(b"dog")
    refs = []
    _models(tmp_path, monkeypatch, lambda n: BARE)
    monkeypatch.setattr(figure, "style", lambda photos, out, *a, **k: (refs.append([p.name for p in photos]),
                                                                      out.write_bytes(b"r"))[1] and out)
    monkeypatch.setattr(figure, "face_crop", lambda src, out: (_ for _ in ()).throw(AssertionError("cropped")))
    figure.make([photo], tmp_path / "a", who="animal", say=lambda s: None)
    monkeypatch.setattr(figure, "face_crop", lambda src, out: None)
    figure.make([photo], tmp_path / "p", say=lambda s: None)
    assert refs == [["dog.jpg"], ["dog.jpg"]]


def test_face_crop_fails_open_without_a_face_or_a_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("STONE_MFLUX", str(tmp_path / "nowhere"))
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"p")
    assert figure.face_crop(photo, tmp_path / "face.png") is None


def test_a_fictional_character_is_styled_as_that_character(jobenv, monkeypatch):
    """Review of PR 23: Mario or a superhero is no human on Wikidata, and was styled as an object."""
    monkeypatch.setenv("STONE_RECIPES", str(REPO / "recipes"))
    stone("new", "A chibi figure of Snoopy, 70 mm tall")
    job = cli.Job(jobenv / "stone-job")
    found = job.image_dir / "found"
    found.mkdir(parents=True)
    (found / "found.json").write_text(json.dumps({"query": "Snoopy", "kind": "character"}))
    assert cli.figure_who(job, "chibi") == "character"
    p = figure.prompt("chibi", "character")
    assert "this exact character" in p and "costume" in p


def test_the_owners_own_photo_is_never_stopped_by_the_face_check(tmp_path, monkeypatch):
    """Review of PR 23: a small or side-on photo the owner gave read as no face, with no way on."""
    photo = tmp_path / "source.jpg"
    photo.write_bytes(b"me")
    calls = _models(tmp_path, monkeypatch, lambda n: {"face": False} if n == "source.jpg" else BARE)
    figure.make([photo], tmp_path / "fig", say=lambda s: None)
    assert calls[0][0] == "source.jpg"


def test_a_render_is_checked_only_for_what_the_photo_check_answered(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"p")
    asked = []
    _models(tmp_path, monkeypatch, lambda n: {"face": True})
    real = figure.traits
    monkeypatch.setattr(figure, "traits", lambda images, who, log: (asked.append(images), real(images, who, log))[1])
    figure.make([photo], tmp_path / "fig", say=lambda s: None)
    assert len(asked) == 1  # the photo only: no glasses, beard or hat answer to hold a render to


def _dies_mid_write(target_arg):
    """A stand-in subprocess.run that writes half a file to the path it was given, then fails."""
    def run(cmd, **kw):
        Path(target_arg(cmd)).write_bytes(b"half")
        return type("R", (), {"returncode": -9})()
    return run


def _writes(target_arg, data=b"done"):
    def run(cmd, **kw):
        Path(target_arg(cmd)).write_bytes(data)
        return type("R", (), {"returncode": 0})()
    return run


def test_a_style_cut_off_mid_write_leaves_no_picture_under_the_final_name(tmp_path, monkeypatch):
    """first-F5: the harness kills a slow build mid-write, and the next run took the half picture as done."""
    monkeypatch.setattr(figure, "_mflux_bin", lambda: tmp_path / "mflux-generate")
    monkeypatch.setattr(figure, "_style_model", lambda: [])
    out_arg = lambda cmd: cmd[cmd.index("--output") + 1]
    out = tmp_path / "style.png"
    monkeypatch.setattr(figure.subprocess, "run", _dies_mid_write(out_arg))
    with pytest.raises(figure.FigureProblem):
        figure.style([tmp_path / "p.jpg"], out, "chibi", "person", 7)
    assert not out.exists()
    monkeypatch.setattr(figure.subprocess, "run", _writes(out_arg))
    assert figure.style([tmp_path / "p.jpg"], out, "chibi", "person", 7).read_bytes() == b"done"
    assert [p.name for p in tmp_path.glob("style*.png")] == ["style.png"]


def test_a_lift_cut_off_mid_write_leaves_no_mesh_under_the_final_name(tmp_path, monkeypatch):
    monkeypatch.setenv("STONE_LIFT", "lift")
    out = tmp_path / "lift.glb"
    monkeypatch.setattr(figure.subprocess, "run", _dies_mid_write(lambda cmd: cmd[2]))
    with pytest.raises(figure.FigureProblem):
        figure.lift(tmp_path / "style.png", out, 7)
    assert not out.exists()
    monkeypatch.setattr(figure.subprocess, "run", _writes(lambda cmd: cmd[2]))
    assert figure.lift(tmp_path / "style.png", out, 7).read_bytes() == b"done"
    assert [p.name for p in tmp_path.glob("lift*.glb")] == ["lift.glb"]


def test_a_repair_that_dies_mid_write_leaves_no_figure_under_the_final_name(tmp_path, monkeypatch):
    src = standin(tmp_path / "lift.glb")
    out = tmp_path / "figure.stl"

    def half(self, path, *a, **k):
        Path(path).write_bytes(b"half")
        raise OSError("killed")
    with monkeypatch.context() as m:
        m.setattr(trimesh.Trimesh, "export", half)
        with pytest.raises(OSError):
            figure.repair(src, out)
    assert not out.exists()
    figure.repair(src, out)
    assert trimesh.load(out).is_watertight and out.with_suffix(".json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["figure.json", "figure.stl", "lift.glb"]


def test_a_style_check_cut_off_while_copying_leaves_no_picture_under_the_final_name(tmp_path, monkeypatch):
    """style_checked copies the kept try onto style.png; a half copy must not stand as the render."""
    monkeypatch.setattr(figure, "style", lambda photos, out, *a, **k: (out.write_bytes(b"render"), out)[1])
    monkeypatch.setattr(figure, "traits", lambda *a, **k: [{"glasses": False}])

    def half(src, dst, *a, **k):
        Path(dst).write_bytes(b"half")
        raise OSError("killed")
    monkeypatch.setattr(figure.shutil, "copyfile", half)
    out = tmp_path / "style.png"
    with pytest.raises(OSError):
        figure.style_checked([tmp_path / "p.jpg"], out, "chibi", "person", 7, "", {"glasses": False})
    assert not out.exists()
