"""Print checks on an STL: watertight, one body, on the bed, walls, overhangs, dims."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

MIN_WALL = 0.8  # mm
WALL_SAMPLES = 3000
WALL_PERCENTILE = 3
OVERHANG_COS = math.cos(math.radians(45))
REST_TOL = 0.05  # mm
CONTACT_TOL = 0.1  # mm
BED_SKIP = 0.2  # mm


@dataclass
class Result:
    name: str
    ok: bool
    value: object = None
    detail: str = ""
    # default advice when no recipe parameter is known
    advice: str = ""
    # "up" or "down" when growing or shrinking a parameter helps
    direction: str = ""
    need: float | None = None
    measured: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def load_off(path) -> "trimesh.Trimesh":
    """Read an OFF file as written by OpenSCAD, keeping its shared vertices. Polygons with more
    than three corners (OpenSCAD passes polyhedron faces through untouched) become triangle fans."""
    import trimesh

    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines or not lines[0].startswith("OFF"):
        raise ValueError("not an OFF file")
    # OFF allows the counts on the header line itself ("OFF nv nf ne"), which newer OpenSCAD writes.
    head = lines[0].split()
    if len(head) >= 3:
        nv, nf = int(head[1]), int(head[2])
        body = lines[1:]
    else:
        nv, nf = (int(w) for w in lines[1].split()[:2])
        body = lines[2:]
    verts = np.array([ln.split()[:3] for ln in body[:nv]], dtype=float).reshape(nv, 3)
    tris = []
    for ln in body[nv:nv + nf]:
        w = ln.split()  # count, indices, then optional face color
        k = int(w[0])
        idx = [int(x) for x in w[1:1 + k]]
        tris.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1))
    return trimesh.Trimesh(verts, np.array(tris, dtype=np.int64).reshape(-1, 3), process=False)


def _load(path: Path):
    import trimesh

    if Path(path).suffix.lower() == ".off":
        return load_off(path)

    # An OFF from OpenSCAD carries the exact shared vertices; merging by position would only add
    # errors. STL has no shared vertices, so it is merged.
    mesh = trimesh.load(str(path), force="mesh", process=Path(path).suffix.lower() != ".off")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("no mesh in file")
    return mesh


def _ray_hits(mesh, origins, dirs):
    """First hit locations per ray. Returns (index_ray, locations) or None."""
    try:
        loc, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        return idx_ray, loc
    except ImportError:
        return None


def thinnest_wall(mesh, samples: int = WALL_SAMPLES, seed: int = 0) -> float | None:
    import trimesh

    try:
        pts, fi = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    except TypeError:
        pts, fi = trimesh.sample.sample_surface(mesh, samples)
    normals = mesh.face_normals[fi]
    origins = pts - normals * 1e-4
    got = _ray_hits(mesh, origins, -normals)
    if got is None:
        return None
    idx_ray, loc = got
    if len(idx_ray) == 0:
        return None
    dist = np.linalg.norm(loc - origins[idx_ray], axis=1)
    dist = dist[dist > 5e-3]  # drop self hits from float error
    if len(dist) == 0:
        return None
    return float(np.percentile(dist, WALL_PERCENTILE))


def _hull_area(xy: np.ndarray) -> float:
    from scipy.spatial import ConvexHull, QhullError

    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)
    except (QhullError, ValueError):
        return 0.0


def check_mesh(mesh, recipe=None, params=None, bed=None) -> list[Result]:
    out: list[Result] = []
    bed = bed or config.DEFAULT_BED
    max_bodies = recipe.max_bodies if recipe else 1
    max_overhang = recipe.max_overhang if recipe else 0.05

    out.append(Result("loads", len(mesh.faces) > 0, len(mesh.faces), f"{len(mesh.faces)} faces",
                      advice="the STL has no faces; rebuild the part"))
    if len(mesh.faces) == 0:
        return out

    wt = bool(mesh.is_watertight)
    out.append(Result("watertight", wt, wt, "closed surface" if wt else "surface has holes",
                      advice="the surface has holes; go back to the default parameters"))

    from .clean import problems, slicer_view

    bad = problems(slicer_view(mesh.vertices, mesh.faces))
    out.append(Result("slicer", not bad, bad or "ok", "; ".join(bad) or "no non-manifold edges or zero-area faces",
                      advice="a slicer will report this mesh as broken; rebuild the part with stone build"))

    bodies = len(mesh.split(only_watertight=False))
    out.append(Result("bodies", bodies <= max_bodies, bodies,
                      f"{bodies} bodies (max {max_bodies})",
                      advice="the part is in separate pieces; make them touch or overlap"))

    vol = float(mesh.volume) if wt else float(abs(mesh.volume))
    out.append(Result("volume", vol > 0, round(vol, 1), f"{_fmt(vol)} mm3",
                      advice="the part has no volume; go back to the default parameters"))

    lo, hi = mesh.bounds
    ext = hi - lo
    fits = bool(all(ext[i] <= bed[i] + 1e-6 for i in range(3)))
    worst = max(ext[i] / bed[i] for i in range(3))
    out.append(Result("bed", fits, [round(float(e), 2) for e in ext],
                      f"{_fmt(ext[0])} x {_fmt(ext[1])} x {_fmt(ext[2])} mm on a "
                      f"{_fmt(bed[0], 0)} x {_fmt(bed[1], 0)} x {_fmt(bed[2], 0)} mm bed",
                      advice="the part is bigger than the bed; make it smaller",
                      direction="down", need=1.0, measured=float(worst)))

    zmin = float(lo[2])
    rest = abs(zmin) <= REST_TOL
    out.append(Result("resting", rest, round(zmin, 3), f"lowest point z = {zmin:.2f} mm",
                      advice=f"move the part so its lowest point is at z = 0 (now {zmin:.2f} mm)"))

    nz = mesh.face_normals[:, 2]
    tri_z = mesh.triangles[:, :, 2]
    areas = mesh.area_faces
    down = (nz < -0.99) & (tri_z.max(axis=1) <= zmin + CONTACT_TOL)
    contact = float(areas[down].sum())
    bbox_area = float(ext[0] * ext[1])
    need = min(200.0, 0.15 * bbox_area)
    ok = contact >= need
    hull = 0.0
    if not ok and 0.15 * bbox_area < 200.0:
        # small part: 30 percent of the footprint hull is enough
        hull = _hull_area(mesh.vertices[:, :2])
        ok = contact >= 0.3 * hull
    out.append(Result("first_layer", ok, round(contact, 1),
                      f"{_fmt(contact)} mm2 on the bed (need {_fmt(need)} mm2)",
                      advice="too little of the part touches the bed; give it a flat base",
                      direction="up", need=need, measured=contact))

    wall = thinnest_wall(mesh)
    need_wall = recipe.min_wall if recipe else MIN_WALL
    if wall is None:
        out.append(Result("min_wall", True, None, "skipped (no ray engine or no hits)"))
    else:
        out.append(Result("min_wall", wall >= need_wall, round(wall, 2),
                          f"thinnest wall {wall:.2f} mm (need {need_wall} mm)",
                          advice="a wall is too thin; make it thicker",
                          direction="up", need=need_wall, measured=wall))

    face_zmin = tri_z.min(axis=1)
    over = (nz < -OVERHANG_COS - 1e-3) & (face_zmin > zmin + BED_SKIP)
    total = float(areas.sum())
    frac = float(areas[over].sum()) / total if total > 0 else 0.0
    out.append(Result("overhang", frac <= max_overhang, round(frac, 4),
                      f"{frac * 100:.1f}% of the surface hangs past 45 deg "
                      f"(max {max_overhang * 100:.0f}%)",
                      advice="too much of the part hangs in the air; add a chamfer or flip it"))

    if recipe is not None and recipe.dims:
        try:
            want = recipe.expected_dims(recipe.merged(params or {}))
        except (ValueError, TypeError, ZeroDivisionError, KeyError) as e:
            out.append(Result("dims", True, None, f"skipped ({e})"))
        else:
            tol = recipe.dims_tol
            axes = {"x": 0, "y": 1, "z": 2}
            bad = []
            for axis, w in want.items():
                got = float(ext[axes[axis]])
                if abs(got - w) > tol:
                    bad.append(f"{axis} {got:.1f} mm, expected {w:.1f} mm")
            detail = "; ".join(bad) if bad else "size matches the recipe"
            out.append(Result("dims", not bad, bad or "ok", detail,
                              advice="size does not match the recipe; go back to the defaults"))
    return out


def check_stl(path, recipe=None, params=None, bed=None) -> list[Result]:
    path = Path(path)
    try:
        mesh = _load(path)
    except Exception as e:  # trimesh raises many types on bad files
        return [Result("loads", False, None, f"cannot read {path.name}: {type(e).__name__}",
                       advice="the STL is missing or broken; rebuild the part")]
    return check_mesh(mesh, recipe=recipe, params=params, bed=bed)


def passed(results: list[Result]) -> bool:
    return all(r.ok for r in results)


def measured_size(results: list[Result]) -> list[float] | None:
    for r in results:
        if r.name == "bed" and isinstance(r.value, list):
            return r.value
    return None


def format_results(results: list[Result], fix_line=None) -> list[str]:
    """One line per check: PASS, or FIX with the measured problem and one change.

    fix_line(result) may return the change text (for example "set h=1.2").
    """
    lines = []
    for r in results:
        if r.ok:
            lines.append(f"PASS {r.name}: {r.detail}")
        else:
            custom = fix_line(r) if fix_line else None
            lines.append(f"FIX: {r.name} {r.detail}; {custom or r.advice}")
    return lines
