from smoothstone import check

CUBE_V = ["0 0 0", "1 0 0", "1 1 0", "0 1 0", "0 0 1", "1 0 1", "1 1 1", "0 1 1"]
CUBE_F = ["4 0 3 2 1", "4 4 5 6 7", "4 0 1 5 4", "4 1 2 6 5", "4 2 3 7 6", "4 3 0 4 7"]


def _write(path, header_lines):
    path.write_text("\n".join(header_lines + CUBE_V + CUBE_F) + "\n")
    return check.load_off(path)


def test_counts_on_their_own_line(tmp_path):
    mesh = _write(tmp_path / "a.off", ["OFF", "8 6 0"])
    assert len(mesh.vertices) == 8 and len(mesh.faces) == 12


def test_counts_on_the_header_line(tmp_path):
    # Newer OpenSCAD nightlies write "OFF nv nf ne" on one line.
    mesh = _write(tmp_path / "b.off", ["OFF 8 6 0"])
    assert len(mesh.vertices) == 8 and len(mesh.faces) == 12
    assert mesh.is_watertight
