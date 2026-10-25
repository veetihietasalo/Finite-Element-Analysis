"""Module-level checks: imports, materials table, demo, plot hook."""
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import fem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_import_has_no_side_effects(tmp_path):
    code = "import sys, fem; print('mayavi' in sys.modules)"
    env = dict(os.environ, PYTHONPATH=REPO)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert list(tmp_path.iterdir()) == []


def test_materials_database(tmp_path):
    db = str(tmp_path / "materials.db")
    fem.create_materials_database(db)
    fem.create_materials_database(db)  # running twice keeps one row per material
    props = fem.assign_material_properties([(0, 1, 2, 3)] * 3, db, "Steel")
    assert props == [{'youngs_modulus': 210000.0, 'poisson_ratio': 0.3}] * 3
    assert fem.assign_material_properties([(0, 1, 2, 3)], db, "Concrete")[0]['youngs_modulus'] == 30000.0
    with pytest.raises(ValueError):
        fem.assign_material_properties([(0, 1, 2, 3)], db, "Unobtainium")


def test_demo_runs(tmp_path):
    result = subprocess.run([sys.executable, os.path.join(REPO, "fem.py"), "--divisions", "10", "1", "1"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "tip deflection" in result.stdout
    assert (tmp_path / "materials.db").exists()


def test_plot_mesh_imports_mayavi_only_when_called(monkeypatch):
    calls = {}
    mlab = types.SimpleNamespace(
        triangular_mesh=lambda x, y, z, triangles, **kw: calls.update(x=x, z=z, triangles=triangles, kw=kw),
        show=lambda: calls.update(shown=True),
    )
    mayavi = types.ModuleType("mayavi")
    mayavi.mlab = mlab
    monkeypatch.setitem(sys.modules, "mayavi", mayavi)
    monkeypatch.setitem(sys.modules, "mayavi.mlab", mlab)

    nodes, elements = fem.generate_tetrahedral_mesh(1.0, 1.0, 1.0, (1, 1, 1))
    u = np.zeros(3 * len(nodes))
    u[2::3] = -0.1
    fem.plot_mesh(nodes, elements, u, scale=2.0)
    assert calls["shown"]
    assert len(calls["triangles"]) == 12
    assert np.allclose(calls["z"], np.array(nodes)[:, 2] - 0.2)
    assert np.allclose(calls["kw"]["scalars"], 0.1)
