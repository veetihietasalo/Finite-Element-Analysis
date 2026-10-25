"""Mesh checks: element count, volume, orientation, conformity."""
import numpy as np
import pytest

import fem

BOXES = [
    (1.0, 1.0, 1.0, (1, 1, 1)),
    (10.0, 1.0, 1.0, (2, 1, 1)),
    (4.0, 2.0, 3.0, (3, 2, 4)),
    (2.5, 1.5, 0.7, (5, 3, 2)),
]


@pytest.mark.parametrize("length,width,height,divisions", BOXES)
def test_mesh_counts(length, width, height, divisions):
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, divisions)
    nx, ny, nz = divisions
    assert len(nodes) == (nx + 1) * (ny + 1) * (nz + 1)
    assert len(elements) == 6 * nx * ny * nz
    assert all(len(set(e)) == 4 for e in elements)


@pytest.mark.parametrize("length,width,height,divisions", BOXES)
def test_mesh_volume_equals_box_and_all_volumes_positive(length, width, height, divisions):
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, divisions)
    P = np.array(nodes)
    volumes = np.array([fem.tetrahedron_volume(P[list(e)]) for e in elements])
    cell = length * width * height / np.prod(divisions)
    assert np.all(volumes > 0)
    assert np.allclose(volumes, cell / 6, rtol=1e-12)
    assert np.isclose(volumes.sum(), length * width * height, rtol=1e-12)


@pytest.mark.parametrize("length,width,height,divisions", BOXES)
def test_mesh_is_conforming(length, width, height, divisions):
    """Every face is shared by 2 elements, except faces on the box surface, which have 1."""
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, divisions)
    P = np.array(nodes)
    count = {}
    for a, b, c, d in elements:
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            count[key] = count.get(key, 0) + 1
    assert set(count.values()) <= {1, 2}

    outer = fem.boundary_faces(elements)
    nx, ny, nz = divisions
    assert len(outer) == 4 * (nx * ny + ny * nz + nx * nz)
    extent = np.array([length, width, height])
    area = 0.0
    for face in outer:
        p = P[list(face)]
        on_box_face = [np.allclose(p[:, a], 0) or np.allclose(p[:, a], extent[a]) for a in range(3)]
        assert any(on_box_face)
        area += 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    assert np.isclose(area, 2 * (length * width + width * height + length * height), rtol=1e-12)
