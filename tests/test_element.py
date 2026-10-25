"""Single-element checks: B matrix, D matrix, element stiffness."""
import numpy as np
import pytest

import fem

E, NU = 210000.0, 0.3
UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def random_tet(seed):
    rng = np.random.default_rng(seed)
    X = UNIT_TET + rng.uniform(-0.2, 0.2, (4, 3))
    assert fem.tetrahedron_volume(X) > 0.05
    return X


def mesh_cell_tets():
    """The tetrahedra of one 2.5 x 1 x 0.5 cell, as the mesh builds them."""
    nodes, elements = fem.generate_tetrahedral_mesh(2.5, 1.0, 0.5, (1, 1, 1))
    P = np.array(nodes)
    return [P[list(e)] for e in elements]


ELEMENTS = [UNIT_TET] + [random_tet(s) for s in range(5)] + mesh_cell_tets()


def rigid_body_modes(points):
    """12 x 6 (or 3N x 6): translations along x, y, z and infinitesimal rotations u = w x r about x, y, z."""
    P = np.asarray(points, dtype=float)
    modes = []
    for a in range(3):
        t = np.zeros_like(P)
        t[:, a] = 1.0
        modes.append(t.ravel())
    for a in range(3):
        w = np.zeros(3)
        w[a] = 1.0
        modes.append(np.cross(w, P).ravel())
    return np.array(modes).T


def engineering_strain(G):
    """[e_xx, e_yy, e_zz, g_xy, g_yz, g_xz] of the displacement gradient G[i, j] = du_i/dx_j."""
    return np.array([G[0, 0], G[1, 1], G[2, 2], G[0, 1] + G[1, 0], G[1, 2] + G[2, 1], G[0, 2] + G[2, 0]])


@pytest.mark.parametrize("X", ELEMENTS)
def test_element_rigid_body_modes_give_zero_forces(X):
    K_e = fem.compute_element_stiffness_matrix(X, E, NU)
    R = rigid_body_modes(X)
    forces = K_e @ R
    assert np.abs(forces).max() <= 1e-12 * np.abs(K_e).max() * np.abs(R).max()


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
@pytest.mark.parametrize("X", ELEMENTS)
def test_element_stiffness_symmetric_psd_with_six_zero_eigenvalues(X, nu):
    K_e = fem.compute_element_stiffness_matrix(X, E, nu)
    assert K_e.shape == (12, 12)
    assert np.allclose(K_e, K_e.T, rtol=0, atol=1e-12 * np.abs(K_e).max())
    w = np.linalg.eigvalsh(K_e)
    tol = 1e-10 * w.max()
    assert np.all(w > -tol)
    assert np.sum(np.abs(w) < tol) == 6
    # the 6 deformation modes are clearly positive, not round-off
    assert w[6] > 1e-3 * w.max()


@pytest.mark.parametrize("X", ELEMENTS)
def test_B_matrix_gives_exact_strain_of_linear_field(X):
    rng = np.random.default_rng(1)
    G = rng.normal(size=(3, 3))
    a = rng.normal(size=3)
    u = (a + X @ G.T).ravel()
    assert np.allclose(fem.compute_B_matrix(X) @ u, engineering_strain(G), rtol=0, atol=1e-12)


def test_B_matrix_does_not_depend_on_node_order_orientation():
    flipped = UNIT_TET[[1, 0, 2, 3]]
    assert fem.tetrahedron_volume(flipped) < 0
    B = fem.compute_B_matrix(UNIT_TET)
    B_flipped = fem.compute_B_matrix(flipped)
    # same element, nodes 0 and 1 swapped: columns of nodes 0 and 1 swap
    assert np.allclose(B_flipped[:, [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]], B)
    K = fem.compute_element_stiffness_matrix(UNIT_TET, E, NU)
    K_flipped = fem.compute_element_stiffness_matrix(flipped, E, NU)
    p = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]
    assert np.allclose(K_flipped[np.ix_(p, p)], K)


def test_B_matrix_rejects_flat_tetrahedron():
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    with pytest.raises(ValueError):
        fem.compute_B_matrix(flat)


def test_elasticity_matrix_isotropic():
    D = fem.elasticity_matrix(E, NU)
    G = E / (2 * (1 + NU))
    K_bulk = E / (3 * (1 - 2 * NU))
    assert np.allclose(np.diag(D)[:3], K_bulk + 4 * G / 3)
    assert np.isclose(D[0, 1], K_bulk - 2 * G / 3)
    assert np.allclose(np.diag(D)[3:], G)
    # uniaxial stress: strain (e, -nu e, -nu e) gives stress (E e, 0, 0)
    e = 1e-3
    assert np.allclose(D @ [e, -NU * e, -NU * e, 0, 0, 0], [E * e, 0, 0, 0, 0, 0], rtol=0, atol=1e-9)
    # calculate_stress is the same matrix
    strain = np.random.default_rng(2).normal(size=6)
    assert np.allclose(fem.calculate_stress(strain, E, NU), D @ strain)


@pytest.mark.parametrize("X", ELEMENTS)
def test_element_strain_energy_matches_continuum(X):
    """For a linear field, 1/2 u^T K_e u equals 1/2 V eps^T D eps (checks the V scaling, not 6V)."""
    rng = np.random.default_rng(3)
    G = rng.normal(size=(3, 3)) * 1e-3
    u = (X @ G.T).ravel()
    eps = engineering_strain(G)
    V = abs(fem.tetrahedron_volume(X))
    K_e = fem.compute_element_stiffness_matrix(X, E, NU)
    assert np.isclose(0.5 * u @ K_e @ u, 0.5 * V * eps @ fem.elasticity_matrix(E, NU) @ eps, rtol=1e-12)

