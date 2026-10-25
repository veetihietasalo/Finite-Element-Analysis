"""Assembled-model checks: rigid body modes, boundary conditions, prescribed displacements."""
import numpy as np
import pytest

import fem

E, NU = 210000.0, 0.3


def rigid_body_modes(points):
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


def free_constraints(n):
    return [{'x': 'free', 'y': 'free', 'z': 'free'} for _ in range(n)]


def perturbed_box_mesh(length, width, height, divisions, amount, seed):
    """Box mesh with every interior node moved by up to `amount` cell sizes along each axis."""
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, divisions)
    P = np.array(nodes)
    extent = np.array([length, width, height])
    on_surface = np.any(np.isclose(P, 0) | np.isclose(P, extent), axis=1)
    h = extent / np.array(divisions)
    rng = np.random.default_rng(seed)
    P[~on_surface] += rng.uniform(-amount, amount, (np.sum(~on_surface), 3)) * h
    assert all(fem.tetrahedron_volume(P[list(e)]) > 0 for e in elements)
    return P, elements, on_surface


@pytest.mark.parametrize("amount", [0.0, 0.15])
def test_assembled_rigid_body_modes_give_zero_forces(amount):
    P, elements, _ = perturbed_box_mesh(2.0, 1.0, 1.5, (3, 2, 2), amount, seed=4)
    K = fem.assemble_global_stiffness_matrix(elements, P, E, NU)
    R = rigid_body_modes(P)
    assert np.abs(K @ R).max() <= 1e-12 * np.abs(K).max() * np.abs(R).max()
    assert np.allclose(K, K.T, rtol=0, atol=1e-12 * np.abs(K).max())
    w = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(w) < 1e-10 * w.max()) == 6


def test_apply_boundary_conditions_stores_the_given_value():
    nodes = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    bc = fem.apply_boundary_conditions(nodes, {'x': 'free', 'y': 0.25, 'z': 0})
    assert bc == [{'x': 'free', 'y': 0.25, 'z': 0}] * 3
    bc = fem.apply_boundary_conditions(nodes, {'z': -0.5}, node_ids=[1])
    assert bc == [{'x': 'free', 'y': 'free', 'z': 'free'},
                  {'x': 'free', 'y': 'free', 'z': -0.5},
                  {'x': 'free', 'y': 'free', 'z': 'free'}]
    with pytest.raises(ValueError):
        fem.apply_boundary_conditions(nodes, {'w': 0})


def test_apply_forces_and_force_vector():
    nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    forces = fem.apply_forces(nodes, {'Fy': -9.81, 'other': 1.0}, node_ids=[1])
    assert forces == [{'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0}, {'Fx': 0.0, 'Fy': -9.81, 'Fz': 0.0}]
    assert np.array_equal(fem.build_force_vector(forces), [0, 0, 0, 0, -9.81, 0])


@pytest.mark.parametrize("mode", ["translation", "rotation"])
def test_prescribed_face_motion_moves_the_whole_body(mode):
    """No load, one face moved rigidly: the free nodes must follow (the old solver left them at 0)."""
    nodes, elements = fem.generate_tetrahedral_mesh(3.0, 1.0, 1.0, (3, 1, 1))
    P = np.array(nodes)
    if mode == "translation":
        exact = np.tile([0.1, -0.2, 0.05], (len(P), 1))
    else:
        exact = np.cross([1e-3, -2e-3, 0.5e-3], P)
    cons = free_constraints(len(P))
    for i in np.flatnonzero(np.isclose(P[:, 0], 0)):
        cons[i] = {'x': exact[i, 0], 'y': exact[i, 1], 'z': exact[i, 2]}
    K = fem.assemble_global_stiffness_matrix(elements, P, E, NU)
    u = fem.solve_for_displacements(K, np.zeros(3 * len(P)), cons)
    assert np.allclose(u.reshape(-1, 3), exact, rtol=0, atol=1e-12)


def test_solver_does_not_change_inputs_and_rejects_bad_values():
    nodes, elements = fem.generate_tetrahedral_mesh(1.0, 1.0, 1.0, (1, 1, 1))
    K = fem.assemble_global_stiffness_matrix(elements, nodes, E, NU)
    F = np.arange(3 * len(nodes), dtype=float)
    K0, F0 = K.copy(), F.copy()
    cons = fem.apply_boundary_conditions(nodes, {'x': 0.01, 'y': 0, 'z': 0}, node_ids=[0, 1, 2, 4])
    fem.solve_for_displacements(K, F, cons)
    assert np.array_equal(K, K0) and np.array_equal(F, F0)
    cons[0]['x'] = 'fixed'
    with pytest.raises(ValueError):
        fem.solve_for_displacements(K, F, cons)

