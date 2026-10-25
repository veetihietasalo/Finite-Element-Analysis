"""Assembled-model checks: rigid body modes, prescribed displacements, patch test, bar, cantilever."""
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


@pytest.mark.parametrize("amount", [0.0, 0.15])
def test_patch_test_linear_field_is_exact(amount):
    """Prescribe u = a + G x on the surface nodes; interior nodes, strain and stress must be exact."""
    P, elements, on_surface = perturbed_box_mesh(2.0, 1.5, 1.0, (4, 3, 3), amount, seed=5)
    rng = np.random.default_rng(6)
    G = rng.normal(size=(3, 3)) * 1e-3
    a = rng.normal(size=3) * 1e-3
    exact = a + P @ G.T

    cons = free_constraints(len(P))
    for i in np.flatnonzero(on_surface):
        cons[i] = {'x': exact[i, 0], 'y': exact[i, 1], 'z': exact[i, 2]}
    K = fem.assemble_global_stiffness_matrix(elements, P, E, NU)
    u = fem.solve_for_displacements(K, np.zeros(3 * len(P)), cons)

    assert np.sum(~on_surface) > 0
    assert np.allclose(u.reshape(-1, 3), exact, rtol=0, atol=1e-14)

    strains, stresses = fem.compute_element_stresses(elements, P, u, E, NU)
    eps = np.array([G[0, 0], G[1, 1], G[2, 2], G[0, 1] + G[1, 0], G[1, 2] + G[2, 1], G[0, 2] + G[2, 0]])
    sigma = fem.elasticity_matrix(E, NU) @ eps
    assert np.allclose(strains, eps, rtol=0, atol=1e-12)
    assert np.allclose(stresses, sigma, rtol=0, atol=1e-6)


def bar_constraints(P, length):
    """Symmetry planes x = 0, y = 0, z = 0: the bar can stretch and contract freely."""
    cons = free_constraints(len(P))
    for axis, name in enumerate('xyz'):
        for i in np.flatnonzero(np.isclose(P[:, axis], 0)):
            cons[i][name] = 0.0
    return cons


def test_bar_under_end_load_matches_FL_over_EA():
    length, width, height, load = 10.0, 1.0, 1.5, 1000.0
    area = width * height
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, (10, 2, 3))
    P = np.array(nodes)
    tip = np.flatnonzero(np.isclose(P[:, 0], length))
    cons = bar_constraints(P, length)
    F = fem.traction_force_vector(nodes, elements, tip, (load / area, 0.0, 0.0))
    assert np.isclose(F[0::3].sum(), load) and np.allclose(F[1::3], 0) and np.allclose(F[2::3], 0)

    K = fem.assemble_global_stiffness_matrix(elements, nodes, E, NU)
    u = fem.solve_for_displacements(K, F, cons).reshape(-1, 3)

    sigma = load / area
    exact = np.column_stack([sigma / E * P[:, 0], -NU * sigma / E * P[:, 1], -NU * sigma / E * P[:, 2]])
    assert np.allclose(u, exact, rtol=0, atol=1e-12 * np.abs(exact).max())
    assert np.allclose(u[tip, 0], load * length / (E * area), rtol=1e-10)

    _, stresses = fem.compute_element_stresses(elements, nodes, u.ravel(), E, NU)
    assert np.allclose(stresses[:, 0], sigma, rtol=1e-10)
    assert np.allclose(stresses[:, 1:], 0, atol=1e-10 * sigma)

    reactions = K @ u.ravel() - F
    root = np.flatnonzero(np.isclose(P[:, 0], 0))
    assert np.isclose(reactions[3 * root].sum(), -load, rtol=1e-10)


def test_bar_with_prescribed_end_displacement():
    length, width, height, delta = 10.0, 1.0, 1.5, 0.02
    area = width * height
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, (10, 2, 3))
    P = np.array(nodes)
    tip = np.flatnonzero(np.isclose(P[:, 0], length))
    cons = bar_constraints(P, length)
    for i in tip:
        cons[i]['x'] = delta

    K = fem.assemble_global_stiffness_matrix(elements, nodes, E, NU)
    u = fem.solve_for_displacements(K, np.zeros(3 * len(P)), cons).reshape(-1, 3)
    strain = delta / length
    assert np.allclose(u[:, 0], strain * P[:, 0], rtol=0, atol=1e-14)
    assert np.allclose(u[:, 1], -NU * strain * P[:, 1], rtol=0, atol=1e-14)

    reactions = K @ u.ravel()
    assert np.isclose(reactions[3 * tip].sum(), E * area * strain, rtol=1e-10)
    _, stresses = fem.compute_element_stresses(elements, nodes, u.ravel(), E, NU)
    assert np.allclose(stresses[:, 0], E * strain, rtol=1e-10)


CANTILEVER_MESHES = [(10, 1, 1), (20, 2, 2), (40, 4, 4)]


def cantilever(divisions, length=10.0, width=1.0, height=1.0, load=1.0):
    """Clamped at x = 0, tip load down as a uniform shear traction on the end face."""
    nodes, elements = fem.generate_tetrahedral_mesh(length, width, height, divisions)
    P = np.array(nodes)
    clamped = np.flatnonzero(np.isclose(P[:, 0], 0))
    tip = np.flatnonzero(np.isclose(P[:, 0], length))
    cons = fem.apply_boundary_conditions(nodes, {'x': 0.0, 'y': 0.0, 'z': 0.0}, clamped)
    F = fem.traction_force_vector(nodes, elements, tip, (0.0, 0.0, -load / (width * height)))
    K = fem.assemble_global_stiffness_matrix(elements, nodes, E, NU)
    u = fem.solve_for_displacements(K, F, cons)
    deflection = (F @ u) / load
    inertia = width * height ** 3 / 12
    euler_bernoulli = load * length ** 3 / (3 * E * inertia)
    timoshenko = euler_bernoulli + load * length / ((5 / 6) * E / (2 * (1 + NU)) * width * height)
    reactions = K @ u - F
    tip_mean = u.reshape(-1, 3)[tip].mean(axis=0)
    return deflection, timoshenko, reactions[3 * clamped + 2].sum(), tip_mean


@pytest.fixture(scope="module")
def cantilever_results():
    return {div: cantilever(div) for div in CANTILEVER_MESHES}


def test_cantilever_equilibrium(cantilever_results):
    for _, _, reaction_z, _ in cantilever_results.values():
        assert np.isclose(reaction_z, 1.0, rtol=1e-8)


def test_cantilever_tip_deflection_converges_from_below(cantilever_results):
    """Linear tets are too stiff in bending, so the tolerance is wide on purpose.

    FE / Timoshenko measured 0.219, 0.504, 0.787 at 10x1x1, 20x2x2, 40x4x4 cells
    (0.887 at 60x6x6, left out here since the dense K takes about 2 GB). The test asks
    for a deflection that grows with the mesh, stays below theory and is at least 0.75
    of it at 40x4x4.
    """
    ratios = [d / t for d, t, _, _ in (cantilever_results[div] for div in CANTILEVER_MESHES)]
    assert all(r1 < r2 for r1, r2 in zip(ratios, ratios[1:]))
    assert ratios[-1] < 1.0
    assert ratios[-1] > 0.75


def test_cantilever_sideways_displacement_shrinks_with_mesh(cantilever_results):
    """The same split in every cell isn't mirror symmetric, so a z load also moves the tip in y.

    Measured u_y / u_z at the tip: -0.27, -0.17, -0.07 at 10x1x1, 20x2x2, 40x4x4
    (-0.035 at 60x6x6, -0.010 at 120x12x12 with a sparse solve outside the tests).
    It is mesh bias, so it has to shrink as the mesh gets finer.
    """
    side = [abs(u[1] / u[2]) for _, _, _, u in (cantilever_results[div] for div in CANTILEVER_MESHES)]
    assert all(s1 > s2 for s1, s2 in zip(side, side[1:]))
    assert side[-1] < 0.1
