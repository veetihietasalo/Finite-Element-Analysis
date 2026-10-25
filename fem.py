"""Linear elastic finite element solver with 4-node tetrahedra.

Units are whatever you keep consistent. The materials table stores Young's
modulus in MPa, so lengths in mm and forces in N give displacements in mm and
stresses in MPa.

Strain and stress vectors use the order xx, yy, zz, xy, yz, xz, with
engineering shear strains (gamma_xy = du/dy + dv/dx).

Run `python fem.py` for the demo, `python fem.py --plot` to also
draw the deformed mesh with mayavi.
"""

import argparse
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

AXES = {'x': 0, 'y': 1, 'z': 2}


def create_materials_database(db_name: str):
    """Create the `materials` table in `db_name` and add Steel, Aluminum and Concrete."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS materials
                 (name TEXT PRIMARY KEY,
                  youngs_modulus REAL,
                  poisson_ratio REAL)''')
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Steel', 210000, 0.3)")
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Aluminum', 69000, 0.33)")
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Concrete', 30000, 0.2)")
    conn.commit()
    conn.close()


def assign_material_properties(elements: List[Tuple[int, int, int, int]], db_name: str,
                               material_name: str) -> List[Dict[str, float]]:
    """Look up `material_name` and return one {'youngs_modulus', 'poisson_ratio'} dict per element."""
    conn = sqlite3.connect(db_name)
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM materials WHERE name=?", (material_name,))
        material = c.fetchone()
    finally:
        conn.close()
    if material is None:
        raise ValueError(f"Material {material_name} not found in the database.")
    youngs_modulus, poisson_ratio = material[1], material[2]
    return [{'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio} for _ in elements]


def generate_tetrahedral_mesh(length: float, width: float, height: float, divisions: Tuple[int, int, int]
                              ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int, int]]]:
    """Mesh the box [0, length] x [0, width] x [0, height] into tetrahedra.

    Nodes sit on a regular grid with `divisions` = (nx, ny, nz) cells. Each cell is
    split into 6 tetrahedra around its main diagonal n0-n7, one for every path from
    n0 to n7 along the cell edges. Every cell uses the same split, so the face
    diagonals of neighbouring cells match and the mesh is conforming. All elements
    have positive volume with the node order used here.

    Returns (nodes, elements): nodes as (x, y, z) tuples, elements as 4-tuples of node indices.
    """
    nx, ny, nz = divisions
    dx, dy, dz = length / nx, width / ny, height / nz

    nodes = []
    node_map = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                node_map[(i, j, k)] = len(nodes)
                nodes.append((i * dx, j * dy, k * dz))

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i + 1, j, k)]
                n2 = node_map[(i, j + 1, k)]
                n3 = node_map[(i + 1, j + 1, k)]
                n4 = node_map[(i, j, k + 1)]
                n5 = node_map[(i + 1, j, k + 1)]
                n6 = node_map[(i, j + 1, k + 1)]
                n7 = node_map[(i + 1, j + 1, k + 1)]
                elements.append((n0, n1, n3, n7))  # x, y, z
                elements.append((n0, n3, n2, n7))  # y, x, z
                elements.append((n0, n2, n6, n7))  # y, z, x
                elements.append((n0, n6, n4, n7))  # z, y, x
                elements.append((n0, n4, n5, n7))  # z, x, y
                elements.append((n0, n5, n1, n7))  # x, z, y

    return nodes, elements


def tetrahedron_volume(element_coordinates: np.ndarray) -> float:
    """Signed volume of a tetrahedron from its 4 x 3 node coordinates (positive for the mesh's node order)."""
    X = np.asarray(element_coordinates, dtype=float)
    return float(np.linalg.det(X[1:] - X[0])) / 6.0


def boundary_faces(elements: Iterable[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
    """Triangles that belong to exactly one element, i.e. the outer surface of the mesh."""
    count: Dict[Tuple[int, ...], int] = {}
    for a, b, c, d in elements:
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(face))
            count[key] = count.get(key, 0) + 1
    return [face for face, n in count.items() if n == 1]


def compute_B_matrix(element_coordinates: np.ndarray) -> np.ndarray:
    """6 x 12 strain-displacement matrix of a linear tetrahedron.

    The shape functions are N_i = a_i + b_i x + c_i y + d_i z with N_i(node j) = 1 if
    i == j else 0. Their coefficients are the columns of inv(A), where row j of A is
    [1, x_j, y_j, z_j], so the derivatives (b_i, c_i, d_i) are rows 1..3 of inv(A).
    B @ u_e gives [e_xx, e_yy, e_zz, g_xy, g_yz, g_xz] for the element's 12 nodal
    displacements u_e = [u1, v1, w1, ..., u4, v4, w4].
    """
    X = np.asarray(element_coordinates, dtype=float)
    scale = np.ptp(X, axis=0).max()
    if abs(tetrahedron_volume(X)) <= 1e-12 * scale ** 3:
        raise ValueError("The tetrahedron has zero volume, check the node positions.")

    A = np.hstack([np.ones((4, 1)), X])
    b, c, d = np.linalg.inv(A)[1:]

    B = np.zeros((6, 12))
    for i in range(4):
        B[0, 3 * i] = b[i]
        B[1, 3 * i + 1] = c[i]
        B[2, 3 * i + 2] = d[i]
        B[3, 3 * i], B[3, 3 * i + 1] = c[i], b[i]
        B[4, 3 * i + 1], B[4, 3 * i + 2] = d[i], c[i]
        B[5, 3 * i], B[5, 3 * i + 2] = d[i], b[i]
    return B


def elasticity_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """6 x 6 isotropic linear elastic matrix D (stress = D @ strain).

    Diagonal lambda + 2 mu (= K + 4G/3), off-diagonal lambda (= K - 2G/3), shear mu (= G).
    """
    E, nu = youngs_modulus, poisson_ratio
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def compute_element_stiffness_matrix(element_coordinates: np.ndarray, youngs_modulus: float,
                                     poisson_ratio: float) -> np.ndarray:
    """12 x 12 stiffness matrix K_e = V * B^T D B of a linear tetrahedron."""
    B = compute_B_matrix(element_coordinates)
    D = elasticity_matrix(youngs_modulus, poisson_ratio)
    V = abs(tetrahedron_volume(element_coordinates))
    return V * B.T @ D @ B


def element_dofs(element: Iterable[int]) -> np.ndarray:
    """Global degree-of-freedom indices [3n, 3n+1, 3n+2] of the element's nodes, in node order."""
    n = np.asarray(element, dtype=int)
    return (3 * n[:, None] + np.arange(3)).ravel()


def assemble_global_stiffness_matrix(elements, nodes, youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Dense 3N x 3N global stiffness matrix, 3 degrees of freedom per node, one material."""
    nodes = np.asarray(nodes, dtype=float)
    K = np.zeros((3 * len(nodes), 3 * len(nodes)))
    for element in elements:
        element = list(element)
        K_e = compute_element_stiffness_matrix(nodes[element], youngs_modulus, poisson_ratio)
        dofs = element_dofs(element)
        K[np.ix_(dofs, dofs)] += K_e
    return K


def apply_boundary_conditions(nodes: List[Tuple[float, float, float]],
                              constraints: Dict[str, Union[float, str]],
                              node_ids: Optional[Iterable[int]] = None) -> List[Dict[str, Union[float, str]]]:
    """One constraint dict per node for `solve_for_displacements`.

    `constraints` maps 'x', 'y', 'z' to 'free' or a prescribed displacement. It is applied
    to the nodes in `node_ids` (all nodes if None); the other nodes are left free.
    """
    for axis in constraints:
        if axis not in AXES:
            raise ValueError(f"Unknown axis {axis!r}, use 'x', 'y' or 'z'.")
    selected = set(range(len(nodes))) if node_ids is None else set(int(i) for i in node_ids)
    free = {'x': 'free', 'y': 'free', 'z': 'free'}
    return [dict(free, **constraints) if i in selected else dict(free) for i in range(len(nodes))]


def apply_forces(nodes: List[Tuple[float, float, float]], forces: Dict[str, float],
                 node_ids: Optional[Iterable[int]] = None) -> List[Dict[str, float]]:
    """One {'Fx', 'Fy', 'Fz'} dict per node: `forces` on the nodes in `node_ids` (all if None), zero elsewhere."""
    selected = set(range(len(nodes))) if node_ids is None else set(int(i) for i in node_ids)
    loaded = {key: float(forces.get(key, 0.0)) for key in ('Fx', 'Fy', 'Fz')}
    zero = {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0}
    return [dict(loaded) if i in selected else dict(zero) for i in range(len(nodes))]


def build_force_vector(node_forces: List[Dict[str, float]]) -> np.ndarray:
    """Global force vector F (length 3N) from per-node force dicts."""
    F = np.zeros(3 * len(node_forces))
    for i, force in enumerate(node_forces):
        F[3 * i:3 * i + 3] = [force.get('Fx', 0.0), force.get('Fy', 0.0), force.get('Fz', 0.0)]
    return F


def traction_force_vector(nodes, elements, node_ids: Iterable[int], traction) -> np.ndarray:
    """Consistent nodal forces (length 3N) for a uniform traction on part of the surface.

    The loaded surface is every boundary triangle whose 3 nodes are all in `node_ids`.
    Each triangle passes traction * area / 3 to each of its nodes (the consistent load
    for linear elements). `traction` is force per area as (tx, ty, tz).
    """
    nodes = np.asarray(nodes, dtype=float)
    selected = set(int(i) for i in node_ids)
    t = np.asarray(traction, dtype=float)
    F = np.zeros(3 * len(nodes))
    for face in boundary_faces(elements):
        if all(n in selected for n in face):
            p0, p1, p2 = nodes[list(face)]
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            for n in face:
                F[3 * n:3 * n + 3] += t * area / 3.0
    return F


def solve_for_displacements(K: np.ndarray, F: np.ndarray,
                            constraints: List[Dict[str, Union[float, str]]]) -> np.ndarray:
    """Solve K u = F with prescribed displacements.

    `constraints` has one dict per node mapping 'x', 'y', 'z' to 'free' or a prescribed
    value. With c the constrained and f the free degrees of freedom:
        u_c = prescribed values
        K_ff u_f = F_f - K_fc u_c
    K and F are not changed. Returns the full displacement vector (length 3N).
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    n = K.shape[0]
    prescribed = {}
    for i, constraint in enumerate(constraints):
        for axis, value in constraint.items():
            if isinstance(value, str):
                if value != 'free':
                    raise ValueError(f"Constraint value must be 'free' or a number, got {value!r}.")
                continue
            if axis not in AXES:
                raise ValueError(f"Unknown axis {axis!r}, use 'x', 'y' or 'z'.")
            prescribed[3 * i + AXES[axis]] = float(value)

    c = np.array(sorted(prescribed), dtype=int)
    f = np.setdiff1d(np.arange(n), c)
    u = np.zeros(n)
    u[c] = [prescribed[i] for i in c]
    rhs = F[f] - K[np.ix_(f, c)] @ u[c]
    u[f] = np.linalg.solve(K[np.ix_(f, f)], rhs)
    return u


def calculate_stress(strain, E, nu):
    """Stress vector D @ strain for an isotropic linear elastic material."""
    return elasticity_matrix(E, nu) @ np.asarray(strain, dtype=float)


def compute_element_stresses(elements, nodes, displacements, youngs_modulus: float,
                             poisson_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Strain B u_e and stress D (B u_e) in every element (constant per element for linear tets).

    Returns (strains, stresses), both of shape (number of elements, 6).
    """
    nodes = np.asarray(nodes, dtype=float)
    u = np.asarray(displacements, dtype=float)
    strains = np.zeros((len(elements), 6))
    for e, element in enumerate(elements):
        element = list(element)
        strains[e] = compute_B_matrix(nodes[element]) @ u[element_dofs(element)]
    stresses = strains @ elasticity_matrix(youngs_modulus, poisson_ratio).T
    return strains, stresses


def von_mises(stress) -> np.ndarray:
    """Von Mises stress of one stress vector or of rows of stress vectors."""
    s = np.asarray(stress, dtype=float)
    sx, sy, sz, txy, tyz, txz = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + txz ** 2))


def plot_mesh(nodes, elements, displacements=None, scale: float = 1.0):
    """Draw the outer surface of the (deformed) mesh as a mayavi wireframe, coloured by |u|.

    mayavi is imported here, so the rest of the module works without it.
    """
    from mayavi import mlab

    nodes = np.asarray(nodes, dtype=float)
    u = np.zeros_like(nodes) if displacements is None else np.asarray(displacements, dtype=float).reshape(-1, 3)
    deformed = nodes + scale * u
    mlab.triangular_mesh(deformed[:, 0], deformed[:, 1], deformed[:, 2], boundary_faces(elements),
                         scalars=np.linalg.norm(u, axis=1), representation="wireframe", line_width=1.0)
    mlab.show()


def main():
    parser = argparse.ArgumentParser(description="Steel box 10 x 1 x 1, node 0 fixed, no load.")
    parser.add_argument("--plot", action="store_true", help="draw the deformed mesh with mayavi")
    args = parser.parse_args()

    db_name = 'materials.db'
    create_materials_database(db_name)
    length, width, height = 10.0, 1.0, 1.0
    divisions = (2, 1, 1)
    nodes, elements = generate_tetrahedral_mesh(length, width, height, divisions)
    element_properties = assign_material_properties(elements, db_name, 'Steel')
    youngs_modulus = element_properties[0]['youngs_modulus']
    poisson_ratio = element_properties[0]['poisson_ratio']

    # Constrain the first node in all directions, keep the remaining nodes free to move
    boundary_conditions = [{'x': 0, 'y': 0, 'z': 0}]
    boundary_conditions += [{'x': 'free', 'y': 'free', 'z': 'free'} for _ in range(len(nodes) - 1)]

    # Applied forces for each node (in Newton), 0 means no force in that direction
    applied_forces = [{'Fx': 0, 'Fy': 0, 'Fz': 0} for _ in range(len(nodes))]

    K = assemble_global_stiffness_matrix(np.array(elements), np.array(nodes), youngs_modulus, poisson_ratio)
    F = np.zeros(3 * len(nodes))
    for i, force in enumerate(applied_forces):
        F[3 * i] = force['Fx']
        F[3 * i + 1] = force['Fy']
        F[3 * i + 2] = force['Fz']

    displacements = solve_for_displacements(K, F, boundary_conditions)

    print(nodes)
    print("Global Stiffness Matrix:")
    print(K)
    print("Node Displacements:")
    print(displacements)

    if args.plot:
        plot_mesh(nodes, elements, displacements)


if __name__ == "__main__":
    main()
