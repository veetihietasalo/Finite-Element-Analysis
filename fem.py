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
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np


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


def apply_boundary_conditions(
    nodes: List[Tuple[float, float, float]], 
    constraints: Dict[str, Union[float, str]]
) -> List[Dict[str, Union[float, str]]]:
    """
    Apply boundary conditions to the nodes.

    Parameters:
    - nodes: List of nodes (x, y, z).
    - constraints: Dictionary defining the boundary conditions. 
                   Keys can be 'x', 'y', 'z', and values can be either a fixed value or 'free'.

    Returns:
    - List of dictionaries representing the boundary conditions for each node.
    """
    
    node_constraints = []
    
    for node in nodes:
        x, y, z = node
        constraint = {}
        
        for axis, value in constraints.items():
            if axis == 'x':
                constraint['x'] = value if value == 'free' else x
            elif axis == 'y':
                constraint['y'] = value if value == 'free' else y
            elif axis == 'z':
                constraint['z'] = value if value == 'free' else z
                
        node_constraints.append(constraint)
        
    return node_constraints


def apply_forces(
    nodes: List[Tuple[float, float, float]], 
    forces: Dict[str, float]
) -> List[Dict[str, float]]:
    """
    Apply forces to the nodes.

    Parameters:
    - nodes: List of nodes (x, y, z).
    - forces: Dictionary defining the forces applied. Keys can be 'Fx', 'Fy', 'Fz'.

    Returns:
    - List of dictionaries representing the forces applied to each node.
    """
    
    node_forces = []
    
    for node in nodes:
        force = {}
        
        for axis, value in forces.items():
            if axis in ['Fx', 'Fy', 'Fz']:
                force[axis] = value
                
        node_forces.append(force)
        
    return node_forces


def solve_for_displacements(K: np.ndarray, F: np.ndarray, constraints: List[Dict[str, Union[float, str]]]) -> np.ndarray:
    """
    Solve for node displacements based on the global stiffness matrix, applied forces, and constraints.

    Parameters:
    - K: Global stiffness matrix.
    - F: Force vector.
    - constraints: List of boundary condition dictionaries for each node.

    Returns:
    - Displacement vector.
    """

    # Apply constraints to the stiffness matrix and force vector
    for i, constraint in enumerate(constraints):
        for axis, value in constraint.items():
            if value != 'free':
                index = 3 * i + {'x': 0, 'y': 1, 'z': 2}[axis]
                K[index, :] = 0
                K[:, index] = 0
                K[index, index] = 1
                F[index] = value

    # Solve the system of equations to find the displacements
    # If you can use scipy, uncomment the following line
    # displacements = scipy.linalg.solve(K, F)
    
    # If you can't use scipy, you can use numpy's solver
    displacements = np.linalg.solve(K, F)

    return displacements


def calculate_stress(strain, E, nu):
    """Stress vector D @ strain for an isotropic linear elastic material."""
    return elasticity_matrix(E, nu) @ np.asarray(strain, dtype=float)


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
