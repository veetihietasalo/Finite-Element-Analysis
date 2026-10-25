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


def generate_tetrahedral_mesh(
    length: float, 
    width: float, 
    height: float, 
    divisions: Tuple[int, int, int]
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int, int]]]:
    """
    Generate a 3D tetrahedral mesh based on the given parameters.

    Parameters:
    - length: Length of the cuboid in which the tetrahedra fit.
    - width: Width of the cuboid in which the tetrahedra fit.
    - height: Height of the cuboid in which the tetrahedra fit.
    - divisions: Number of divisions along each axis (x, y, z).

    Returns:
    - nodes: List of nodes (x, y, z).
    - elements: List of tetrahedral elements. Each element is defined by 4 node indices.
    """
    
    # Initialize lists to store nodes and elements
    nodes = []
    elements = []
    
    # Calculate the step size for each division
    dx = length / divisions[0]
    dy = width / divisions[1]
    dz = height / divisions[2]
    
    # Generate nodes
    node_id = 0
    node_map = {}  # Mapping from (i, j, k) to node_id
    for i in range(divisions[0] + 1):
        for j in range(divisions[1] + 1):
            for k in range(divisions[2] + 1):
                x = i * dx
                y = j * dy
                z = k * dz
                nodes.append((x, y, z))
                node_map[(i, j, k)] = node_id
                node_id += 1
    
    # Generate tetrahedral elements
    for i in range(divisions[0]):
        for j in range(divisions[1]):
            for k in range(divisions[2]):
                # Node indices for the corner points of the current cube
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i + 1, j, k)]
                n2 = node_map[(i, j + 1, k)]
                n3 = node_map[(i + 1, j + 1, k)]
                n4 = node_map[(i, j, k + 1)]
                n5 = node_map[(i + 1, j, k + 1)]
                n6 = node_map[(i, j + 1, k + 1)]
                n7 = node_map[(i + 1, j + 1, k + 1)]
                
                # Generate 5 tetrahedra that fill the cube
                elements.append((n0, n1, n3, n7))
                elements.append((n0, n3, n2, n7))
                elements.append((n0, n2, n6, n7))
                elements.append((n0, n6, n4, n7))
                elements.append((n0, n4, n5, n7))
                
    return nodes, elements


def extract_triangles_from_tetrahedra(elements: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
    triangles = []
    for tet in elements:
        triangles.append((tet[0], tet[1], tet[2]))
        triangles.append((tet[0], tet[1], tet[3]))
        triangles.append((tet[0], tet[2], tet[3]))
        triangles.append((tet[1], tet[2], tet[3]))
    return triangles


def extract_unique_triangles_from_tetrahedra(elements: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
    triangle_set = set()
    for tet in elements:
        triangle_set.add(tuple(sorted([tet[0], tet[1], tet[2]])))
        triangle_set.add(tuple(sorted([tet[0], tet[1], tet[3]])))
        triangle_set.add(tuple(sorted([tet[0], tet[2], tet[3]])))
        triangle_set.add(tuple(sorted([tet[1], tet[2], tet[3]])))
    return list(triangle_set)


def compute_B_matrix(tetrahedron_nodes):
    # Node coordinates
    x1, y1, z1 = tetrahedron_nodes[0]
    x2, y2, z2 = tetrahedron_nodes[1]
    x3, y3, z3 = tetrahedron_nodes[2]
    x4, y4, z4 = tetrahedron_nodes[3]

    # Volume of the tetrahedron
    V = (1/6) * np.linalg.det([
        [1, x1, y1, z1],
        [1, x2, y2, z2],
        [1, x3, y3, z3],
        [1, x4, y4, z4]
    ])

    # Check for non-zero volume
    if V == 0:
        raise ValueError("The tetrahedron has zero volume, check the node positions.")

    # Compute derivatives of shape functions
    b = [(y2*z3 - y3*z2 - y2*z4 + y4*z2 + y3*z4 - y4*z3) / (6*V),
         (y3*z1 - y1*z3 - y3*z4 + y4*z3 + y1*z4 - y4*z1) / (6*V),
         (y1*z2 - y2*z1 - y1*z4 + y4*z1 + y2*z4 - y4*z2) / (6*V),
         (y2*z1 - y1*z2 - y2*z3 + y3*z2 + y1*z3 - y3*z1) / (6*V)]

    c = [(x3*z2 - x2*z3 - x3*z4 + x4*z3 + x2*z4 - x4*z2) / (6*V),
         (x1*z3 - x3*z1 - x1*z4 + x4*z1 + x3*z4 - x4*z3) / (6*V),
         (x2*z1 - x1*z2 - x2*z4 + x4*z2 + x1*z4 - x4*z1) / (6*V),
         (x2*z3 - x3*z2 - x2*z1 + x1*z2 + x3*z1 - x1*z3) / (6*V)]

    d = [(x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3) / (6*V),
         (x3*y1 - x1*y3 - x3*y4 + x4*y3 + x1*y4 - x4*y1) / (6*V),
         (x1*y2 - x2*y1 - x1*y4 + x4*y1 + x2*y4 - x4*y2) / (6*V),
         (x3*y2 - x2*y3 - x3*y1 + x1*y3 + x2*y1 - x1*y2) / (6*V)]

    # Assemble the B matrix
    B = np.zeros((6, 12))
    for i in range(4):
        B[0, i*3] = b[i]
        B[1, i*3+1] = c[i]
        B[2, i*3+2] = d[i]
        B[3, i*3] = c[i]
        B[3, i*3+1] = b[i]
        B[4, i*3+1] = d[i]
        B[4, i*3+2] = c[i]
        B[5, i*3] = d[i]
        B[5, i*3+2] = b[i]

    return B


def compute_element_stiffness_matrix(
    element_coordinates: np.ndarray, 
    youngs_modulus: float, 
    poisson_ratio: float
) -> np.ndarray:
    """
    Compute the stiffness matrix for a tetrahedral element.

    Parameters:
    - element_coordinates: 4x3 array containing the x, y, z coordinates of the element's nodes.
    - youngs_modulus: Young's modulus of the material.
    - poisson_ratio: Poisson's ratio of the material.

    Returns:
    - 12x12 stiffness matrix for the tetrahedral element.
    """
    
    # Compute the element volume (V)
    v1 = element_coordinates[1, :] - element_coordinates[0, :]
    v2 = element_coordinates[2, :] - element_coordinates[0, :]
    v3 = element_coordinates[3, :] - element_coordinates[0, :]
    V = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    # Construct the B matrix (strain-displacement matrix)
    b = np.zeros((6, 12))
    for i in range(4):
        xi, yi, zi = element_coordinates[i, :]
        b[0, i * 3] = xi
        b[1, i * 3 + 1] = yi
        b[2, i * 3 + 2] = zi
        b[3, i * 3:i * 3 + 3] = [yi, xi, 0]
        b[4, i * 3:i * 3 + 3] = [0, zi, yi]
        b[5, i * 3:i * 3 + 3] = [zi, 0, xi]
    b /= (6.0 * V)

    # Construct the D matrix (material matrix)
    G = youngs_modulus / (2 * (1 + poisson_ratio))
    K = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    D = np.array([
        [K, K - 2 * G / 3, K - 2 * G / 3, 0, 0, 0],
        [K - 2 * G / 3, K, K - 2 * G / 3, 0, 0, 0],
        [K - 2 * G / 3, K - 2 * G / 3, K, 0, 0, 0],
        [0, 0, 0, G, 0, 0],
        [0, 0, 0, 0, G, 0],
        [0, 0, 0, 0, 0, G]
    ])

    # Compute the element stiffness matrix (K_e)
    K_e = np.dot(b.T, np.dot(D, b)) * V * 6

    return K_e


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
    """
    Calculate the stress for a given strain using the material properties.
    Args:
    - strain (array): The strain vector.
    - E (float): Young's modulus.
    - nu (float): Poisson's ratio.
    
    Returns:
    - stress (array): The stress vector.
    """
    # Calculate stiffness matrix components
    C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    C12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    C44 = E / (2 * (1 + nu))
    
    # Stiffness matrix
    C = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44]
    ])
    
    # Compute stress
    stress = np.dot(C, strain)
    
    return stress


def plot_mesh(nodes, elements, displacements=None, scale: float = 1.0):
    """Draw the (deformed) mesh as a mayavi wireframe, coloured by |u|.

    mayavi is imported here, so the rest of the module works without it.
    """
    from mayavi import mlab

    nodes = np.asarray(nodes, dtype=float)
    u = np.zeros_like(nodes) if displacements is None else np.asarray(displacements, dtype=float).reshape(-1, 3)
    deformed = nodes + scale * u
    mlab.triangular_mesh(deformed[:, 0], deformed[:, 1], deformed[:, 2], extract_unique_triangles_from_tetrahedra(elements),
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
