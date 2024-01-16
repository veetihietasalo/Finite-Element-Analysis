from typing import List, Tuple, Dict, Union
import sqlite3
import numpy as np
# import scipy.linalg  # Uncomment if you can use scipy
nodes = []
elements = []

def create_materials_database(db_name: str):
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

def generate_tetrahedral_mesh(length: float, width: float, height: float, divisions: Tuple[int, int, int]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int, int]]]:
    nodes = []
    elements = []
    dx = length / divisions[0]
    dy = width / divisions[1]
    dz = height / divisions[2]
    node_id = 0
    node_map = {}
    for i in range(divisions[0] + 1):
        for j in range(divisions[1] + 1):
            for k in range(divisions[2] + 1):
                x = i * dx
                y = j * dy
                z = k * dz
                nodes.append((x, y, z))
                node_map[(i, j, k)] = node_id
                node_id += 1
    for i in range(divisions[0]):
        for j in range(divisions[1]):
            for k in range(divisions[2]):
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i + 1, j, k)]
                n2 = node_map[(i, j + 1, k)]
                n3 = node_map[(i + 1, j + 1, k)]
                n4 = node_map[(i, j, k + 1)]
                n5 = node_map[(i + 1, j, k + 1)]
                n6 = node_map[(i, j + 1, k + 1)]
                n7 = node_map[(i + 1, j + 1, k + 1)]
                elements.append((n0, n1, n3, n7))
                elements.append((n0, n3, n2, n7))
                elements.append((n0, n2, n6, n7))
                elements.append((n0, n6, n4, n7))
                elements.append((n0, n4, n5, n7))
    return nodes, elements

def assign_material_properties(elements: List[Tuple[int, int, int, int]], db_name: str, material_name: str) -> List[Dict[str, float]]:
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT * FROM materials WHERE name=?", (material_name,))
    material = c.fetchone()
    if material is None:
        raise ValueError(f"Material {material_name} not found in the database.")
    youngs_modulus, poisson_ratio = material[1], material[2]
    conn.close()
    element_properties = []
    for _ in elements:
        element_properties.append({
            'youngs_modulus': youngs_modulus,
            'poisson_ratio': poisson_ratio
        })
    return element_properties

def apply_boundary_conditions(nodes: List[Tuple[float, float, float]], constraints: Dict[str, Union[float, str]]) -> List[Dict[str, Union[float, str]]]:
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

def apply_forces(nodes: List[Tuple[float, float, float]], forces: Dict[str, float]) -> List[Dict[str, float]]:
    node_forces = []
    for node in nodes:
        force = {}
        for axis, value in forces.items():
            if axis in ['Fx', 'Fy', 'Fz']:
                force[axis] = value
        node_forces.append(force)
    return node_forces

def compute_element_stiffness_matrix(element_coordinates: np.ndarray, youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    element_coordinates: np.ndarray
    youngs_modulus: float
    poisson_ratio: float
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
    
    return K_e  # Replace with actual computation

def assemble_global_stiffness_matrix(elements: np.ndarray, nodes: np.ndarray, youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    N = len(nodes)
    K = np.zeros((3 * N, 3 * N))
    for element in elements:
        element_coordinates = nodes[element, :]
        K_e = compute_element_stiffness_matrix(element_coordinates, youngs_modulus, poisson_ratio)
        global_dof_indices = np.array([3 * n, 3 * n + 1, 3 * n + 2] for n in element).flatten()
        for i, global_i in enumerate(global_dof_indices):
            for j, global_j in enumerate(global_dof_indices):
                K[global_i, global_j] += K_e[i, j]
    return K

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


