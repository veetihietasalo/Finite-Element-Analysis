from typing import List, Tuple, Dict
import sqlite3
from typing import List, Dict, Union

def create_materials_database(db_name: str):
    """
    Create an SQLite database to store material properties.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table for materials
    c.execute('''CREATE TABLE IF NOT EXISTS materials
                 (name TEXT PRIMARY KEY, 
                  youngs_modulus REAL, 
                  poisson_ratio REAL)''')

    # Insert example materials
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Steel', 210000, 0.3)")
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Aluminum', 69000, 0.33)")
    c.execute("INSERT OR REPLACE INTO materials VALUES ('Concrete', 30000, 0.2)")

    conn.commit()
    conn.close()

# Create the materials database
db_name = 'materials.db'
create_materials_database(db_name)

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

def assign_material_properties(
    elements: List[Tuple[int, int, int, int]], 
    db_name: str, 
    material_name: str
) -> List[Dict[str, float]]:
    """
    Assign material properties to each tetrahedral element from an SQLite database.

    Parameters:
    - elements: List of tetrahedral elements. Each element is defined by 4 node indices.
    - db_name: SQLite database name where material properties are stored.
    - material_name: Name of the material to be assigned to the elements.

    Returns:
    - List of dictionaries, each containing material properties for a tetrahedral element.
    """
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Fetch material properties from the database
    c.execute("SELECT * FROM materials WHERE name=?", (material_name,))
    material = c.fetchone()
    
    if material is None:
        raise ValueError(f"Material {material_name} not found in the database.")
    
    youngs_modulus, poisson_ratio = material[1], material[2]
    
    conn.close()
    
    # Assign material properties to elements
    element_properties = []
    for _ in elements:
        element_properties.append({
            'youngs_modulus': youngs_modulus,
            'poisson_ratio': poisson_ratio
        })
        
    return element_properties

# Import necessary functions here

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

import numpy as np

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

# Example tetrahedral element coordinates
element_coordinates = np.array([
    [0.0, 0.0, 0.0],  # Node 1
    [1.0, 0.0, 0.0],  # Node 2
    [0.0, 1.0, 0.0],  # Node 3
    [0.0, 0.0, 1.0]   # Node 4
])

# Material properties for Steel
youngs_modulus = 210000  # MPa
poisson_ratio = 0.3

# Compute the element stiffness matrix
K_e = compute_element_stiffness_matrix(element_coordinates, youngs_modulus, poisson_ratio)
K_e

print(element_coordinates, youngs_modulus, poisson_ratio)



# Generate the tetrahedral mesh
length = 4.0
width = 2.0
height = 3.0
divisions = (2, 1, 1)  # Divide along each axis
nodes, elements = generate_tetrahedral_mesh(length, width, height, divisions)
# Assign material properties to elements



constraints = {'x': 'free', 'y': 0, 'z': 'free'}
boundary_conditions = apply_boundary_conditions(nodes, constraints)
print("Boundary Conditions for first 5 nodes:", boundary_conditions[:5])

forces = {'Fx': 0, 'Fy': -9.81, 'Fz': 0}
applied_forces = apply_forces(nodes, forces)
print("Forces applied to first 5 nodes:", applied_forces[:5])

# Print the element_properties to the terminal
element_properties = assign_material_properties(elements, db_name, 'Steel')
print("Element Properties:", element_properties)