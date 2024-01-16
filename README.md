

Documentation for Finite Element Analysis Python Script
Overview

This Python script is designed for Finite Element Analysis (FEA) in mechanical engineering. It includes functionality for creating a materials database, generating a tetrahedral mesh, assigning material properties, applying boundary conditions, applying forces, and solving for node displacements.
Key Functions

    create_materials_database(db_name: str)
        Purpose: Creates a SQLite database for storing material properties.
        Parameters:
            db_name (str): Name of the database file.
        Functionality:
            Initializes a table for materials with columns for name, Young's modulus, and Poisson's ratio.
            Inserts predefined materials (Steel, Aluminum, Concrete) into the database.

    generate_tetrahedral_mesh(length: float, width: float, height: float, divisions: Tuple[int, int, int])
        Purpose: Generates a 3D tetrahedral mesh.
        Parameters:
            length, width, height (float): Dimensions of the mesh.
            divisions (Tuple[int, int, int]): Number of divisions along each axis (x, y, z).
        Returns: A tuple of nodes (List[Tuple[float, float, float]]) and elements (List[Tuple[int, int, int, int]]).

    assign_material_properties(elements: List[Tuple[int, int, int, int]], db_name: str, material_name: str)
        Purpose: Assigns material properties to elements from the database.
        Parameters:
            elements: Tetrahedral elements.
            db_name, material_name: Database name and material name for property lookup.
        Returns: A list of dictionaries with 'youngs_modulus' and 'poisson_ratio' for each element.

    apply_boundary_conditions(nodes: List[Tuple[float, float, float]], constraints: Dict[str, Union[float, str]])
        Purpose: Applies boundary conditions to nodes.
        Parameters:
            nodes: List of node coordinates.
            constraints: Dictionary of constraints with keys as 'x', 'y', 'z' and values either 'free' or a fixed value.
        Returns: List of node-specific constraint dictionaries.

    apply_forces(nodes: List[Tuple[float, float, float]], forces: Dict[str, float])
        Purpose: Applies forces to nodes.
        Parameters:
            nodes: List of node coordinates.
            forces: Dictionary of forces with keys as 'Fx', 'Fy', 'Fz' and their respective values.
        Returns: List of node-specific force dictionaries.

    compute_element_stiffness_matrix(element_coordinates: np.ndarray, youngs_modulus: float, poisson_ratio: float)
        Purpose: Computes the stiffness matrix for a tetrahedral element.
        Parameters:
            element_coordinates: 4x3 array of the element's node coordinates.
            youngs_modulus, poisson_ratio: Material properties.
        Returns: 12x12 stiffness matrix (np.ndarray).

    assemble_global_stiffness_matrix(elements: np.ndarray, nodes: np.ndarray, youngs_modulus: float, poisson_ratio: float)
        Purpose: Assembles the global stiffness matrix.
        Parameters:
            elements, nodes: Tetrahedral elements and their node coordinates.
            youngs_modulus, poisson_ratio: Material properties.
        Returns: Global stiffness matrix (np.ndarray).

    solve_for_displacements(K: np.ndarray, F: np.ndarray, constraints: List[Dict[str, Union[float, str]]])
        Purpose: Solves for node displacements.
        Parameters:
            K: Global stiffness matrix.
            F: Force vector.
            constraints: List of boundary conditions for each node.
        Returns: Displacement vector (np.ndarray).

Usage

    Setup and Initialization:
        Install necessary Python libraries: numpy, sqlite3, and optionally scipy.
        Define the database name and execute create_materials_database to initialize the materials database.
        Specify the dimensions and divisions for the mesh, and use generate_tetrahedral_mesh to create the mesh.

    Assigning Material Properties:
        Call assign_material_properties with the generated elements, database name, and desired material name (e.g., 'Steel') to assign material properties to each element.

    Applying Boundary Conditions and Forces:
        Define a dictionary for boundary conditions (constraints) and another for forces (forces). The keys should correspond to the axes ('x', 'y', 'z' for constraints and 'Fx', 'Fy', 'Fz' for forces).
        Use apply_boundary_conditions and apply_forces with the list of nodes to generate node-specific constraints and forces.

    Computing Stiffness Matrices and Solving for Displacements:
        Use compute_element_stiffness_matrix for individual elements if needed.
        Assemble the global stiffness matrix using assemble_global_stiffness_matrix.
        Solve for node displacements with solve_for_displacements, passing the global stiffness matrix, force vector, and constraints.

    Visualization and Analysis:
        The displacements can be used for further structural analysis or visualization. This script does not include visualization capabilities, so additional tools or libraries (e.g., matplotlib, mayavi) may be necessary for graphical representation.

Example Usage

db_name = "materials.db"
create_materials_database(db_name)

mesh_dimensions = (10.0, 5.0, 5.0)
divisions = (10, 5, 5)
nodes, elements = generate_tetrahedral_mesh(*mesh_dimensions, divisions)

material_name = "Steel"
element_properties = assign_material_properties(elements, db_name, material_name)

constraints = {'x': 0, 'y': 'free', 'z': 'free'}
node_constraints = apply_boundary_conditions(nodes, constraints)

forces = {'Fx': 100, 'Fy': 0, 'Fz': 0}
node_forces = apply_forces(nodes, forces)

youngs_modulus = element_properties[0]['youngs_modulus']
poisson_ratio = element_properties[0]['poisson_ratio']
K = assemble_global_stiffness_matrix(np.array(elements), np.array(nodes), youngs_modulus, poisson_ratio)

F = np.zeros(len(nodes) * 3)  # Initialize force vector
# Populate F with forces from node_forces

displacements = solve_for_displacements(K, F, node_constraints)


Project Documentation: Finite Element Analysis (FEA) Simulator
Overview

This project provides a Python-based Finite Element Analysis (FEA) simulator for mechanical engineering applications. It focuses on tetrahedral mesh generation, material property assignment, stiffness matrix computation, and structural deformation under applied forces.
Dependencies

    Python 3.x
    numpy: For numerical operations
    sqlite3: For material property database management
    mayavi: For 3D visualization

Key Components

    Material Database Creation (create_materials_database):
        Purpose: To create a SQLite database for storing material properties.
        Parameters:
            db_name (str): Name of the database file.
        Functionality:
            Creates a table for materials with properties: name, Young's modulus, and Poisson's ratio.
            Inserts default materials (Steel, Aluminum) into the database.

    Tetrahedral Mesh Generation (generate_tetrahedral_mesh):
        Purpose: To generate a 3D tetrahedral mesh within a specified cuboid space.
        Parameters:
            length, width, height (float): Dimensions of the cuboid.
            divisions (Tuple[int, int, int]): Number of divisions along each axis (x, y, z).
        Returns:
            A tuple containing a list of nodes (x, y, z) and a list of tetrahedral elements (4 node indices).

    Unique Triangle Extraction (extract_unique_triangles_from_tetrahedra):
        Purpose: To extract unique triangular surfaces from tetrahedral elements.
        Parameters:
            elements (List[Tuple[int, int, int, int]]): List of tetrahedral elements.
        Returns:
            A list of unique triangles (3 node indices).

    Material Property Assignment (assign_material_properties):
        Purpose: To assign material properties to elements from the SQLite database.
        Parameters:
            elements: List of tetrahedral elements.
            db_name, material_name: Database and material name for property lookup.
        Returns:
            A list of dictionaries containing 'youngs_modulus' and 'poisson_ratio' for each element.

    Element Stiffness Matrix Computation (compute_element_stiffness_matrix):
        Purpose: To compute the stiffness matrix for a tetrahedral element.
        Parameters:
            element_coordinates (np.ndarray): Coordinates of the element's nodes.
            youngs_modulus, poisson_ratio (float): Material properties.
        Returns:
            A 12x12 stiffness matrix for the tetrahedral element.

    Global Stiffness Matrix Assembly (assemble_global_stiffness_matrix):
        Purpose: To assemble the global stiffness matrix for the entire structure.
        Parameters:
            elements, nodes: Tetrahedral elements and their node coordinates.
            youngs_modulus, poisson_ratio: Material properties.
        Returns:
            The global stiffness matrix (numpy.ndarray).

    Displacement Solver (solve_for_displacements):
        Purpose: To solve for node displacements based on applied forces and constraints.
        Parameters:
            K (np.ndarray): Global stiffness matrix.
            F (np.ndarray): Force vector.
            constraints (List[Dict[str, Union[float, str]]

]]): Boundary conditions for each node.

    Returns:
        Displacement vector (numpy.ndarray) representing the displacements of each node.

    Stress Calculation (calculate_stress):
        Purpose: To calculate the stress for a given strain using material properties.
        Parameters:
            strain (array): Strain vector.
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
        Returns:
            Stress vector (array).

    B Matrix Computation (compute_B_matrix):
        Purpose: To compute the B matrix for a tetrahedral element.
        Parameters:
            tetrahedron_nodes: Coordinates of the tetrahedron's nodes.
        Returns:
            B matrix (numpy.ndarray).

Main Execution

The main code block initializes the material database, generates the tetrahedral mesh, assigns material properties, computes the global stiffness matrix, and solves for node displacements. It also uses mayavi for 3D visualization of the deformed structure.
Usage

    Setup:
        Install required Python packages (numpy, sqlite3, mayavi).
        Run the script in a Python environment.

    Customization:
        Modify material properties in the SQLite database.
        Adjust mesh dimensions and divisions for different structures.
        Alter applied forces and boundary conditions as per requirement.

Visualization

The mayavi package is used to visualize the deformed structure in 3D. It shows the initial and displaced positions of the nodes, giving a clear representation of how the structure responds to the applied forces.
Limitations and Considerations

    The current implementation assumes linear elastic materials.
    Boundary conditions and forces are simplified for demonstration purposes and need to be adapted for real-world scenarios.
    The visualization is basic and may need enhancements for more complex structures.

Future Enhancements

    Incorporating non-linear material behavior.
    Enhancing visualization capabilities.
    Integrating more complex loading and boundary condition scenarios.

Customization and Extensibility

    The script can be extended to include more complex loading conditions, different element types, or non-linear material behavior.
    Visualization and post-processing capabilities can be integrated for a more comprehensive FEA tool.

Limitations

    Currently limited to linear elastic materials and tetrahedral elements.
    Boundary conditions and forces are simplified and need refinement for complex real-world scenarios.
