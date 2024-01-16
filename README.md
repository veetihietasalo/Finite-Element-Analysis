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
