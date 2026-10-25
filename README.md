# Finite-Element-Analysis

A small linear elastic finite element solver in Python, an old math-inspired side project from January 2024. It meshes a box into tetrahedra, reads material properties from SQLite, assembles a global stiffness matrix and solves K u = F with NumPy. It was left unfinished: the element stiffness matrix is wrong, so the displacements it gives are not physical.

What works:

- Box mesh: nodes on a regular grid, 5 tetrahedra per cell.
- Materials table in SQLite (Steel, Aluminum, Concrete).
- Assembly into a dense 3N x 3N matrix (N nodes) and a solve with fixed degrees of freedom.
- `generate_model.py` runs the whole chain on a 12-node demo (36 x 36 matrix) and ends with a mayavi wireframe of the deformed mesh.

What's missing or wrong:

- `compute_element_stiffness_matrix` uses node coordinates where it needs shape function derivatives, has the wrong diagonal in the material matrix and an extra factor 6. `compute_B_matrix` has the right derivatives (with the sign flipped) but isn't used yet.
- The mesh leaves out 1 of the 6 tetrahedra per cell, so it covers 5/6 of the box.
- `apply_boundary_conditions` fixes an axis to the node's own coordinate, not to the value passed in.
- `solve_for_displacements` handles only zero prescribed displacements correctly.
- No stresses are computed from the solution, and there are no tests.
- The functions are copied across three files, and no single file has the full set in a form you can import (see Files).

Each point has the details in the function reference below.

## Stack

Python, NumPy, `sqlite3` from the standard library, mayavi for the 3D plot. The `Pipfile` asks for Python 3.11 and lists only matplotlib, which none of the scripts import.

## Run

```
pip install numpy
python generate_model.py
python generate_mesh.py
```

Both scripts create or update `materials.db` in the current directory. The plot at the end of `generate_model.py` needs mayavi (`pip install mayavi`). We ran both scripts with numpy 2.5.4 and no mayavi: `generate_model.py` printed its results and then stopped with `ModuleNotFoundError: No module named 'mayavi'`, and `generate_mesh.py` ran to the end.

## Files

- `generate_model.py`: every function below except `apply_boundary_conditions` and `apply_forces`, plus a demo. The demo meshes a 10 x 1 x 1 box into 2 x 1 x 1 cells (12 nodes, 10 tetrahedra), uses steel, fixes node 0 in x, y and z and applies no forces. It prints the 36 x 36 stiffness matrix and the displacements (all zero, since there is no load), then draws the deformed mesh as a mayavi wireframe. Only the start of the demo is under `if __name__ == "__main__":`, so `import generate_model` fails with a `NameError`.
- `generate_mesh.py`: the materials, mesh, boundary condition, force and element stiffness functions, with demo code at module level, so importing it runs the demo. The demo computes the stiffness of one unit tetrahedron (it prints the inputs, not the matrix), meshes a 4 x 2 x 3 box into 2 x 1 x 1 cells, prints the constraints and forces (Fy = -9.81 on every node) for the first 5 nodes and prints the element properties. It doesn't assemble or solve.
- `generate_calculate.py`: function definitions only. Its `assemble_global_stiffness_matrix` is broken (see below).
- `materials.db`: SQLite file with the `materials` table and rows for Concrete, Steel and Aluminum.

## Function reference

### `create_materials_database(db_name: str)`

In all three files. Creates the table `materials` (`name` TEXT primary key, `youngs_modulus` REAL, `poisson_ratio` REAL) in the SQLite file `db_name` if it doesn't exist, then inserts or replaces Steel (210000, 0.3) and Aluminum (69000, 0.33). The versions in `generate_mesh.py` and `generate_calculate.py` also add Concrete (30000, 0.2). Young's modulus is in MPa, going by the comment in `generate_mesh.py`.

### `generate_tetrahedral_mesh(length: float, width: float, height: float, divisions: Tuple[int, int, int])`

In all three files. Splits a box of `length` x `width` x `height` into `divisions` cells along x, y and z, with nodes on the grid points. Each cell gets 5 tetrahedra that share the diagonal from corner n0 to corner n7.

Returns `(nodes, elements)`: `nodes` is a list of `(x, y, z)` tuples and `elements` a list of 4-tuples of node indices.

The sixth tetrahedron of that split, `(n0, n5, n1, n7)`, is missing, so the elements cover 5/6 of each cell (total volume 0.833 on a unit cube). The code comment says the 5 fill the cube.

### `extract_triangles_from_tetrahedra(elements: List[Tuple[int, int, int, int]])`

In `generate_model.py`. Returns the 4 faces of every tetrahedron as 3-tuples of node indices, shared faces included twice.

### `extract_unique_triangles_from_tetrahedra(elements: List[Tuple[int, int, int, int]])`

In `generate_model.py`. Same as above, but each face once (node indices sorted, collected in a set). The mayavi plot uses this one.

### `assign_material_properties(elements: List[Tuple[int, int, int, int]], db_name: str, material_name: str)`

In all three files. Looks up `material_name` in the database and returns one `{'youngs_modulus': E, 'poisson_ratio': nu}` dict per element, the same values for every element. Raises `ValueError` if the material isn't in the table.

### `apply_boundary_conditions(nodes: List[Tuple[float, float, float]], constraints: Dict[str, Union[float, str]])`

In `generate_mesh.py` and `generate_calculate.py`. Applies the same constraint dict to every node and returns one dict per node. Keys are `'x'`, `'y'`, `'z'`, and a value is either `'free'` or a number.

For an axis that isn't `'free'`, it stores the node's own coordinate on that axis, not the number passed in. The `generate_mesh.py` demo shows it: `'y': 0` comes out as 0.0 or 2.0 depending on the node. `solve_for_displacements` then uses that coordinate as the prescribed displacement.

### `apply_forces(nodes: List[Tuple[float, float, float]], forces: Dict[str, float])`

In `generate_mesh.py` and `generate_calculate.py`. Copies the force dict (keys `'Fx'`, `'Fy'`, `'Fz'`, other keys are dropped) to every node and returns one dict per node. Building the force vector `F` from these is left to the caller, see the loop in the example below.

### `compute_element_stiffness_matrix(element_coordinates: np.ndarray, youngs_modulus: float, poisson_ratio: float)`

In all three files. Takes the 4 x 3 node coordinates of one tetrahedron and returns a symmetric 12 x 12 matrix. The volume V is computed correctly. The rest isn't right yet:

- The B matrix is filled with the raw node coordinates instead of the shape function derivatives.
- The material matrix has the bulk modulus K on the diagonal where isotropic elasticity needs K + 4G/3. The off-diagonal K - 2G/3 and the shear terms G are right.
- The result is scaled by 6V instead of V.

A rigid translation should give zero nodal forces. For a unit tetrahedron with steel values, a unit translation in x gives forces with a norm of about 2.9e5. The code still has the comment "Replace with actual computation".

### `assemble_global_stiffness_matrix(elements: np.ndarray, nodes: np.ndarray, youngs_modulus: float, poisson_ratio: float)`

In `generate_model.py` (works) and `generate_calculate.py` (broken). Builds a dense 3N x 3N matrix with 3 degrees of freedom per node and adds each element's 12 x 12 matrix at its nodes' rows and columns. One E and nu for the whole model.

In `generate_calculate.py` the degree-of-freedom indices are built from a generator (`np.array([3 * n, 3 * n + 1, 3 * n + 2] for n in element)`), which fails with `IndexError`.

### `solve_for_displacements(K: np.ndarray, F: np.ndarray, constraints: List[Dict[str, Union[float, str]]])`

In `generate_model.py` and `generate_calculate.py`. For every constrained degree of freedom it zeroes that row and column of K, puts 1 on the diagonal and writes the prescribed value into F. Then `np.linalg.solve(K, F)` returns the displacement vector (length 3N). K and F are changed in place. A commented-out line uses `scipy.linalg.solve` instead.

The column of a constrained degree of freedom is zeroed without moving its value to the right-hand side of the other equations, so only zero prescribed displacements come out right.

### `calculate_stress(strain, E, nu)`

In `generate_model.py`. Multiplies a 6-component strain vector (3 normal strains, then 3 engineering shear strains) by the isotropic linear elastic matrix (C11, C12, C44) and returns the stress vector. Nothing calls it yet.

### `compute_B_matrix(tetrahedron_nodes)`

In `generate_model.py`. Computes the 6 x 12 strain-displacement matrix of a linear tetrahedron from its 4 node coordinates, with strains in the order xx, yy, zz, xy, yz, xz. Raises `ValueError` for a zero-volume element. Checked against a linear displacement field, it gives the exact strains with the sign flipped. Nothing calls it yet.

## Example

The intended flow with the functions above. Run as written, with the functions from `generate_model.py` plus `apply_boundary_conditions` and `apply_forces` from `generate_calculate.py`, it stops in `solve_for_displacements` with `numpy.linalg.LinAlgError: Singular matrix`, because of the stiffness matrix problem above.

```python
import numpy as np

db_name = "materials.db"
create_materials_database(db_name)

nodes, elements = generate_tetrahedral_mesh(10.0, 5.0, 5.0, (10, 5, 5))

element_properties = assign_material_properties(elements, db_name, "Steel")
youngs_modulus = element_properties[0]['youngs_modulus']
poisson_ratio = element_properties[0]['poisson_ratio']

# fixes x on every node, to that node's own x coordinate
node_constraints = apply_boundary_conditions(nodes, {'x': 0, 'y': 'free', 'z': 'free'})
node_forces = apply_forces(nodes, {'Fx': 100, 'Fy': 0, 'Fz': 0})

K = assemble_global_stiffness_matrix(np.array(elements), np.array(nodes), youngs_modulus, poisson_ratio)

F = np.zeros(len(nodes) * 3)
for i, force in enumerate(node_forces):
    F[3 * i] = force['Fx']
    F[3 * i + 1] = force['Fy']
    F[3 * i + 2] = force['Fz']

displacements = solve_for_displacements(K, F, node_constraints)
```

## Status

Old side project, not being worked on. The code was committed on 2024-01-16 and the README updated on 2024-01-29, no commits since.

Left open: the element stiffness matrix and the missing tetrahedron. The original README also planned non-linear materials, better visualization and more load and boundary condition cases.
