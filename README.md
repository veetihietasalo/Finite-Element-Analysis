# Finite-Element-Analysis

A small linear elastic finite element solver in Python, an old math-inspired side project from January 2024. It meshes a box into 4-node tetrahedra, reads material properties from SQLite, assembles the global stiffness matrix, solves $K u = F$ with NumPy and recovers the strain and stress in every element.

The 2024 version had the maths wrong in a few places: the element stiffness, the mesh, the boundary condition values and prescribed displacements. Those are fixed in October 2026 and covered by 106 pytest tests in `tests/`, which check the solver against exact answers (rigid body modes, the patch test, a bar under end load) and a cantilever against beam theory. What was wrong in the 2024 code is listed at the end.

## Run

Python 3.12 or newer (numpy 2.5.4 needs 3.12).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fem.py
.venv/bin/python -m pytest
```

`fem.py` is the whole solver. Running it solves a steel cantilever, 10 x 1 x 1 mm, clamped at x = 0, with a 1 N tip load spread over the end face as a uniform shear traction:

```
mesh 40 x 4 x 4 cells, 1025 nodes, 3840 tetrahedra, 3075 dof
tip deflection   1.510998e-02 mm
Euler-Bernoulli  1.904762e-02 mm  (FE / theory 0.793)
Timoshenko       1.919619e-02 mm  (FE / theory 0.787)
max |sigma_xx|   59.22 MPa
max von Mises    46.85 MPa
```

Linear tetrahedra are too stiff in bending, so this mesh gives 79% of the beam theory deflection. See the cantilever numbers below for finer meshes.

`--divisions NX NY NZ` changes the mesh. `--plot` draws the deformed outer surface with mayavi, which isn't in `requirements.txt` (`pip install mayavi`). mayavi is imported only inside `plot_mesh`, so everything else runs without it. We didn't have mayavi installed, so the plot itself hasn't been run, the test only checks the call with a stand-in module.

The demo creates `materials.db` in the current directory. Young's modulus is stored in MPa, so lengths in mm and forces in N give displacements in mm and stresses in MPa.

## How it works

Mesh. `generate_tetrahedral_mesh(length, width, height, (nx, ny, nz))` puts the nodes on a regular grid and splits each cell into 6 tetrahedra around its main diagonal n0-n7, one for every path from n0 to n7 along the cell edges. With n0 = (i, j, k), n1 = +x, n2 = +y, n3 = +x+y, n4 = +z, n5 = +x+z, n6 = +y+z and n7 = +x+y+z, the elements are (n0, n1, n3, n7), (n0, n3, n2, n7), (n0, n2, n6, n7), (n0, n6, n4, n7), (n0, n4, n5, n7) and (n0, n5, n1, n7). Every cell uses the same split, so the face diagonals of neighbouring cells match, and each element has volume $V_\text{cell}/6 > 0$ in this node order.

Element. A linear tetrahedron has shape functions $N_i = a_i + b_i x + c_i y + d_i z$ with $N_i(\mathbf{x}_j) = \delta_{ij}$. With the nodes as the rows of

```math
A = \begin{bmatrix} 1 & x_1 & y_1 & z_1 \\ 1 & x_2 & y_2 & z_2 \\ 1 & x_3 & y_3 & z_3 \\ 1 & x_4 & y_4 & z_4 \end{bmatrix}
```

the coefficients of $N_i$ are column $i$ of $A^{-1}$, so the derivatives $(b_i, c_i, d_i)$ are rows 2 to 4 of $A^{-1}$, and $V = \det A / 6$. Strains are $\varepsilon = [\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \gamma_{xy}, \gamma_{yz}, \gamma_{xz}]^T$ with engineering shear strains, and $\varepsilon = B u_e$ for the element's nodal displacements $u_e = [u_1, v_1, w_1, \dots, u_4, v_4, w_4]^T$:

```math
B = \begin{bmatrix} B_1 & B_2 & B_3 & B_4 \end{bmatrix}, \qquad
B_i = \begin{bmatrix}
b_i & 0 & 0 \\ 0 & c_i & 0 \\ 0 & 0 & d_i \\ c_i & b_i & 0 \\ 0 & d_i & c_i \\ d_i & 0 & b_i
\end{bmatrix}
```

Isotropic material, with $\lambda = \dfrac{E\nu}{(1+\nu)(1-2\nu)}$ and $\mu = G = \dfrac{E}{2(1+\nu)}$:

```math
D = \begin{bmatrix}
\lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
\lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}
```

In terms of the bulk modulus $K$, $\lambda + 2\mu = K + 4G/3$ and $\lambda = K - 2G/3$. B is constant over the element, so the stiffness integral is exact:

```math
K_e = \int_{V_e} B^T D B \, dV = V \, B^T D B
```

Assembly adds each 12 x 12 $K_e$ into a dense 3N x 3N matrix at the rows and columns $3n, 3n+1, 3n+2$ of its 4 nodes.

Boundary conditions. Every node gets a dict like `{'x': 0.0, 'y': 'free', 'z': 0.001}`. With $c$ the prescribed and $f$ the free degrees of freedom, `solve_for_displacements` sets $u_c = \bar u_c$ and solves

```math
K_{ff} \, u_f = F_f - K_{fc} \, \bar u_c
```

The reactions are $K u - F$ at the prescribed degrees of freedom.

Loads. `apply_forces` and `build_force_vector` for point forces. `traction_force_vector` for a uniform traction $t$ on part of the surface: each boundary triangle passes $t A_\triangle / 3$ to each of its 3 nodes, the consistent load for linear elements.

Stress. In each element $\varepsilon_e = B u_e$ and $\sigma_e = D \varepsilon_e$ (`compute_element_stresses`), constant over the element. `von_mises` gives

```math
\sigma_{vm} = \sqrt{\tfrac12\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3\left(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{xz}^2\right)}
```

## Tests

`python -m pytest` runs 106 tests in about 3 s. Run on Python 3.12.15 and 3.14.8 with numpy 2.5.4 and pytest 9.1.1. Steel, E = 210000 MPa and ν = 0.3, unless noted.

- Rigid body modes, one element: the 3 translations and 3 small rotations $u = \omega \times x$ give zero nodal forces on 12 elements (unit tet, 5 random tets, the 6 tets of a 2.5 x 1 x 0.5 cell). Worst $\max|K_e r| / (\max|K_e| \max|r|)$ is 2.2e-16.
- Rigid body modes, assembled: same on a 3 x 2 x 2 cell mesh (36 nodes), regular and with the interior nodes moved by up to 0.15 cell sizes. Residual 1.0e-16 and 1.2e-16, and K has exactly 6 zero eigenvalues.
- $K_e$ is symmetric positive semi-definite with exactly 6 zero eigenvalues, for the 12 elements and ν = 0, 0.3 and 0.45. Unit tet: the 6 zero eigenvalues are below 7e-17 of the largest, the smallest nonzero one is 0.128 of it.
- B gives the exact strain of a linear displacement field, sign included. For a linear field $\tfrac12 u^T K_e u = \tfrac12 V \varepsilon^T D \varepsilon$, which pins the scaling to V.
- D: diagonal K + 4G/3, and the uniaxial strain (ε, −νε, −νε) gives the stress (Eε, 0, 0).
- Mesh, 4 boxes: 6 nx ny nz elements, every element volume is cell / 6 > 0 and the total is the box volume to 1e-12 (unit cube 1.0, the 2024 mesh gave 0.833). Conforming: every face is shared by 2 elements or lies on the box surface, and the 4(nx ny + ny nz + nx nz) surface triangles add up to the box surface area.
- Patch test: 2 x 1.5 x 1 box, 4 x 3 x 3 cells (80 nodes, 12 interior, 216 tets), regular and with the interior nodes moved by up to 0.15 cell sizes. $u = a + G x$ prescribed on the 68 surface nodes, no load. The interior displacements come out exact to 1.3e-18 (|u| up to 5.3e-3), every element strain to 5.2e-18 (|ε| up to 2.8e-3) and every element stress to 1.1e-12 MPa (|σ| up to 456 MPa).
- Bar under end load: 10 x 1 x 1.5 bar, 10 x 2 x 3 cells (132 nodes, 360 tets), symmetry planes x = 0, y = 0 and z = 0, 1000 N on the end face. End displacement 3.174603174603181e-02 mm vs F L / (E A) = 3.174603174603174e-02 mm, relative error 2.7e-15. σxx = F / A = 666.67 MPa in every element, the lateral contraction −ν σ W / E is exact and the reaction at x = 0 is −1000 N.
- Prescribed end displacement: same bar, u_x = 0.02 mm on the end face, no load. Reaction 630 N vs E A δ / L = 630 N (relative error 2.3e-15), and u_x = δ x / L at every node to 2.4e-17.
- Prescribed face motion: one face translated, or rotated by a small rotation, no load. Every free node follows to 1e-12.
- `apply_boundary_conditions` stores the value passed in. The solver leaves K and F as they were and rejects values other than `'free'` or a number.
- Cantilever, below.
- Importing `fem` prints nothing, writes no file and doesn't import mayavi. The demo runs. The materials table round-trips.

With the 2024 version of a function put back into `fem.py`, its tests fail: 37 for the element stiffness, 16 for the old `compute_B_matrix`, 17 for the mesh, 8 for the solver, and 3 failures and 3 errors for `apply_boundary_conditions`.

### Cantilever

Steel, 10 x 1 x 1 mm, clamped at x = 0 (all three components on the whole face), 1 N tip load P as a uniform shear traction on the end face. The tip deflection is the load-weighted mean of the end face displacement, $F \cdot u / P$. Theory: Euler-Bernoulli $\delta = P L^3 / (3 E I)$ = 1.9048e-2 mm, Timoshenko adds $P L / (\kappa G A)$ with κ = 5/6, 1.9196e-2 mm.

| cells | nodes | dof | tip deflection (mm) | FE / Timoshenko | tip u_y / u_z |
|---|---|---|---|---|---|
| 10 x 1 x 1 | 44 | 132 | 4.2112e-3 | 0.219 | −0.268 |
| 20 x 2 x 2 | 189 | 567 | 9.6691e-3 | 0.504 | −0.172 |
| 40 x 4 x 4 | 1025 | 3075 | 1.5110e-2 | 0.787 | −0.070 |
| 60 x 6 x 6 | 2989 | 8967 | 1.7027e-2 | 0.887 | −0.035 |
| 80 x 8 x 8 | 6561 | 19683 | 1.7844e-2 | 0.930 | −0.021 |
| 120 x 12 x 12 | 20449 | 61347 | 1.8492e-2 | 0.963 | −0.010 |
| 160 x 16 x 16 | 46529 | 139587 | 1.8735e-2 | 0.976 | −0.005 |

Linear tetrahedra lock in bending, so the coarse meshes are far off: 21% low at 40 x 4 x 4, 11% at 60 x 6 x 6. The deflection grows with the mesh and approaches the Timoshenko value from below, as it should for a displacement method with consistent loads. The rows from 80 x 8 x 8 up were run once with the same element functions and a `scipy.sparse` solve outside the repo, since the dense K would need 3 GB, 30 GB and 156 GB.

The same split in every cell isn't mirror symmetric, so a z load also moves the tip sideways in y. That's mesh bias and it goes away with the mesh (last column).

The tests run the first three rows. They check equilibrium (the clamp reaction is 1 N), that the deflection grows with the mesh, stays below Timoshenko and is at least 0.75 of it at 40 x 4 x 4, and that the sideways ratio shrinks. Outside the tests: bending stress σxx in the middle 60% of the span is 0.84 of beam theory at 40 x 4 x 4 and 0.92 at 60 x 6 x 6 (least squares fit against $P (L - x)(z - H/2) / I$ at the element centroids).

## Functions

All in `fem.py`.

- `create_materials_database(db_name)` and `assign_material_properties(elements, db_name, material_name)`: SQLite table `materials` (`name`, `youngs_modulus`, `poisson_ratio`) with Steel (210000, 0.3), Aluminum (69000, 0.33) and Concrete (30000, 0.2). Unknown names raise `ValueError`.
- `generate_tetrahedral_mesh(length, width, height, divisions)`: returns `(nodes, elements)`, nodes as `(x, y, z)` tuples, elements as 4-tuples of node indices.
- `tetrahedron_volume(coords)` (signed), `boundary_faces(elements)` (outer surface triangles).
- `compute_B_matrix(coords)`, `elasticity_matrix(E, nu)`, `compute_element_stiffness_matrix(coords, E, nu)`, `element_dofs(element)`, `assemble_global_stiffness_matrix(elements, nodes, E, nu)`. One material per model.
- `apply_boundary_conditions(nodes, constraints, node_ids=None)` and `apply_forces(nodes, forces, node_ids=None)`: one dict per node, the given values on `node_ids` (all nodes if None), free or zero on the rest. `build_force_vector(node_forces)`, `traction_force_vector(nodes, elements, node_ids, traction)`.
- `solve_for_displacements(K, F, constraints)`: returns u (length 3N), doesn't change K or F.
- `calculate_stress(strain, E, nu)`, `compute_element_stresses(elements, nodes, u, E, nu)` returning `(strains, stresses)` of shape (elements, 6), `von_mises(stress)`.
- `plot_mesh(nodes, elements, displacements, scale)`: mayavi wireframe of the deformed outer surface, coloured by |u|.

Example, the same cantilever as the demo:

```python
import numpy as np
import fem

fem.create_materials_database("materials.db")
nodes, elements = fem.generate_tetrahedral_mesh(10.0, 1.0, 1.0, (40, 4, 4))
props = fem.assign_material_properties(elements, "materials.db", "Steel")
E, nu = props[0]["youngs_modulus"], props[0]["poisson_ratio"]

# clamp the face x = 0, pull the face x = 10 down with 1 N/mm^2 (1 N on the 1 x 1 mm face)
P = np.array(nodes)
clamped = np.flatnonzero(np.isclose(P[:, 0], 0.0))
tip = np.flatnonzero(np.isclose(P[:, 0], 10.0))
constraints = fem.apply_boundary_conditions(nodes, {"x": 0.0, "y": 0.0, "z": 0.0}, node_ids=clamped)
F = fem.traction_force_vector(nodes, elements, tip, (0.0, 0.0, -1.0))

K = fem.assemble_global_stiffness_matrix(elements, nodes, E, nu)
u = fem.solve_for_displacements(K, F, constraints)
strains, stresses = fem.compute_element_stresses(elements, nodes, u, E, nu)

print("mean tip displacement", u.reshape(-1, 3)[tip].mean(axis=0))
print("max von Mises", fem.von_mises(stresses).max())
```

## What was wrong in the 2024 version

The January 2024 code had the functions copied across `generate_model.py`, `generate_mesh.py` and `generate_calculate.py`, with differences between the copies.

- Element stiffness: B was filled with the raw node coordinates instead of the shape function derivatives, D had the bulk modulus K on the diagonal where it needs K + 4G/3, and $K_e$ was scaled by 6V instead of V. A unit translation of a unit tet gave nodal forces with a norm of 2.9e5 instead of 0. Now $K_e = V B^T D B$.
- `compute_B_matrix` had the right derivatives with the sign flipped and wasn't used. B now comes from $A^{-1}$ and is the one used for the stiffness and the stresses.
- Mesh: 5 tetrahedra per cell with (n0, n5, n1, n7) left out, so it covered 5/6 of the box. Now all 6.
- `apply_boundary_conditions` stored the node's own coordinate instead of the value passed in.
- `solve_for_displacements` zeroed the constrained rows and columns without moving $K_{fc} \bar u_c$ to the right-hand side, so only zero prescribed displacements were right. Now it solves the partitioned system above.
- `calculate_stress` existed but nothing called it. Now `compute_element_stresses` gives strain and stress per element.
- `generate_calculate.py`'s assembly failed with `IndexError`, `import generate_model` failed with `NameError` (demo code at module level) and `import generate_mesh` ran its demo. All three files are replaced by `fem.py`, with the demo under `if __name__ == "__main__":` and mayavi imported only for the plot. The old files are in the git history.
- The `Pipfile` asked for Python 3.11 and matplotlib, which nothing imported. Replaced by `requirements.txt` with numpy and pytest pinned. `materials.db` is no longer tracked, the demo creates it.

## Status

Old side project, fixed in October 2026. Left open:

- Linear tetrahedra only. They need fine meshes in bending (11% low at 60 x 6 x 6). 10-node quadratic tets would fix that.
- Dense K and `numpy.linalg.solve`. 60 x 6 x 6 (8967 dof) takes about 2 GB and 2 s, much past that needs a sparse solver.
- Box meshes only, one material per model, no body loads such as gravity.
- The mayavi plot hasn't been run.
- The 2024 plans for non-linear materials, better visualization and more load cases weren't started.
