# Finite-Element-Analysis

A small linear elastic finite element solver in Python. It meshes a box into 4-node tetrahedra, assembles the stiffness matrix, solves K u = F with NumPy and recovers strain and stress in every element.

I started it in January 2024 as a side project, and the maths was wrong in a few places. In October 2026 it was fixed and checked against exact answers. That's the interesting part: 106 pytest tests compare the solver with rigid body modes, the patch test, a bar under load and a cantilever against beam theory.

## What's inside

`fem.py` is the whole solver, 359 lines.

- Mesh: a regular grid, each cell split into 6 tetrahedra around its main diagonal. Every element volume is positive and they add up to the box volume.
- Element: linear tetrahedron. B (strain from displacement) comes from the inverse of the matrix of rows [1, x, y, z], it's constant over the element, and K_e = V Bᵀ D B is exact. D is the isotropic elasticity matrix from E and ν.
- Solve: a dense 3N x 3N global K. Prescribed displacements are moved to the right-hand side, K_ff u_f = F_f − K_fc ū_c, so non-zero values work. Point loads, and consistent loads for a uniform traction on the surface.
- Stress: σ = D B u per element, plus von Mises.
- Materials: a SQLite table with Steel (210000 MPa, ν 0.3), Aluminum (69000, 0.33) and Concrete (30000, 0.2). Lengths in mm, forces in N, stresses in MPa.

## Run

Python 3.12 or newer.

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fem.py
python -m pytest
```

`fem.py` solves a steel cantilever, 10 x 1 x 1 mm, clamped at x = 0, with a 1 N tip load. `--divisions NX NY NZ` changes the mesh. `--plot` draws the deformed surface with mayavi, which isn't in `requirements.txt`. The plot hasn't been run.

## Results

The demo output (run on Python 3.14, about 1 s):

```
mesh 40 x 4 x 4 cells, 1025 nodes, 3840 tetrahedra, 3075 dof
tip deflection   1.510998e-02 mm
Euler-Bernoulli  1.904762e-02 mm  (FE / theory 0.793)
Timoshenko       1.919619e-02 mm  (FE / theory 0.787)
max |sigma_xx|   59.22 MPa
max von Mises    46.85 MPa
```

Linear tetrahedra are too stiff in bending, so coarse meshes come out low. Tip deflection against Timoshenko beam theory:

- 10 x 1 x 1 cells (132 dof): 4.211e-3 mm, 0.219
- 20 x 2 x 2 (567 dof): 9.669e-3 mm, 0.504
- 40 x 4 x 4 (3075 dof): 1.511e-2 mm, 0.787

The tests also check that the deflection grows with the mesh and stays below the beam value. A 10 x 1 x 1.5 bar pulled with 1000 N gives an end displacement of 3.174603e-2 mm, the same as F L / (E A).

`python -m pytest`: 106 passed in 1.4 s.

## How it was built

The 2024 code is mine (commits as Venomity). The October 2026 rewrite into `fem.py`, the fixes and the 106 tests were written by coding agents (Mad Tech Agents, 10 of the 16 commits).

## What I learned

- My 2024 element stiffness was wrong three ways: B was filled with raw node coordinates instead of shape function derivatives, D had the bulk modulus K where K + 4G/3 belongs, and K_e was scaled by 6V instead of V.
- The 2024 mesh used 5 tetrahedra per cell and covered 5/6 of the box. It ran without complaint, so a plausible number is no proof.
- The 2024 code had no tests. The new ones check exact answers: rigid body motion must give zero force, a patch test must reproduce a linear field, a bar must match F L / (E A).
- Linear tetrahedra lock in bending, so a coarse mesh can be 21% low (40 x 4 x 4) and still run clean.

## Left

- Linear tetrahedra only. 10-node quadratic ones would fix the bending.
- Dense K and `numpy.linalg.solve`, so a sparse solver is needed for large meshes.
- Box meshes only, one material per model, no body loads such as gravity.
- The 2024 plans for non-linear materials, better visualisation and more load cases weren't started.

No license file in this repo yet.
