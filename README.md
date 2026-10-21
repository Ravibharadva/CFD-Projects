# Computational Fluid Dynamics in Python — Three Student Projects

Three small, self-contained CFD projects that show three different ways of computing fluid flow, from a classical grid-based solver to a particle-based method to a neural network. Every project is a single, readable Python file, runs on a normal laptop, is validated against a reference solution, and has its own README explaining the theory with citations.

| # | Project | Method | Libraries | Validation |
|---|---|---|---|---|
| 1 | [Lid-driven cavity](01-lid-driven-cavity/) | Finite differences + projection method for the Navier–Stokes equations | NumPy, Matplotlib | Ghia et al. (1982) benchmark |
| 2 | [Flow past a cylinder](02-lattice-boltzmann-cylinder/) | Lattice-Boltzmann method (D2Q9, BGK) | NumPy, Matplotlib | Strouhal number of the vortex street |
| 3 | [Kovasznay flow with a PINN](03-pinn-kovasznay-flow/) | Physics-informed neural network, forward + inverse problem | PyTorch, NumPy, Matplotlib | Exact analytical solution |


## Why these three?

- **Project 1** solves the Navier–Stokes equations directly on a grid — the "textbook" CFD approach.
- **Project 2** reaches the same physics from a kinetic (particle distribution) model, which is simple, local and well suited to GPUs.
- **Project 3** replaces the grid with a neural network trained on the equations themselves, and also shows how to infer an unknown parameter (the Reynolds number) from sparse data.

Together they cover discretisation, stability, boundary conditions, validation and scientific machine learning.

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

python 01-lid-driven-cavity/cavity.py
python 02-lattice-boltzmann-cylinder/lbm_cylinder.py
python 03-pinn-kovasznay-flow/pinn_kovasznay.py
python 03-pinn-kovasznay-flow/pinn_kovasznay.py --mode inverse
```

Each script writes its figures to a `results/` folder next to it. Tested with Python 3.10+, NumPy 2, Matplotlib 3.x and PyTorch 2.x.

## Repository layout

```
.
├── README.md
├── 01-lid-driven-cavity/
│   ├── README.md
│   ├── cavity.py
│   └── results/
├── 02-lattice-boltzmann-cylinder/
│   ├── README.md
│   ├── lbm_cylinder.py
│   └── results/
└── 03-pinn-kovasznay-flow/
    ├── README.md
    ├── pinn_kovasznay.py
    └── results/
```


## Main references

- J. H. Ferziger, M. Perić, R. L. Street. *Computational Methods for Fluid Dynamics*, 4th ed. Springer, 2020.
- T. Krüger et al. *The Lattice Boltzmann Method: Principles and Practice*. Springer, 2017.
- M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *J. Comput. Phys.*, 378:686–707, 2019.

The full reference lists are in each project's README.
