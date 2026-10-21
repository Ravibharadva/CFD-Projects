"""
Project 1 - Lid-Driven Cavity Flow (2D incompressible Navier-Stokes)
=====================================================================

Solves the steady lid-driven cavity problem at Re = 100 with a simple
finite-difference projection (fractional-step) method written in NumPy.
The result is compared with the benchmark data of Ghia, Ghia & Shin (1982).

Run:
    python cavity.py

Outputs (written to ./results):
    velocity_streamlines.png  - speed contour + streamlines
    pressure.png              - pressure field
    ghia_comparison.png       - centre-line profiles vs. Ghia et al. (1982)
    convergence.png           - residual history
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Problem set-up
# ---------------------------------------------------------------------------
N = 81                # grid points in each direction (N x N)
L = 1.0               # cavity side length
U_LID = 1.0           # lid velocity
RE = 100.0            # Reynolds number  Re = U_LID * L / nu
NU = U_LID * L / RE   # kinematic viscosity
RHO = 1.0             # density

dx = dy = L / (N - 1)
# Time step from the two usual explicit stability limits (CFL + diffusion),
# multiplied by a safety factor.
DT = 0.25 * min(dx / U_LID, 0.25 * dx**2 / NU)

MAX_STEPS = 30000     # upper limit of time steps
TOL = 1e-7            # steady-state tolerance on max |u^{n+1} - u^n|
P_ITERS = 50          # Jacobi sweeps for the pressure Poisson equation

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)  # arrays are indexed [j, i] = [y, x]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------
def apply_velocity_bc(u, v):
    """No-slip walls everywhere, except the top wall which moves with U_LID."""
    u[0, :] = 0.0;  v[0, :] = 0.0      # bottom wall
    u[:, 0] = 0.0;  v[:, 0] = 0.0      # left wall
    u[:, -1] = 0.0; v[:, -1] = 0.0     # right wall
    u[-1, :] = U_LID; v[-1, :] = 0.0   # moving lid
    return u, v


def apply_pressure_bc(p):
    """dp/dn = 0 on all walls.

    With Neumann conditions on every wall the pressure is only defined up to
    an additive constant (only grad p enters the momentum equation).  To keep
    that constant from drifting, we shift the field so that its mean is zero.
    """
    p[:, -1] = p[:, -2]
    p[:, 0] = p[:, 1]
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p -= p.mean()
    return p


def poisson_rhs(u, v):
    """Right-hand side of the pressure Poisson equation:  (rho/dt) * div(u*)."""
    b = np.zeros_like(u)
    dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    b[1:-1, 1:-1] = RHO / DT * (dudx + dvdy)
    return b


def solve_pressure(p, b):
    """Jacobi iterations for  lap(p) = b  (dx = dy)."""
    for _ in range(P_ITERS):
        pn = p.copy()
        p[1:-1, 1:-1] = 0.25 * (pn[1:-1, 2:] + pn[1:-1, :-2]
                                + pn[2:, 1:-1] + pn[:-2, 1:-1]
                                - dx**2 * b[1:-1, 1:-1])
        p = apply_pressure_bc(p)
    return p


def predictor(u, v):
    """Intermediate velocity u* = u^n + dt * (- convection + diffusion).

    Convection: first-order upwind (robust and easy to read).
    Diffusion : second-order central differences.
    """
    us, vs = u.copy(), v.copy()
    uc, vc = u[1:-1, 1:-1], v[1:-1, 1:-1]

    def convect(f):
        # upwind derivative of field f, choosing the side the flow comes from
        dfdx_b = (f[1:-1, 1:-1] - f[1:-1, :-2]) / dx
        dfdx_f = (f[1:-1, 2:] - f[1:-1, 1:-1]) / dx
        dfdy_b = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dy
        dfdy_f = (f[2:, 1:-1] - f[1:-1, 1:-1]) / dy
        dfdx = np.where(uc > 0, dfdx_b, dfdx_f)
        dfdy = np.where(vc > 0, dfdy_b, dfdy_f)
        return uc * dfdx + vc * dfdy

    def laplacian(f):
        return ((f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2
                + (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2)

    us[1:-1, 1:-1] = uc + DT * (-convect(u) + NU * laplacian(u))
    vs[1:-1, 1:-1] = vc + DT * (-convect(v) + NU * laplacian(v))
    return apply_velocity_bc(us, vs)


def corrector(us, vs, p):
    """Project u* onto a divergence-free field:  u^{n+1} = u* - dt/rho * grad p."""
    u, v = us.copy(), vs.copy()
    u[1:-1, 1:-1] -= DT / RHO * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] -= DT / RHO * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    return apply_velocity_bc(u, v)


# ---------------------------------------------------------------------------
# 3. Main time loop
# ---------------------------------------------------------------------------
def run():
    u = np.zeros((N, N)); v = np.zeros((N, N)); p = np.zeros((N, N))
    u, v = apply_velocity_bc(u, v)
    residuals = []

    print(f"Grid {N}x{N}, Re = {RE:.0f}, dt = {DT:.2e}")
    t0 = time.time()
    for n in range(1, MAX_STEPS + 1):
        us, vs = predictor(u, v)                 # step 1: predictor
        p = solve_pressure(p, poisson_rhs(us, vs))  # step 2: pressure
        un, vn = corrector(us, vs, p)            # step 3: corrector

        res = max(np.abs(un - u).max(), np.abs(vn - v).max())
        residuals.append(res)
        u, v = un, vn
        if n % 1000 == 0:
            print(f"  step {n:6d}   residual = {res:.3e}")
        if res < TOL:
            print(f"Converged after {n} steps (residual {res:.2e}).")
            break
    print(f"Wall time: {time.time() - t0:.1f} s")
    return u, v, p, np.array(residuals)


# ---------------------------------------------------------------------------
# 4. Benchmark data: Ghia, Ghia & Shin (1982), Re = 100
#    Table I  (u along vertical centre-line x = 0.5)
#    Table II (v along horizontal centre-line y = 0.5)
# ---------------------------------------------------------------------------
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                   0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                   0.0625, 0.0547, 0.0000])
GHIA_U = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
                   0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                   -0.06434, -0.04775, -0.04192, -0.03717, 0.00000])
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                   0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                   0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                   -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,
                   0.12317, 0.10890, 0.10091, 0.09233, 0.00000])


# ---------------------------------------------------------------------------
# 5. Post-processing
# ---------------------------------------------------------------------------
def plot_results(u, v, p, residuals):
    speed = np.sqrt(u**2 + v**2)
    mid = N // 2

    # (a) speed + streamlines
    fig, ax = plt.subplots(figsize=(6, 5.4))
    cf = ax.contourf(X, Y, speed, levels=40, cmap="viridis")
    ax.streamplot(X, Y, u, v, color="white", density=1.4, linewidth=0.7)
    fig.colorbar(cf, ax=ax, label="|u|")
    ax.set(title=f"Lid-driven cavity, Re = {RE:.0f}", xlabel="x", ylabel="y",
           aspect="equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "velocity_streamlines.png"), dpi=150)
    plt.close(fig)

    # (b) pressure
    # The two top corners are singular points (the moving lid meets a fixed
    # wall), so p is very large there.  Clip the colour range to the 1st-99th
    # percentile so the pressure distribution in the rest of the cavity is visible.
    lo, hi = np.percentile(p, [1, 99])
    fig, ax = plt.subplots(figsize=(6, 5.4))
    cf = ax.contourf(X, Y, np.clip(p, lo, hi), levels=np.linspace(lo, hi, 41),
                     cmap="coolwarm")
    ax.contour(X, Y, np.clip(p, lo, hi), levels=15, colors="k", linewidths=0.4)
    fig.colorbar(cf, ax=ax, label="p (mean set to 0)")
    ax.set(title="Pressure field", xlabel="x", ylabel="y", aspect="equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "pressure.png"), dpi=150)
    plt.close(fig)

    # (c) validation against Ghia et al.
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot(u[:, mid], y, "b-", label="This code (NumPy)")
    axs[0].plot(GHIA_U, GHIA_Y, "ko", mfc="none", label="Ghia et al. (1982)")
    axs[0].set(xlabel="u at x = 0.5", ylabel="y", title="u along vertical centre-line")
    axs[1].plot(x, v[mid, :], "r-", label="This code (NumPy)")
    axs[1].plot(GHIA_X, GHIA_V, "ko", mfc="none", label="Ghia et al. (1982)")
    axs[1].set(xlabel="x", ylabel="v at y = 0.5", title="v along horizontal centre-line")
    for a in axs:
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "ghia_comparison.png"), dpi=150)
    plt.close(fig)

    # (d) convergence history
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(residuals)
    ax.set(xlabel="time step", ylabel="max |u^{n+1} - u^n|", title="Convergence to steady state")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig)

    # simple error numbers against the benchmark (linear interpolation)
    err_u = np.abs(np.interp(GHIA_Y, y, u[:, mid]) - GHIA_U).max()
    err_v = np.abs(np.interp(GHIA_X, x, v[mid, :]) - GHIA_V).max()
    print(f"Max deviation from Ghia et al.:  u -> {err_u:.4f},  v -> {err_v:.4f}")


if __name__ == "__main__":
    u, v, p, residuals = run()
    plot_results(u, v, p, residuals)
    print(f"Figures saved in {OUT_DIR}")