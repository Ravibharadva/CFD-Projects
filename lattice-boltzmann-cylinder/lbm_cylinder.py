"""
Project 2 - Lattice-Boltzmann Method: Flow Past a Cylinder (D2Q9, BGK)
=======================================================================

A compact, fully vectorised NumPy implementation of the lattice-Boltzmann
method (LBM).  A uniform flow hits a circular cylinder at Re = 100; after a
short transient the wake becomes unsteady and a von Karman vortex street
appears.  The shedding frequency is measured and turned into a Strouhal
number, which is compared with experiments (Williamson, 1996).

Run:
    python lbm_cylinder.py            # full run  (~3-6 min on a laptop)
    python lbm_cylinder.py --quick    # short test run

Outputs (written to ./results):
    vorticity.png        - vorticity field with the vortex street
    velocity.png         - velocity magnitude
    probe_signal.png     - transverse velocity behind the cylinder + spectrum
    vortex_street.gif    - animation of the wake (needs Pillow)
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------------
# 1. Physical / lattice parameters (everything in lattice units)
# ---------------------------------------------------------------------------
QUICK = "--quick" in sys.argv

NX, NY = 400, 100          # lattice size (length x height)
D = 20                     # cylinder diameter in lattice cells
CX, CY = NX // 5, NY // 2  # cylinder centre
U_IN = 0.1                 # inflow velocity (keep << speed of sound 1/sqrt(3))
RE = 100.0                 # Reynolds number  Re = U_IN * D / nu

NU = U_IN * D / RE         # kinematic viscosity
TAU = 3.0 * NU + 0.5       # BGK relaxation time  (nu = c_s^2 (tau - 1/2))
OMEGA = 1.0 / TAU

N_STEPS = 6000 if QUICK else 30000
SAVE_EVERY = 200           # frames for the animation
PROBE = (CX + 3 * D, CY)   # probe point in the wake (x, y)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. D2Q9 lattice
#
#   6   2   5
#     \ | /
#   3 - 0 - 1
#     / | \
#   7   4   8
# ---------------------------------------------------------------------------
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]])          # velocities c_i
W = np.array([4/9] + [1/9] * 4 + [1/36] * 4)                 # weights w_i
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])                  # opposite direction


def equilibrium(rho, ux, uy):
    """Second-order Maxwell-Boltzmann equilibrium f_i^eq(rho, u).

    f_i^eq = w_i rho [1 + 3 (c_i.u) + 9/2 (c_i.u)^2 - 3/2 |u|^2]
    """
    cu = C[:, 0, None, None] * ux + C[:, 1, None, None] * uy   # shape (9, NY, NX)
    usq = ux**2 + uy**2
    return W[:, None, None] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq)


def macroscopic(f):
    """Density and velocity are the zeroth and first moments of f."""
    rho = f.sum(axis=0)
    ux = (f * C[:, 0, None, None]).sum(axis=0) / rho
    uy = (f * C[:, 1, None, None]).sum(axis=0) / rho
    return rho, ux, uy


def vorticity(ux, uy):
    """omega_z = dv/dx - du/dy  (central differences, lattice spacing = 1)."""
    return (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / 2 \
         - (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2


# ---------------------------------------------------------------------------
# 3. Geometry and initial condition
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[0:NY, 0:NX]
solid = (xx - CX)**2 + (yy - CY)**2 <= (D / 2)**2           # cylinder mask

rho0 = np.ones((NY, NX))
ux0 = U_IN * np.ones((NY, NX))
# A small one-signed "kick" of transverse velocity just behind the cylinder
# breaks the up/down symmetry, so that vortex shedding starts quickly
# (a perfectly symmetric start can stay steady for a very long time).
uy0 = 0.1 * U_IN * np.exp(-((xx - CX - D)**2 + (yy - CY)**2) / D**2)
ux0[solid] = 0.0; uy0[solid] = 0.0
f = equilibrium(rho0, ux0, uy0)

# Inflow populations (fixed): equilibrium with rho = 1, u = (U_IN, 0)
f_in = equilibrium(np.ones((NY, 1)), U_IN * np.ones((NY, 1)), np.zeros((NY, 1)))[:, :, 0]


# ---------------------------------------------------------------------------
# 4. Time loop:  collide -> stream -> boundary conditions
# ---------------------------------------------------------------------------
def run():
    global f
    probe_signal, frames = [], []
    print(f"Grid {NX}x{NY}, D = {D}, Re = {RE:.0f}, tau = {TAU:.4f}, steps = {N_STEPS}")
    t0 = time.time()

    for step in range(N_STEPS):
        # --- macroscopic quantities --------------------------------------
        rho, ux, uy = macroscopic(f)

        # --- collision (BGK): relax towards equilibrium ------------------
        feq = equilibrium(rho, ux, uy)
        f_post = f - OMEGA * (f - feq)

        # --- bounce-back on the cylinder (no-slip wall) -------------------
        # populations that hit a solid node are sent back where they came from
        f_post[:, solid] = f[OPP][:, solid]

        # --- streaming: move every population one cell along c_i ---------
        for i in range(9):
            f_post[i] = np.roll(f_post[i], shift=(C[i, 1], C[i, 0]), axis=(0, 1))
        f = f_post

        # --- inlet (x = 0): impose equilibrium inflow ---------------------
        f[:, :, 0] = f_in
        # --- outlet (x = NX-1): zero-gradient (copy neighbouring column) ---
        f[:, :, -1] = f[:, :, -2]
        # top/bottom are periodic automatically because np.roll wraps around

        # --- diagnostics --------------------------------------------------
        probe_signal.append(uy[PROBE[1], PROBE[0]])
        if step % SAVE_EVERY == 0:
            w = vorticity(ux, uy); w[solid] = np.nan
            frames.append(w)
        if step % 5000 == 0:
            print(f"  step {step:6d}   max|u| = {np.sqrt(ux**2 + uy**2).max():.4f}")

    print(f"Wall time: {time.time() - t0:.1f} s  "
          f"({N_STEPS * NX * NY / (time.time() - t0) / 1e6:.2f} MLUPS)")
    rho, ux, uy = macroscopic(f)
    return ux, uy, np.array(probe_signal), frames


# ---------------------------------------------------------------------------
# 5. Post-processing
# ---------------------------------------------------------------------------
def strouhal(signal):
    """Shedding frequency from the probe signal (second half only) -> St = f D / U.

    The frequency is measured from the average spacing of upward zero
    crossings (more precise than one FFT bin); the FFT is kept for plotting.
    """
    s = signal[len(signal) // 2:]
    s = s - s.mean()
    up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]          # upward zero crossings
    # linear interpolation of the exact crossing times
    t_cross = up + (-s[up]) / (s[up + 1] - s[up])
    period = np.diff(t_cross).mean()                       # time steps per cycle
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freq = np.fft.rfftfreq(len(s), d=1.0)                  # cycles per time step
    return D / (U_IN * period), freq, spec


def plot_results(ux, uy, signal, frames):
    ux = ux.copy(); uy = uy.copy()
    ux[solid] = np.nan; uy[solid] = np.nan
    w = vorticity(np.nan_to_num(ux), np.nan_to_num(uy)); w[solid] = np.nan
    wlim = 0.6 * np.nanmax(np.abs(w))

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(w, origin="lower", cmap="RdBu_r", vmin=-wlim, vmax=wlim)
    fig.colorbar(im, ax=ax, label="vorticity (lattice units)")
    ax.set(title=f"LBM D2Q9 - vortex street behind a cylinder, Re = {RE:.0f}",
           xlabel="x (lattice cells)", ylabel="y")
    ax.plot(*PROBE, "k+", ms=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "vorticity.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(np.sqrt(ux**2 + uy**2), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|u| (lattice units)")
    ax.set(title="Velocity magnitude", xlabel="x (lattice cells)", ylabel="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "velocity.png"), dpi=150)
    plt.close(fig)

    st, freq, spec = strouhal(signal)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].plot(signal, lw=0.8)
    axs[0].set(xlabel="time step", ylabel="u_y at probe", title="Wake probe signal")
    axs[1].plot(freq * D / U_IN, spec / spec.max())
    axs[1].axvline(st, color="r", ls="--", label=f"St = {st:.3f}")
    axs[1].set(xlim=(0, 1), xlabel="Strouhal number f D / U", ylabel="normalised amplitude",
               title="Spectrum (second half of the run)")
    axs[1].legend()
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "probe_signal.png"), dpi=150)
    plt.close(fig)
    print(f"Strouhal number St = {st:.3f}   (unconfined cylinder, Williamson 1996: ~0.164 at Re=100)")

    # animation of the last frames
    try:
        fig, ax = plt.subplots(figsize=(8, 2.3))
        im = ax.imshow(frames[0], origin="lower", cmap="RdBu_r", vmin=-wlim, vmax=wlim)
        ax.axis("off"); fig.tight_layout()
        sel = frames[-60:]
        ani = animation.FuncAnimation(fig, lambda k: [im.set_data(sel[k])],
                                      frames=len(sel), interval=80)
        ani.save(os.path.join(OUT_DIR, "vortex_street.gif"),
                 writer=animation.PillowWriter(fps=12))
        plt.close(fig)
    except Exception as exc:  # Pillow missing etc.
        print(f"(GIF skipped: {exc})")


if __name__ == "__main__":
    ux, uy, signal, frames = run()
    plot_results(ux, uy, signal, frames)
    print(f"Figures saved in {OUT_DIR}")
