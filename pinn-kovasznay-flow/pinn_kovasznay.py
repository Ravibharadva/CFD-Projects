"""
Project 3 - Physics-Informed Neural Network (PINN) for Kovasznay Flow
=====================================================================

A small PyTorch PINN that solves the steady 2D incompressible Navier-Stokes
equations for Kovasznay flow (Re = 20).  Kovasznay flow has an exact analytical
solution, so we can measure exactly how accurate the network is.

Two modes are available:

  forward : the viscosity is known, the network learns u, v, p from the PDE
            and the boundary values only (no interior data).
  inverse : the viscosity is UNKNOWN. The network gets a few scattered velocity
            measurements and learns u, v, p AND the Reynolds number together.

Run:
    python pinn_kovasznay.py                 # forward problem
    python pinn_kovasznay.py --mode inverse  # inverse problem (find Re)
    python pinn_kovasznay.py --quick         # short test run

Outputs (written to ./results):
    <mode>_fields.png   - PINN vs exact solution vs absolute error
    <mode>_loss.png     - training loss history
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["forward", "inverse"], default="forward")
parser.add_argument("--quick", action="store_true", help="short run for testing")
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RE_TRUE = 20.0
LAM = RE_TRUE / 2 - np.sqrt(RE_TRUE**2 / 4 + 4 * np.pi**2)   # Kovasznay lambda
X_MIN, X_MAX, Y_MIN, Y_MAX = -0.5, 1.0, -0.5, 1.5            # domain

N_PDE = 2000          # collocation points inside the domain
N_BC = 400            # points on the boundary
N_DATA = 100          # velocity measurements (inverse mode only)
ADAM_ITERS = 1000 if args.quick else 6000
LBFGS_ITERS = 100 if args.quick else 1000

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Exact solution (Kovasznay, 1948)
# ---------------------------------------------------------------------------
def exact(x, y):
    u = 1 - np.exp(LAM * x) * np.cos(2 * np.pi * y)
    v = LAM / (2 * np.pi) * np.exp(LAM * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * LAM * x))
    return u, v, p


# ---------------------------------------------------------------------------
# 2. Training points
# ---------------------------------------------------------------------------
def to_tensor(a, grad=False):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE, requires_grad=grad)


# (a) collocation points: random points where the PDE residual is minimised
x_pde = np.random.uniform(X_MIN, X_MAX, (N_PDE, 1))
y_pde = np.random.uniform(Y_MIN, Y_MAX, (N_PDE, 1))

# (b) boundary points on the four edges, with exact u and v
n = N_BC // 4
xb = np.concatenate([np.full(n, X_MIN), np.full(n, X_MAX),
                     np.random.uniform(X_MIN, X_MAX, n), np.random.uniform(X_MIN, X_MAX, n)])
yb = np.concatenate([np.random.uniform(Y_MIN, Y_MAX, n), np.random.uniform(Y_MIN, Y_MAX, n),
                     np.full(n, Y_MIN), np.full(n, Y_MAX)])
ub, vb, _ = exact(xb, yb)

# (c) interior "measurements" (inverse problem only)
xd = np.random.uniform(X_MIN, X_MAX, N_DATA)
yd = np.random.uniform(Y_MIN, Y_MAX, N_DATA)
ud, vd, _ = exact(xd, yd)

X_pde, Y_pde = to_tensor(x_pde, True), to_tensor(y_pde, True)
X_bc, Y_bc = to_tensor(xb[:, None]), to_tensor(yb[:, None])
U_bc, V_bc = to_tensor(ub[:, None]), to_tensor(vb[:, None])
X_d, Y_d = to_tensor(xd[:, None]), to_tensor(yd[:, None])
U_d, V_d = to_tensor(ud[:, None]), to_tensor(vd[:, None])


# ---------------------------------------------------------------------------
# 3. Neural network  (x, y) -> (u, v, p)
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, width=50, depth=4):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

        # 1/Re: fixed in forward mode, trainable in inverse mode.
        # We train log(1/Re) so that it always stays positive. Start guess: Re = 5.
        init = np.log(1 / RE_TRUE) if args.mode == "forward" else np.log(1 / 5.0)
        self.log_inv_re = nn.Parameter(torch.tensor(init, dtype=torch.float32),
                                       requires_grad=(args.mode == "inverse"))

    def forward(self, x, y):
        # scale inputs to roughly [-1, 1] - helps tanh networks train
        xs = 2 * (x - X_MIN) / (X_MAX - X_MIN) - 1
        ys = 2 * (y - Y_MIN) / (Y_MAX - Y_MIN) - 1
        out = self.net(torch.cat([xs, ys], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    @property
    def re(self):
        return 1.0 / torch.exp(self.log_inv_re)


def grad(f, x):
    """df/dx with automatic differentiation (keeps the graph for 2nd derivatives)."""
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def pde_residual(model, x, y):
    """Residuals of the steady incompressible Navier-Stokes equations.

    momentum-x : u u_x + v u_y + p_x - (1/Re)(u_xx + u_yy) = 0
    momentum-y : u v_x + v v_y + p_y - (1/Re)(v_xx + v_yy) = 0
    continuity : u_x + v_y = 0
    """
    u, v, p = model(x, y)
    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)
    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)
    inv_re = torch.exp(model.log_inv_re)

    r_mx = u * u_x + v * u_y + p_x - inv_re * (u_xx + u_yy)
    r_my = u * v_x + v * v_y + p_y - inv_re * (v_xx + v_yy)
    r_c = u_x + v_y
    return r_mx, r_my, r_c


def loss_fn(model):
    r_mx, r_my, r_c = pde_residual(model, X_pde, Y_pde)
    l_pde = (r_mx**2).mean() + (r_my**2).mean() + (r_c**2).mean()

    ub_pred, vb_pred, _ = model(X_bc, Y_bc)
    l_bc = ((ub_pred - U_bc)**2).mean() + ((vb_pred - V_bc)**2).mean()

    l_data = torch.tensor(0.0, device=DEVICE)
    if args.mode == "inverse":
        ud_pred, vd_pred, _ = model(X_d, Y_d)
        l_data = ((ud_pred - U_d)**2).mean() + ((vd_pred - V_d)**2).mean()

    return l_pde + l_bc + l_data, (l_pde.item(), l_bc.item(), l_data.item())


# ---------------------------------------------------------------------------
# 4. Training: Adam first (robust), then L-BFGS (fast final convergence)
# ---------------------------------------------------------------------------
def train(model):
    history = {"total": [], "pde": [], "bc": [], "data": [], "re": []}

    def log(loss, parts):
        history["total"].append(loss.item())
        history["pde"].append(parts[0]); history["bc"].append(parts[1])
        history["data"].append(parts[2]); history["re"].append(model.re.item())

    print(f"Mode: {args.mode},  device: {DEVICE},  Adam {ADAM_ITERS} + L-BFGS {LBFGS_ITERS}")
    t0 = time.time()

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(adam, step_size=2000, gamma=0.5)
    for it in range(ADAM_ITERS):
        adam.zero_grad()
        loss, parts = loss_fn(model)
        loss.backward()
        adam.step(); sched.step()
        log(loss, parts)
        if it % 1000 == 0:
            print(f"  Adam   {it:5d}  loss = {loss.item():.3e}  Re = {model.re.item():.3f}")

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              history_size=50, tolerance_grad=1e-9,
                              tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, parts = loss_fn(model)
        loss.backward()
        log(loss, parts)
        return loss

    lbfgs.step(closure)
    loss, _ = loss_fn(model)
    print(f"  L-BFGS done   loss = {loss.item():.3e}  Re = {model.re.item():.3f}")
    print(f"Training time: {time.time() - t0:.1f} s")
    return history


# ---------------------------------------------------------------------------
# 5. Evaluation and plots
# ---------------------------------------------------------------------------
def evaluate(model, history):
    nx, ny = 151, 201
    xg, yg = np.meshgrid(np.linspace(X_MIN, X_MAX, nx), np.linspace(Y_MIN, Y_MAX, ny))
    with torch.no_grad():
        up, vp, pp = model(to_tensor(xg.reshape(-1, 1)), to_tensor(yg.reshape(-1, 1)))
    up, vp, pp = (a.cpu().numpy().reshape(ny, nx) for a in (up, vp, pp))
    ue, ve, pe = exact(xg, yg)
    # pressure is only defined up to a constant -> compare after removing the mean
    pp = pp - pp.mean() + pe.mean()

    def rel_l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    errs = {"u": rel_l2(up, ue), "v": rel_l2(vp, ve), "p": rel_l2(pp, pe)}
    print("Relative L2 errors: " + ", ".join(f"{k} = {v:.2e}" for k, v in errs.items()))
    if args.mode == "inverse":
        re_err = abs(model.re.item() - RE_TRUE) / RE_TRUE * 100
        print(f"Identified Re = {model.re.item():.3f}  (true {RE_TRUE}, error {re_err:.2f} %)")

    # fields: rows = u, v, p ; columns = PINN, exact, |error|
    fig, axs = plt.subplots(3, 3, figsize=(11, 11))
    for r, (name, pred, ex) in enumerate([("u", up, ue), ("v", vp, ve), ("p", pp, pe)]):
        vmin, vmax = ex.min(), ex.max()
        for c, (title, data, cmap) in enumerate([("PINN", pred, "jet"), ("Exact", ex, "jet"),
                                                  ("|error|", np.abs(pred - ex), "magma")]):
            kw = dict(vmin=vmin, vmax=vmax) if c < 2 else {}
            im = axs[r, c].contourf(xg, yg, data, levels=50, cmap=cmap, **kw)
            fig.colorbar(im, ax=axs[r, c], fraction=0.046)
            axs[r, c].set(title=f"{name}: {title}", aspect="equal")
    if args.mode == "inverse":
        axs[0, 0].plot(xd, yd, "k.", ms=3, label="measurements")
        axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"PINN for Kovasznay flow ({args.mode} problem), "
                 f"rel. L2 error u = {errs['u']:.1e}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{args.mode}_fields.png"), dpi=130)
    plt.close(fig)

    ncols = 2 if args.mode == "inverse" else 1
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
    ax = axs[0, 0]
    ax.semilogy(history["total"], label="total")
    ax.semilogy(history["pde"], label="PDE residual", alpha=0.7)
    ax.semilogy(history["bc"], label="boundary", alpha=0.7)
    if args.mode == "inverse":
        ax.semilogy(history["data"], label="data", alpha=0.7)
    ax.axvline(ADAM_ITERS, color="k", ls=":", lw=1)
    ax.text(ADAM_ITERS, ax.get_ylim()[1], " L-BFGS", va="top", fontsize=8)
    ax.set(xlabel="iteration", ylabel="loss", title="Training loss"); ax.legend(); ax.grid(alpha=0.3)
    if args.mode == "inverse":
        ax = axs[0, 1]
        ax.plot(history["re"], label="PINN estimate")
        ax.axhline(RE_TRUE, color="r", ls="--", label=f"true Re = {RE_TRUE:.0f}")
        ax.set(xlabel="iteration", ylabel="Re", title="Identified Reynolds number")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{args.mode}_loss.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    model = PINN().to(DEVICE)
    hist = train(model)
    evaluate(model, hist)
    print(f"Figures saved in {OUT_DIR}")
