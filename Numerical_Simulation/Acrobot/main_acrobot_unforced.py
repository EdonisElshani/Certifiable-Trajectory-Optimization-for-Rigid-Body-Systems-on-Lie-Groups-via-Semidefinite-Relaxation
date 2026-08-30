"""Unforced SO(2) Acrobot numerical simulation.

This main script compares constraint drift for three trajectories:

1. LGVI in maximal coordinates
   - constraints are enforced at the position level by the implicit DEL solve.

2. RK4 in relative/minimal coordinates
   - constraints are satisfied by reconstruction from q=[theta1, theta2].

3. RK4 in maximal coordinates with acceleration-level constraints
   - Lagrange multipliers enforce ddot(phi)=0, but phi=0 is not projected.
   - this is the method that can show joint drift over long horizons.

The old matrix-RK4 variables were removed from this script, because they were
referenced without being computed. Tiny bug, huge runtime tantrum. Classic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in [str(REPO_ROOT), str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Acrobot.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (  # noqa: E402
    AcrobotSO2Model,
    AcrobotSO2Params,
)
from Acrobot.solver_rk4_acrobot import (  # noqa: E402
    simulate_rk4_acrobot_relative,
    simulate_rk4_acrobot_maximal_accel,
)
from Acrobot.solver_lgvi_acrobot import (  # noqa: E402
    simulate_lgvi_acrobot,
    diagnostics_lgvi,
)


# ---------------------------------------------------------------------------
# User choices
# ---------------------------------------------------------------------------
# h=0.15 may be too large for the implicit LGVI solve. For reliable validation,
# start with H=0.05, then try H=0.1. Yes, numerical solvers have trust issues.
H = 0.01
TF = 1000.0

RUN_LGVI = True
LGVI_TF: Optional[float] = TF     # set e.g. 200.0 for a shorter LGVI run

SAVE_PNG_ALSO = True
EPS = 1e-16


def make_output_dir() -> Path:
    out = THIS_DIR / "output" / "acrobot_so2_unforced"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_plot_style() -> None:
    """Match the visual style used in the 3D pendulum plotting code."""
    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.8,
        "ytick.major.width": 2.8,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "lines.linewidth": 3.2,
        "lines.markersize": 11,
        "legend.frameon": True,
    })


def save_current_figure(outpath_pdf: Path) -> None:
    """Save the current matplotlib figure as PDF and optionally PNG."""
    plt.tight_layout()
    plt.savefig(outpath_pdf)
    if SAVE_PNG_ALSO:
        plt.savefig(outpath_pdf.with_suffix(".png"), dpi=200)
    plt.close()


def compute_constraint_norms(
    model: AcrobotSO2Model,
    sim: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute constraint norms from X, R1, R2 and store them in sim.

    This makes the main script robust even if a solver forgot to return phi_norm.
    Because trusting every dictionary key is how humans invented debugging.
    """
    X = np.asarray(sim["X"])
    R1 = np.asarray(sim["R1"])
    R2 = np.asarray(sim["R2"])

    n = X.shape[0]
    phi0_norm = np.zeros(n)
    phi12_norm = np.zeros(n)
    phi_norm = np.zeros(n)

    for k in range(n):
        phi = model.constraints(X[k], R1[k], R2[k])
        phi0_norm[k] = np.linalg.norm(phi[:2])
        phi12_norm[k] = np.linalg.norm(phi[2:])
        phi_norm[k] = np.linalg.norm(phi)

    sim["phi0_norm"] = phi0_norm
    sim["phi12_norm"] = phi12_norm
    sim["phi_norm"] = phi_norm
    return sim


def print_constraint_summary(name: str, sim: Dict[str, np.ndarray]) -> None:
    print(f"{name:38s} max ||phi||:   {np.nanmax(sim['phi_norm']):.3e}")
    print(f"{name:38s} max ||phi0||:  {np.nanmax(sim['phi0_norm']):.3e}")
    print(f"{name:38s} max ||phi12||: {np.nanmax(sim['phi12_norm']):.3e}")


def plot_position_error_norm(
    out: Path,
    rk4_relative: Dict[str, np.ndarray],
    lgvi: Dict[str, np.ndarray],
) -> None:
    """Plot the norm of the LGVI-vs-RK4 maximal-coordinate position error."""
    n_common = min(len(lgvi["X"]), len(rk4_relative["X"]))
    t_nodes = lgvi["t"][:n_common]

    x_diff = lgvi["X"][:n_common] - rk4_relative["X"][:n_common]
    pos_err = np.sqrt(
        np.sum(x_diff[:, :2] ** 2, axis=1) +
        np.sum(x_diff[:, 2:] ** 2, axis=1)
    )

    plt.figure(figsize=(8.0, 5.5))
    plt.plot(
        t_nodes,
        pos_err,
        "--",
        label=r"$\|x_{\mathrm{RK4}} - x_{\mathrm{LGVI}}\|$",
    )
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$\|\Delta x_k\|\;[\mathrm{m}]$")
    plt.ylim(-1.0, 4.0)
    plt.grid(True, alpha=0.3)
    save_current_figure(out / "acrobot_position_error_norm.pdf")


def plot_total_energy(
    out: Path,
    rk4_relative: Dict[str, np.ndarray],
    lgvi: Dict[str, np.ndarray],
    lgvi_diag: Dict[str, np.ndarray],
) -> None:
    """Plot total mechanical energy for LGVI and RK4 over time."""
    n_lgvi = len(lgvi_diag["energy"])
    n_rk4 = min(len(rk4_relative["energy"]) - 1, n_lgvi)
    t_lgvi = lgvi["t"][:n_lgvi]
    t_rk4 = rk4_relative["t"][:n_rk4]

    plt.figure(figsize=(8.0, 5.5))
    plt.plot(t_lgvi, lgvi_diag["energy"], label="LGVI")
    plt.plot(t_rk4, rk4_relative["energy"][:n_rk4], "--", label="RK4")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$E_k\;[\mathrm{J}]$")
    plt.ylim(19.0, 26.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    save_current_figure(out / "acrobot_total_energy.pdf")


def plot_orthogonality_error(
    out: Path,
    rk4_relative: Dict[str, np.ndarray],
    lgvi: Dict[str, np.ndarray],
    lgvi_diag: Dict[str, np.ndarray],
) -> None:
    """Plot SO(2) orthogonality errors for both LGVI link rotations."""
    n_common = min(
        len(lgvi_diag["orth_R1"]),
        len(rk4_relative["orth_R1"]),
        len(rk4_relative["orth_R2"]),
    )
    t_nodes = lgvi["t"][:n_common]
    rk4_orth = np.maximum(
        rk4_relative["orth_R1"][:n_common],
        rk4_relative["orth_R2"][:n_common],
    )

    plt.figure(figsize=(8.0, 5.5))
    plt.semilogy(t_nodes, np.maximum(lgvi_diag["orth_R1"][:n_common], EPS), label="LGVI link 1")
    plt.semilogy(t_nodes, np.maximum(lgvi_diag["orth_R2"][:n_common], EPS), "--", label="LGVI link 2")
    plt.semilogy(t_nodes, np.maximum(rk4_orth, EPS), ":", label="RK4")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$e_{\mathrm{orth},i,k}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    save_current_figure(out / "acrobot_orthogonality_error.pdf")


def plot_holonomic_constraint_residuals(
    out: Path,
    lgvi: Dict[str, np.ndarray],
    lgvi_diag: Dict[str, np.ndarray],
) -> None:
    """Plot the LGVI holonomic constraint residuals only.

    RK4 in minimal relative coordinates satisfies the holonomic constraints by
    construction through the angle parametrization, so including it here would
    not be a meaningful apples-to-apples comparison.
    """
    t_nodes = lgvi["t"][:len(lgvi_diag["phi0_norm"])]

    plt.figure(figsize=(8.0, 5.5))
    plt.semilogy(t_nodes, np.maximum(lgvi_diag["phi0_norm"], EPS), label=r"$\|\phi_{0,k}\|$ base constraint")
    plt.semilogy(t_nodes, np.maximum(lgvi_diag["phi12_norm"], EPS), "--", label=r"$\|\phi_{12,k}\|$ elbow constraint")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$\|\phi_k\|\;[\mathrm{m}]$")
    plt.ylim(top=1e-12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    save_current_figure(out / "acrobot_holonomic_constraint_residuals.pdf")


def main() -> None:
    set_plot_style()
    out = make_output_dir()

    params = AcrobotSO2Params(
        m1=1.0,
        m2=1.0,
        l1=1.0,
        l2=1.0,
        lc1=0.5,
        lc2=0.5,
        J1=1.0 / 12.0,
        J2=1.0 / 12.0,
        g=9.81,
        p0=(0.0, 0.0),
    )
    model = AcrobotSO2Model(params)

    h = float(H)
    tf = float(TF)
    steps = int(round(tf / h))

    # Standard Acrobot relative coordinates:
    # q = [theta1, theta2], zero = both links straight down.
    # This choice gives alpha1 = alpha2 = 0 and hence R1_0 = R2_0 = I.
    q0 = np.array([0.5 * np.pi, 0.0], dtype=float)
    # qdot = [theta1dot, theta2dot]; this gives omega1_0 = omega2_0 = 4.14 rad/s.
    qdot0 = np.array([4.14, 0.0], dtype=float)
    u_fun = None  # unforced

    print(f"h={h}, tf={tf}, steps={steps}")
    if h > 0.1:
        print("Warning: h > 0.1 can make the LGVI nonlinear solve fragile.")

    print("Running RK4-relative reference...")
    rk4_relative = simulate_rk4_acrobot_relative(model, h, steps, q0, qdot0, u_fun=u_fun)
    rk4_relative = compute_constraint_norms(model, rk4_relative)

    print("Running RK4-maximal with acceleration-level constraints...")
    rk4_maximal_accel = simulate_rk4_acrobot_maximal_accel(model, h, steps, q0, qdot0, u_fun=u_fun)
    rk4_maximal_accel = compute_constraint_norms(model, rk4_maximal_accel)

    lgvi = None
    diag = None
    t_lgvi = None

    if RUN_LGVI:
        lgvi_tf = tf if LGVI_TF is None else min(float(LGVI_TF), tf)
        lgvi_steps = int(round(lgvi_tf / h))
        print(f"Running LGVI for tf={lgvi_tf}, steps={lgvi_steps}.")

        alpha0 = model.absolute_angles_from_relative(q0[0], q0[1])
        omega_abs0 = np.array([qdot0[0], qdot0[0] + qdot0[1]], dtype=float)

        lgvi = simulate_lgvi_acrobot(
            model=model,
            h=h,
            steps=lgvi_steps,
            alpha0=alpha0,
            omega0=omega_abs0,
            u_fun=u_fun,
            first_step="rk4",
            root_tol=1e-10,
            maxfev=100,
            verbose=True,
        )
        diag = diagnostics_lgvi(model, lgvi)
        t_lgvi = lgvi["t"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print("\n=== Constraint diagnostics ===")
    print_constraint_summary("relative RK4", rk4_relative)
    print_constraint_summary("maximal RK4 accel-level", rk4_maximal_accel)

    print("\n=== Energy diagnostics ===")
    print(f"relative RK4 max |Delta E|:           {np.nanmax(np.abs(rk4_relative['energy_error'])):.3e}")
    print(f"maximal RK4 accel max |Delta E|:      {np.nanmax(np.abs(rk4_maximal_accel['energy_error'])):.3e}")

    if diag is not None and lgvi is not None:
        print("\n=== LGVI diagnostics ===")
        print(f"LGVI max ||phi||:                     {np.nanmax(diag['phi_norm']):.3e}")
        print(f"LGVI max orth R1:                     {np.nanmax(diag['orth_R1']):.3e}")
        print(f"LGVI max orth R2:                     {np.nanmax(diag['orth_R2']):.3e}")
        print(f"LGVI max root residual:               {np.nanmax(lgvi['residual_inf']):.3e}")
        print(f"LGVI max |Delta E| proxy:             {np.nanmax(np.abs(diag['energy_error'])):.3e}")

    print(f"Output directory:                     {out}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if lgvi is not None and diag is not None:
        plot_position_error_norm(out, rk4_relative, lgvi)
        plot_total_energy(out, rk4_relative, lgvi, diag)
        plot_orthogonality_error(out, rk4_relative, lgvi, diag)
        plot_holonomic_constraint_residuals(out, lgvi, diag)

        print(f"Saved: {out / 'acrobot_position_error_norm.pdf'}")
        print(f"Saved: {out / 'acrobot_total_energy.pdf'}")
        print(f"Saved: {out / 'acrobot_orthogonality_error.pdf'}")
        print(f"Saved: {out / 'acrobot_holonomic_constraint_residuals.pdf'}")

    # ------------------------------------------------------------------
    # Save numeric results
    # ------------------------------------------------------------------
    np.savez(
        out / "acrobot_so2_unforced_results.npz",
        h=h,
        tf=tf,
        q0=q0,
        qdot0=qdot0,
        rk4_relative_t=rk4_relative["t"],
        rk4_relative_q=rk4_relative["q"],
        rk4_relative_qdot=rk4_relative["qdot"],
        rk4_relative_energy=rk4_relative["energy"],
        rk4_relative_energy_error=rk4_relative["energy_error"],
        rk4_relative_phi_norm=rk4_relative["phi_norm"],
        rk4_maximal_t=rk4_maximal_accel["t"],
        rk4_maximal_q=rk4_maximal_accel["q"],
        rk4_maximal_qdot=rk4_maximal_accel["qdot"],
        rk4_maximal_energy=rk4_maximal_accel["energy"],
        rk4_maximal_energy_error=rk4_maximal_accel["energy_error"],
        rk4_maximal_phi_norm=rk4_maximal_accel["phi_norm"],
        lgvi_t=np.array([]) if lgvi is None else lgvi["t"],
        lgvi_X=np.array([]) if lgvi is None else lgvi["X"],
        lgvi_R1=np.array([]) if lgvi is None else lgvi["R1"],
        lgvi_R2=np.array([]) if lgvi is None else lgvi["R2"],
        lgvi_phi_norm=np.array([]) if diag is None else diag["phi_norm"],
        lgvi_energy_error=np.array([]) if diag is None else diag["energy_error"],
    )


if __name__ == "__main__":
    main()
