"""Forced SO(2) Acrobot simulation with sinusoidal elbow torque.

Purpose
-------
Validate the forced maximal-coordinate LGVI against the standard continuous
Acrobot equations integrated with RK4 in relative coordinates.

The input is the classical Acrobot elbow torque

    u(t) = U0 sin(omega_u t),

which enters the relative-coordinate model as B u with B=[0,1]^T and enters the
absolute-link virtual work as tau=[-u,+u]^T.

The plotting structure mirrors the 3D pendulum numerical simulation script:
trajectory/components, input, velocities, RK4-vs-LGVI errors, constraint drift,
energy, and solver diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in [str(REPO_ROOT), str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Acrobot.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
    AcrobotSO2Model,
    AcrobotSO2Params,
)
from Acrobot.solver_rk4_acrobot import simulate_rk4_acrobot_relative
from Acrobot.solver_lgvi_acrobot import simulate_lgvi_acrobot, diagnostics_lgvi


# ---------------------------------------------------------------------------
# User choices
# ---------------------------------------------------------------------------
H = 0.001
TF = 100.0

# Sinusoidal elbow torque input u(t) = U0 sin(omega_u t).
U0 = 0.15
OMEGA_U = 0.7

# Initial condition in standard Acrobot coordinates.
# q = [theta1, theta2], where theta2 is the relative elbow angle.
# This choice gives alpha1 = alpha2 = 0 and hence R1_0 = R2_0 = I.
Q0 = np.array([0.5 * np.pi, 0.0], dtype=float)
# qdot = [theta1dot, theta2dot]; this gives omega1_0 = omega2_0 = 4.14 rad/s.
QDOT0 = np.array([4.14, 0.0], dtype=float)

# LGVI nonlinear solve settings.
ROOT_TOL = 1e-10
MAXFEV = 150
VERBOSE_LGVI = True
SAVE_PNG_ALSO = True

TRAJ_FIGSIZE = (7.0, 6.0)
STACK_FIGSIZE = (9.0, 8.0)
BENCHMARK_FIGSIZE = (8.0, 5.5)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def u_fun(t: float) -> float:
    """Classical Acrobot scalar elbow torque."""
    return float (U0 * np.sin(OMEGA_U * t))


def make_output_dir() -> Path:
    out = THIS_DIR / "output" / "acrobot_so2_forced_sine"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_plot_style() -> None:
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


def savefig(out: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(out / name)
    if SAVE_PNG_ALSO:
        plt.savefig((out / name).with_suffix(".png"), dpi=200)
    plt.close()


def angle_unwrap(a: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(a, dtype=float), axis=0)


def compute_lgvi_relative_quantities(model: AcrobotSO2Model, lgvi: dict, diag: dict, h: float) -> dict:
    """Convert LGVI absolute SO(2) trajectory to relative Acrobot variables."""
    alpha_abs = angle_unwrap(diag["alpha"])
    q = np.zeros_like(alpha_abs)
    q[:, 0] = alpha_abs[:, 0] + 0.5 * np.pi
    q[:, 1] = alpha_abs[:, 1] - alpha_abs[:, 0]

    # Absolute angular velocity over intervals from F_k.
    omega_abs = diag["omega"]
    qdot = np.zeros((omega_abs.shape[0], 2))
    qdot[:, 0] = omega_abs[:, 0]
    qdot[:, 1] = omega_abs[:, 1] - omega_abs[:, 0]

    return {
        "q": q,
        "qdot": qdot,
        "alpha_abs": alpha_abs,
        "X": lgvi["X"],
        "R1": lgvi["R1"],
        "R2": lgvi["R2"],
    }


def compute_control_history(t: np.ndarray) -> np.ndarray:
    return np.array([u_fun(ti) for ti in t], dtype=float)


def add_constraint_norms_if_missing(model: AcrobotSO2Model, data: dict) -> dict:
    """Ensure RK4 dictionary contains phi0_norm, phi12_norm, and phi_norm."""
    if "phi_norm" in data and "phi0_norm" in data and "phi12_norm" in data:
        return data

    X = data["X"]
    R1 = data["R1"]
    R2 = data["R2"]
    n = X.shape[0]
    phi0_norm = np.zeros(n)
    phi12_norm = np.zeros(n)
    phi_norm = np.zeros(n)

    for k in range(n):
        phi = model.constraints(X[k], R1[k], R2[k])
        phi0_norm[k] = np.linalg.norm(phi[:2])
        phi12_norm[k] = np.linalg.norm(phi[2:])
        phi_norm[k] = np.linalg.norm(phi)

    data["phi0_norm"] = phi0_norm
    data["phi12_norm"] = phi12_norm
    data["phi_norm"] = phi_norm
    return data


def compute_forced_power_reference(rk4: dict, t: np.ndarray) -> np.ndarray:
    """Compute continuous power input u * theta2dot for relative RK4."""
    u = compute_control_history(t)
    theta2dot = rk4["qdot"][:, 1]
    return u * theta2dot


def compute_position_error_norm(X_ref: np.ndarray, X_cmp: np.ndarray) -> np.ndarray:
    """Return the combined COM-position error norm used in the unforced case."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_cmp = np.asarray(X_cmp, dtype=float)
    x_diff = X_ref - X_cmp
    return np.sqrt(
        np.sum(x_diff[:, :2] ** 2, axis=1) +
        np.sum(x_diff[:, 2:] ** 2, axis=1)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(model: AcrobotSO2Model, lgvi: dict, diag: dict, rk4: dict, out: Path, h: float) -> None:
    t_nodes = lgvi["t"]
    t_int = t_nodes[:-1]
    lgvi_rel = compute_lgvi_relative_quantities(model, lgvi, diag, h)

    q_lgvi = lgvi_rel["q"]
    q_rk4 = angle_unwrap(rk4["q"])

    qdot_lgvi = lgvi_rel["qdot"]
    qdot_rk4_int = rk4["qdot"][:-1]

    u_nodes = compute_control_history(t_nodes)

    # ============================================================
    # 1) COM trajectory comparison in the plane
    # ============================================================
    X_lgvi = lgvi["X"]
    X_rk4 = rk4["X"]

    plt.figure(figsize=TRAJ_FIGSIZE)
    plt.plot(X_lgvi[:, 0], X_lgvi[:, 1], label=r"LGVI $x_1$")
    plt.plot(X_rk4[:, 0], X_rk4[:, 1], "--", label=r"RK4 $x_1$")
    plt.plot(X_lgvi[:, 2], X_lgvi[:, 3], label=r"LGVI $x_2$")
    plt.plot(X_rk4[:, 2], X_rk4[:, 3], "--", label=r"RK4 $x_2$")
    plt.scatter(model.p0[0], model.p0[1], c="k", s=60, label="base")
    plt.xlabel(r"$x\;[\mathrm{m}]$")
    plt.ylabel(r"$y\;[\mathrm{m}]$")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    savefig(out, "forced_sine_com_trajectory.pdf")

    # ============================================================
    # 2) Relative angle components
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=STACK_FIGSIZE, sharex=True)
    names = [r"$\theta_1\;[\mathrm{rad}]$", r"$\theta_2\;[\mathrm{rad}]$"]
    for j, ax in enumerate(axes):
        ax.plot(t_nodes, q_lgvi[:, j], label="LGVI")
        ax.plot(t_nodes, q_rk4[:, j], "--", label="RK4")
        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r"time $t_k$ [s]")
    axes[0].legend()
    savefig(out, "forced_sine_angles.pdf")

    # ============================================================
    # 3) Input torque
    # ============================================================
    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.plot(t_nodes, u_nodes, label=rf"$u(t)={U0}\sin({OMEGA_U}t)$")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$u\;[\mathrm{Nm}]$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    savefig(out, "forced_sine_input.pdf")

    # ============================================================
    # 4) Relative angular velocities
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=STACK_FIGSIZE, sharex=True)
    names = [r"$\Omega_1\;[\mathrm{rad/s}]$", r"$\Omega_2\;[\mathrm{rad/s}]$"]
    for j, ax in enumerate(axes):
        ax.plot(t_int, qdot_lgvi[:, j], label="LGVI")
        ax.plot(t_int, qdot_rk4_int[:, j], "--", label="RK4")
        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r"time $t_k$ [s]")
    axes[0].legend()
    savefig(out, "forced_sine_angular_velocities.pdf")

    # ============================================================
    # 5) Trajectory difference relative to RK4
    # ============================================================

    q_err_rad = q_lgvi - q_rk4
    q_err_deg_components = np.rad2deg(q_err_rad)

    q_err_norm_deg = np.linalg.norm(q_err_deg_components, axis=1)
    theta1_err_deg = np.abs(q_err_deg_components[:, 0])
    theta2_err_deg = np.abs(q_err_deg_components[:, 1])

    X_err = compute_position_error_norm(X_lgvi, X_rk4)

    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.plot(t_nodes, q_err_norm_deg, label=r"$\|q_{LGVI}-q_{RK4}\|$ [deg]")
    plt.plot(t_nodes, theta1_err_deg, "--", label=r"$|\theta_{1,LGVI}-\theta_{1,RK4}|$ [deg]")
    plt.plot(t_nodes, theta2_err_deg, "--", label=r"$|\theta_{2,LGVI}-\theta_{2,RK4}|$ [deg]")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$\mathrm{angle\ error}\;[\mathrm{deg}]$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    savefig(out, "forced_sine_angle_error_degrees.pdf")

    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.plot(
        t_nodes,
        X_err,
        "--",
        label=r"$\|x_{\mathrm{RK4}} - x_{\mathrm{LGVI}}\|$",
    )
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$\|\Delta x_k\|\;[\mathrm{m}]$")
    plt.grid(True, alpha=0.3)
    savefig(out, "forced_sine_position_error.pdf")

    # ============================================================
    # 6) SO(2) orthogonality errors
    # ============================================================
    eps = 1e-16
    n_common_orth = min(len(diag["orth_R1"]), len(rk4["orth_R1"]), len(rk4["orth_R2"]))
    t_orth = t_nodes[:n_common_orth]
    rk4_orth = np.maximum(rk4["orth_R1"][:n_common_orth], rk4["orth_R2"][:n_common_orth])

    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.semilogy(t_orth, np.maximum(diag["orth_R1"][:n_common_orth], eps), label="LGVI link 1")
    plt.semilogy(t_orth, np.maximum(diag["orth_R2"][:n_common_orth], eps), "--", label="LGVI link 2")
    plt.semilogy(t_orth, np.maximum(rk4_orth, eps), ":", label="RK4")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$e_{\mathrm{orth},i,k}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    savefig(out, "forced_sine_orthogonality_error.pdf")

    # ============================================================
    # 7) Holonomic constraint residuals
    # ============================================================
    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.semilogy(t_nodes, np.maximum(diag["phi0_norm"], eps), label=r"$\|\phi_{0,k}\|$ base constraint")
    plt.semilogy(t_nodes, np.maximum(diag["phi12_norm"], eps), "--", label=r"$\|\phi_{12,k}\|$ elbow constraint")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$\|\phi_k\|\;[\mathrm{m}]$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    savefig(out, "forced_sine_constraint_residuals.pdf")

    # ============================================================
    # 8) Total energy under forcing
    # ============================================================
    n_energy = min(len(diag["energy"]), len(rk4["energy"]) - 1)
    plt.figure(figsize=BENCHMARK_FIGSIZE)
    plt.plot(t_int[:n_energy], diag["energy"][:n_energy], label="LGVI")
    plt.plot(t_int[:n_energy], rk4["energy"][:n_energy], "--", label="RK4")
    plt.xlabel(r"time $t_k$ [s]")
    plt.ylabel(r"$E_k\;[\mathrm{J}]$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out, "forced_sine_total_energy.pdf")

    # ============================================================
    # 9) Applied power for RK4 reference
    # ============================================================
    #power = compute_forced_power_reference(rk4, t_nodes)
    #plt.figure(figsize=(8.8, 4.8))
    #plt.plot(t_nodes, power, label=r"$u\dot\theta_2$ RK4")
    #plt.xlabel(r"time $t_k$ [s]")
    #plt.ylabel("power [W]")
    #plt.title("Input power reference")
    #plt.grid(True, alpha=0.3)
    #plt.legend()
    #savefig(out, "forced_sine_input_power.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    print("=== Forced SO(2) Acrobot sine-input validation ===")
    print(f"h={h}, tf={tf}, steps={steps}")
    print(f"u(t) = {U0} * sin({OMEGA_U} * t)")

    print("Running RK4-relative reference...")
    rk4 = simulate_rk4_acrobot_relative(
        model=model,
        h=h,
        steps=steps,
        q0=Q0,
        qdot0=QDOT0,
        u_fun=u_fun,
    )
    rk4 = add_constraint_norms_if_missing(model, rk4)

    print("Running forced maximal-coordinate LGVI...")
    alpha0 = model.absolute_angles_from_relative(Q0[0], Q0[1])
    omega_abs0 = np.array([QDOT0[0], QDOT0[0] + QDOT0[1]], dtype=float)
    lgvi = simulate_lgvi_acrobot(
        model=model,
        h=h,
        steps=steps,
        alpha0=alpha0,
        omega0=omega_abs0,
        u_fun=u_fun,
        first_step="rk4",
        root_tol=ROOT_TOL,
        maxfev=MAXFEV,
        verbose=VERBOSE_LGVI,
    )
    diag = diagnostics_lgvi(model, lgvi)

    make_plots(model, lgvi, diag, rk4, out, h)

    lgvi_rel = compute_lgvi_relative_quantities(model, lgvi, diag, h)
    q_lgvi = lgvi_rel["q"]
    q_rk4 = angle_unwrap(rk4["q"])
    q_err = np.linalg.norm(q_lgvi - q_rk4, axis=1)
    X_err = np.linalg.norm(lgvi["X"] - rk4["X"], axis=1)

    print("\n=== Diagnostics ===")
    print(f"max |q_LGVI - q_RK4|:       {np.max(q_err):.3e}")
    print(f"final |q_LGVI - q_RK4|:     {q_err[-1]:.3e}")
    print(f"max |X_LGVI - X_RK4|:       {np.max(X_err):.3e}")
    print(f"max LGVI constraint norm:   {np.nanmax(diag['phi_norm']):.3e}")
    print(f"max LGVI root residual:     {np.nanmax(lgvi['residual_inf']):.3e}")
    print(f"max RK4 constraint norm:    {np.nanmax(rk4['phi_norm']):.3e}")
    print(f"max |Delta E| LGVI proxy:   {np.nanmax(np.abs(diag['energy_error'])):.3e}")
    print(f"max |Delta E| RK4-relative: {np.nanmax(np.abs(rk4['energy_error'])):.3e}")
    print(f"Output directory:           {out}")

    np.savez(
        out / "acrobot_so2_forced_sine_results.npz",
        t=lgvi["t"],
        q0=Q0,
        qdot0=QDOT0,
        u=np.array([u_fun(ti) for ti in lgvi["t"]]),
        lgvi_X=lgvi["X"],
        lgvi_R1=lgvi["R1"],
        lgvi_R2=lgvi["R2"],
        lgvi_q=q_lgvi,
        lgvi_qdot=lgvi_rel["qdot"],
        lgvi_phi_norm=diag["phi_norm"],
        lgvi_energy_error=diag["energy_error"],
        lgvi_residual_inf=lgvi["residual_inf"],
        rk4_X=rk4["X"],
        rk4_q=q_rk4,
        rk4_qdot=rk4["qdot"],
        rk4_phi_norm=rk4["phi_norm"],
        rk4_energy_error=rk4["energy_error"],
    )


if __name__ == "__main__":
    main()
