from __future__ import annotations

import datetime
import io
import os
import pickle
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# Current file:
#   .../SPOT/Python-examples/SPOT_MPC_Acrobot/SDP/solve.py
THIS_FILE = Path(__file__).resolve()

# Project folder:
#   .../SPOT/Python-examples/SPOT_MPC_Acrobot
PROJECT_ROOT = THIS_FILE.parents[1]

# Outer SPOT folder:
#   .../SPOT
SPOT_OUTER_DIR = THIS_FILE.parents[3]

# SPOT Python package folder:
#   .../SPOT/SPOT/PYTHON
SPOT_PYTHON_DIR = SPOT_OUTER_DIR / "SPOT" / "PYTHON"

for path in [
    str(PROJECT_ROOT),      # for config, SDP, Numerical_Simulation
    str(SPOT_OUTER_DIR),    # for from SPOT.PYTHON...
    str(SPOT_PYTHON_DIR),   # for direct imports if needed
]:
    if path not in sys.path:
        sys.path.insert(0, path)

print(f"[solve.py] PROJECT_ROOT   = {PROJECT_ROOT}")
print(f"[solve.py] SPOT_OUTER_DIR = {SPOT_OUTER_DIR}")
print(f"[solve.py] SPOT_PYTHON_DIR = {SPOT_PYTHON_DIR}")


from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr, numpoly_visualize
from SPOT.PYTHON.naive_extract import naive_extract
from SPOT.PYTHON.robust_extract_CS import robust_extract_CS, ordered_extract_CS

from config.config_loader import load_yaml_config, build_common_params

from SDP.mapping import attach_mapping_to_params, get_remapped_ids
from SDP.constraints import (
    get_init_constraints,
    get_rotational_kinematics_link1,
    get_rotational_kinematics_link2,
    get_SO2_orthogonality_constraint_rotation_R,
    get_SO2_orthogonality_constraint_rotation_F,
    get_step_angle_bound_constraint_link_1,
    get_step_angle_bound_constraint_link_2,
    get_translational_dynamics_link1,
    get_translational_dynamics_link2,
    get_rotational_dynamics_link1,
    get_rotational_dynamics_link2,
    get_control_bounds,
    get_lambda_bounds,
)
from SDP.objective import build_objective
from SDP.cliques import get_cliques_for_cstss
from SDP.extraction import (
    extract_solution_variables,
    compute_SO2_errors,
    get_first_mpc_control,
)


class _TeeTextStream:
    """Mirror solver output to the terminal while retaining a local copy."""

    def __init__(self, terminal, capture: io.StringIO):
        self.terminal = terminal
        self.capture = capture

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()


def _parse_mosek_statuses(solver_output: str) -> tuple[str, str]:
    """Parse the final MOSEK solution summary emitted by the backend."""

    def last_status(label: str) -> str:
        matches = re.findall(
            rf"{label}\s*status\s*:\s*([A-Za-z0-9_-]+)",
            solver_output,
            flags=re.IGNORECASE,
        )
        if not matches:
            return "UNKNOWN"
        return matches[-1].replace("-", "_").upper()

    return last_status("Problem"), last_status("Solution")


def _status_text(x) -> str:
    return str(x).upper() if x is not None else ""


def is_infeasible_status(problem_status, solution_status) -> bool:
    statuses = [_status_text(problem_status), _status_text(solution_status)]
    return any("INFEASIBLE" in status for status in statuses)


def is_unknown_status(problem_status, solution_status) -> bool:
    statuses = [_status_text(problem_status), _status_text(solution_status)]
    return any("UNKNOWN" in status for status in statuses)


def is_finite_control(u) -> bool:
    try:
        return u is not None and np.isfinite(float(u))
    except Exception:
        return False


def _fixed_f0_step_bound_diagnostic(params: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnostic for the fixed previous-step rotations supplied to the SDP."""

    thetaF1_prev = float(np.arctan2(float(params["b1_prev"]), float(params["a1_prev"])))
    thetaF2_prev = float(np.arctan2(float(params["b2_prev"]), float(params["a2_prev"])))
    max_step_angle1 = float(params["max_step_angle1"])
    max_step_angle2 = float(params["max_step_angle2"])
    tol = 1e-12
    F1_warning = bool(abs(thetaF1_prev) > max_step_angle1 + tol)
    F2_warning = bool(abs(thetaF2_prev) > max_step_angle2 + tol)

    return {
        "thetaF1_prev_rad": thetaF1_prev,
        "thetaF2_prev_rad": thetaF2_prev,
        "thetaF1_prev_deg": float(np.rad2deg(thetaF1_prev)),
        "thetaF2_prev_deg": float(np.rad2deg(thetaF2_prev)),
        "thetaF1_current_sdp_F0_rad": thetaF1_prev,
        "thetaF1_current_sdp_F0_deg": float(np.rad2deg(thetaF1_prev)),
        "thetaF2_current_sdp_F0_rad": thetaF2_prev,
        "thetaF2_current_sdp_F0_deg": float(np.rad2deg(thetaF2_prev)),
        "max_step_angle1_rad": max_step_angle1,
        "max_step_angle2_rad": max_step_angle2,
        "max_step_angle1_deg": float(np.rad2deg(max_step_angle1)),
        "max_step_angle2_deg": float(np.rad2deg(max_step_angle2)),
        "max_step_angle_deg_link1": float(np.rad2deg(max_step_angle1)),
        "max_step_angle_deg_link2": float(np.rad2deg(max_step_angle2)),
        "F1_0_step_bound_warning": F1_warning,
        "F2_0_step_bound_warning": F2_warning,
        "F0_step_bound_warning": bool(F1_warning or F2_warning),
        "note": (
            "F0 is fixed from measured rotation history on the dt_sdp time "
            "scale and is not constrained by the future step-angle bound."
        ),
    }


def _write_fixed_f0_diagnostic(f, diag: Dict[str, Any]) -> None:
    """Write the fixed-F0 step-bound diagnostic in a readable format."""

    f.write("Fixed F0 used by current SDP iteration:\n")
    f.write(
        "  thetaF1_current_sdp_F0_rad = "
        f"{diag['thetaF1_current_sdp_F0_rad']:+.12e}\n"
    )
    f.write(
        "  thetaF1_current_sdp_F0_deg = "
        f"{diag['thetaF1_current_sdp_F0_deg']:+.12e}\n"
    )
    f.write(
        "  thetaF2_current_sdp_F0_rad = "
        f"{diag['thetaF2_current_sdp_F0_rad']:+.12e}\n"
    )
    f.write(
        "  thetaF2_current_sdp_F0_deg = "
        f"{diag['thetaF2_current_sdp_F0_deg']:+.12e}\n"
    )
    f.write(
        "  max_step_angle_deg_link1 = "
        f"{diag['max_step_angle_deg_link1']:+.12e}\n"
    )
    f.write(
        "  max_step_angle_deg_link2 = "
        f"{diag['max_step_angle_deg_link2']:+.12e}\n"
    )
    f.write(f"  F1_0_step_bound_warning = {diag['F1_0_step_bound_warning']}\n")
    f.write(f"  F2_0_step_bound_warning = {diag['F2_0_step_bound_warning']}\n")
    f.write(f"  F0_step_bound_warning = {diag['F0_step_bound_warning']}\n")
    f.write(f"  Note: {diag['note']}\n")


def complete_sdp_params(params: Dict[str, Any], mpc_initial: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Add SDP-specific derived parameters.

    YAML is still the single source of truth. This only derives:
        dt = dt_sdp
        rho vectors
        nonstandard SO(2) delta parameters
        desired c/s and F desired scalars from YAML thetaF values
        CSTSS defaults
        optional MPC initial boundary data
    """
    params = dict(params)

    # CSTSS defaults.
    params.setdefault("kappa", 2)
    params.setdefault("relax_mode", "SOS")
    params.setdefault("cs_mode", "SELF")
    params.setdefault("ts_mode", "NON")
    params.setdefault("ts_mom_mode", "NON")
    params.setdefault("ts_eq_mode", "NON")
    params.setdefault("if_solve", True)
    params.setdefault("if_mex", True)

    # SDP step uses dt_sdp.
    if "dt_sdp" in params:
        params["dt"] = float(params["dt_sdp"])
    elif "dt" in params:
        params["dt"] = float(params["dt"])
    else:
        raise KeyError("Missing dt_sdp or dt in params.")

    # Horizon.
    if "N" not in params:
        params["N"] = int(params.get("horizon_N", 5))

    # Geometry with thesis convention.
    l1 = float(params["l1"])
    l2 = float(params["l2"])
    params.setdefault("rho_10", np.array([0.0, 0.5 * l1], dtype=float))
    params.setdefault("rho_112", np.array([0.0, -0.5 * l1], dtype=float))
    params.setdefault("rho_212", np.array([0.0, 0.5 * l2], dtype=float))

    params["rho_10"] = np.asarray(params["rho_10"], dtype=float).reshape(2)
    params["rho_112"] = np.asarray(params["rho_112"], dtype=float).reshape(2)
    params["rho_212"] = np.asarray(params["rho_212"], dtype=float).reshape(2)

    if "p_0" not in params:
        if "p0" in params:
            params["p_0"] = np.asarray(params["p0"], dtype=float).reshape(2)
        else:
            params["p_0"] = np.array([0.0, 0.0], dtype=float)
    else:
        params["p_0"] = np.asarray(params["p_0"], dtype=float).reshape(2)

    # Nonstandard SO(2) inertia parameters.
    if "deltaJ1" not in params:
        if "Jd1" in params:
            params["deltaJ1"] = float(np.asarray(params["Jd1"], dtype=float).reshape(2, 2)[0, 0])
        else:
            params["deltaJ1"] = 0.05
    if "deltaJ2" not in params:
        if "Jd2" in params:
            params["deltaJ2"] = float(np.asarray(params["Jd2"], dtype=float).reshape(2, 2)[0, 0])
        else:
            params["deltaJ2"] = 0.05

    params["Jd1"] = np.diag([params["deltaJ1"], params["deltaJ1"]])
    params["Jd2"] = np.diag([params["deltaJ2"], params["deltaJ2"]])

    # Desired rotations.
    params["c1_des"] = float(np.cos(params["thetaR1_des"]))
    params["s1_des"] = float(np.sin(params["thetaR1_des"]))
    params["c2_des"] = float(np.cos(params["thetaR2_des"]))
    params["s2_des"] = float(np.sin(params["thetaR2_des"]))

    # Desired final step comes from YAML-derived thetaF target values.
    if "thetaF1_des" not in params or "thetaF2_des" not in params:
        raise KeyError(
            "Missing thetaF target values. Expected 'thetaF1_des' and "
            "'thetaF2_des' from the YAML-derived params."
        )

    params["a1_des"] = float(np.cos(params["thetaF1_des"]))
    params["b1_des"] = float(np.sin(params["thetaF1_des"]))
    params["a2_des"] = float(np.cos(params["thetaF2_des"]))
    params["b2_des"] = float(np.sin(params["thetaF2_des"]))

    # Initial previous step rotation F_0 from YAML.
    if "thetaF1_0" not in params or "thetaF2_0" not in params:
        raise KeyError(
            "Missing thetaF initial values. Expected 'thetaF1_0' and "
            "'thetaF2_0' from the YAML-derived params."
        )

    # Bounds and regularization.
    params.setdefault("alpha_lam", 0.0)
    params.setdefault("u_max", 1.0)
    params.setdefault("lambda_max", 1.0)

    # Solver normalization metadata.  The SDP variables u and lambda are
    # normalized to [-1, 1]; these scales map them back to physical quantities.
    params["u_solver_scale"] = float(params["u_max"])
    params["lambda_solver_scale"] = float(params["lambda_max"])
    params["lambda_bar_scale"] = float(params["lambda_max"])

    if "max_step_angle" not in params:
        if "theta_step_max" in params:
            params["max_step_angle"] = float(params["theta_step_max"])
        else:
            params["max_step_angle"] = float(np.deg2rad(params.get("max_step_angle_deg", 20.0)))

    if "max_step_angle1" not in params:
        if "max_step_angle_deg_link1" in params:
            params["max_step_angle1"] = float(np.deg2rad(params["max_step_angle_deg_link1"]))
        elif "max_step_angle1_deg" in params:
            params["max_step_angle1"] = float(np.deg2rad(params["max_step_angle1_deg"]))
        else:
            params["max_step_angle1"] = float(params["max_step_angle"])
    if "max_step_angle2" not in params:
        if "max_step_angle_deg_link2" in params:
            params["max_step_angle2"] = float(np.deg2rad(params["max_step_angle_deg_link2"]))
        elif "max_step_angle2_deg" in params:
            params["max_step_angle2"] = float(np.deg2rad(params["max_step_angle2_deg"]))
        else:
            params["max_step_angle2"] = float(params["max_step_angle"])

    params["max_step_angle1_deg"] = float(np.rad2deg(params["max_step_angle1"]))
    params["max_step_angle2_deg"] = float(np.rad2deg(params["max_step_angle2"]))
    params["max_step_angle_deg_link1"] = params["max_step_angle1_deg"]
    params["max_step_angle_deg_link2"] = params["max_step_angle2_deg"]
    params["a1_min"] = float(np.cos(params["max_step_angle1"]))
    params["a2_min"] = float(np.cos(params["max_step_angle2"]))

    # Optional MPC boundary data from numerical simulation.
    #
    # Expected:
    #   c1_current, s1_current, c2_current, s2_current
    #   a1_prev, b1_prev, a2_prev, b2_prev
    #
    # Or directly:
    #   c1_1, s1_1, c2_1, s2_1
    #   a1_0, b1_0, a2_0, b2_0
    if mpc_initial is not None:
        params.update(mpc_initial)

    # If mpc_initial uses names from convert_state_to_sdp_initial_scalars,
    # convert them into SDP boundary names.
    if "c1_0" in params and "a1_prev" in params and "c1_1" not in params:
        # Here c1_0 from the simulation conversion means "current".
        # Rename internally to avoid confusion.
        params["c1_current"] = float(params["c1_0"])
        params["s1_current"] = float(params["s1_0"])
        params["c2_current"] = float(params["c2_0"])
        params["s2_current"] = float(params["s2_0"])

    if "c1_current" not in params:
        params["c1_current"] = float(np.cos(params["thetaR1_0"]))
        params["s1_current"] = float(np.sin(params["thetaR1_0"]))
        params["c2_current"] = float(np.cos(params["thetaR2_0"]))
        params["s2_current"] = float(np.sin(params["thetaR2_0"]))

    if "a1_prev" not in params:
        params["a1_prev"] = float(np.cos(params["thetaF1_0"]))
        params["b1_prev"] = float(np.sin(params["thetaF1_0"]))
        params["a2_prev"] = float(np.cos(params["thetaF2_0"]))
        params["b2_prev"] = float(np.sin(params["thetaF2_0"]))

    attach_mapping_to_params(params)
    params["ids_remap"] = get_remapped_ids(params)

    return params


def create_output_dirs(prefix: str):
    for folder in ["data", "markdown", "figs", "logs"]:
        path = PROJECT_ROOT / folder / prefix
        path.mkdir(parents=True, exist_ok=True)


def build_polynomial_system(params: Dict[str, Any]):
    """
    Build NumPolySystem with constraints and objective.
    """
    N = int(params["N"])
    total_var_num = int(params["total_var_num"])

    ps = NumPolySystem(n_vars=total_var_num)
    idf = params["id"]

    def v(prefix, k):
        return ps.var(idf(prefix, k) - 1)

    eq_mask_sys = []

    # ------------------------------------------------------------
    # Initial/MPC boundary constraints:
    # fix R0, F0, R1.
    # ------------------------------------------------------------
    eqs, ineqs, eq_mask = get_init_constraints(
        v("c1", 0),
        v("s1", 0),
        v("c2", 0),
        v("s2", 0),
        v("a1", 0),
        v("b1", 0),
        v("a2", 0),
        v("b2", 0),
        v("c1", 1),
        v("s1", 1),
        v("c2", 1),
        v("s2", 1),
        params,
    )

    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)

    # ------------------------------------------------------------
    # Constraints over horizon.
    # ------------------------------------------------------------
    for k in range(N + 1):
        # R SO(2), k=0,...,N.
        eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_R(
            v("c1", k),
            v("s1", k),
            v("c2", k),
            v("s2", k),
            params,
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        if k < N:
            # F SO(2), k=0,...,N-1.
            eqs, ineqs, eq_mask = get_SO2_orthogonality_constraint_rotation_F(
                v("a1", k),
                v("b1", k),
                v("a2", k),
                v("b2", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            if 1 <= k < N:
                # Future optimized step bounds only.
                #
                # F_0 is a fixed past value from the previous MPC simulation
                # interval. It remains fixed, remains SO(2), and remains in
                # the kinematic/dynamic equations, but it is not constrained by
                # the future step-angle bound.
                eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_1(
                    v("a1", k),
                    params,
                )
                for eq in eqs:
                    ps.add_eq(eq)
                for ineq in ineqs:
                    ps.add_ineq(ineq)
                eq_mask_sys.extend(eq_mask)

                eqs, ineqs, eq_mask = get_step_angle_bound_constraint_link_2(
                    v("a2", k),
                    params,
                )
                for eq in eqs:
                    ps.add_eq(eq)
                for ineq in ineqs:
                    ps.add_ineq(ineq)
                eq_mask_sys.extend(eq_mask)

        if 1 <= k <= N:
            # Kinematics R_k = R_{k-1} F_{k-1}.
            eqs, ineqs, eq_mask = get_rotational_kinematics_link1(
                v("c1", k - 1),
                v("s1", k - 1),
                v("c1", k),
                v("s1", k),
                v("a1", k - 1),
                v("b1", k - 1),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_rotational_kinematics_link2(
                v("c2", k - 1),
                v("s2", k - 1),
                v("c2", k),
                v("s2", k),
                v("a2", k - 1),
                v("b2", k - 1),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

        if 1 <= k < N:
            # Reduced translational dynamics.
            eqs, ineqs, eq_mask = get_translational_dynamics_link1(
                v("c1", k),
                v("s1", k),
                v("a1", k - 1),
                v("b1", k - 1),
                v("a1", k),
                v("b1", k),
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_translational_dynamics_link2(
                v("c1", k),
                v("s1", k),
                v("a1", k - 1),
                v("b1", k - 1),
                v("a1", k),
                v("b1", k),
                v("c2", k),
                v("s2", k),
                v("a2", k - 1),
                v("b2", k - 1),
                v("a2", k),
                v("b2", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # Reduced rotational dynamics.
            # Consistent interior-node indexing: the DEL stencil uses
            # F_{k-1}, F_k, lambda_k, u_k, so the constraint moments are
            # evaluated at R_k, not R_{k+1}.
            eqs, ineqs, eq_mask = get_rotational_dynamics_link1(
                v("b1", k - 1),
                v("b1", k),
                v("c1", k),
                v("s1", k),
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                v("u", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_rotational_dynamics_link2(
                v("b2", k - 1),
                v("b2", k),
                v("c2", k),
                v("s2", k),
                v("lam12x", k),
                v("lam12y", k),
                v("u", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            # Control and lambda bounds.
            eqs, ineqs, eq_mask = get_control_bounds(v("u", k), params)
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

            eqs, ineqs, eq_mask = get_lambda_bounds(
                v("lam0x", k),
                v("lam0y", k),
                v("lam12x", k),
                v("lam12y", k),
                params,
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

    # Objective.
    ps.set_obj(build_objective(v, params))

    params["eq_mask_sys"] = eq_mask_sys

    return ps, params


def solve_sdp(
    params: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build and solve the reduced Acrobot SDP.

    Returns:
        result dictionary containing:
            params
            result
            res
            coeff_info
            aux_info
            solutions
            extracted_vectors
            first_control
    """
    total_start = time.time()

    params = complete_sdp_params(params)
    f0_step_bound_diagnostic = _fixed_f0_step_bound_diagnostic(params)
    params["f0_step_bound_diagnostic"] = f0_step_bound_diagnostic

    if prefix is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Timestamp alone is only 1-second resolution, which is not unique
        # across concurrently running processes (e.g. parallel SLURM jobs)
        # sharing this same project directory. Salt it with the SLURM job ID
        # (or PID if not running under SLURM) so concurrent runs never share
        # a prefix and can't delete each other's in-progress log directory.
        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        prefix = f"Acrobot_SO2_Reduced_MPC/{current_time}_{job_id}/"

    create_output_dirs(prefix)

    log_path = PROJECT_ROOT / "logs" / prefix / "log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Acrobot SO(2) Reduced No-X/No-V SDP-MPC Optimization\n")
        f.write("=" * 80 + "\n")
        f.write("Thesis indexing: u_k, lambda_k for k=1,...,N-1.\n")
        f.write("MPC interpretation: current state is SDP node 1; first applied control is u_1.\n")
        f.write("=" * 80 + "\n")
        f.write("params:\n")
        for key in sorted(params.keys()):
            if callable(params[key]):
                continue
            f.write(f"{key}: {params[key]}\n")
        f.write("\n")
        _write_fixed_f0_diagnostic(f, f0_step_bound_diagnostic)

    ps, params = build_polynomial_system(params)

    kappa = int(params["kappa"])
    total_var_num = int(params["total_var_num"])
    var_mapping = params["var_mapping"]

    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    poly_data = ps.get_supp_rpt_data(kappa)

    print("Construction finished.")

    if params["cs_mode"] == "SELF":
        cliques = get_cliques_for_cstss(int(params["N"]), params)
    else:
        cliques = []

    params["cliques"] = cliques

    start_time = time.time()

    solver_output_buffer = io.StringIO()
    solver_exception: Optional[Exception] = None
    try:
        with redirect_stdout(_TeeTextStream(sys.stdout, solver_output_buffer)):
            result, res, coeff_info, aux_info = CSTSS_pybind(
                poly_data,
                kappa,
                total_var_num,
                params,
            )
    except Exception as exc:
        solver_exception = exc
        result, res, coeff_info, aux_info = float("nan"), {}, {}, {}

    elapsed = time.time() - start_time
    problem_status, solution_status = _parse_mosek_statuses(
        solver_output_buffer.getvalue()
    )
    aux_info["problem_status"] = problem_status
    aux_info["solution_status"] = solution_status
    aux_info["f0_step_bound_diagnostic"] = f0_step_bound_diagnostic
    if solver_exception is not None:
        aux_info["solver_exception"] = str(solver_exception)

    infeasible_status = is_infeasible_status(problem_status, solution_status)
    unknown_status = is_unknown_status(problem_status, solution_status)
    valid_solution = (
        problem_status == "PRIMAL_AND_DUAL_FEASIBLE"
        and solution_status == "OPTIMAL"
    )
    extraction_allowed = (
        not infeasible_status
        and solver_exception is None
    )
    aux_info["infeasible_status"] = infeasible_status
    aux_info["unknown_status"] = unknown_status
    aux_info["extraction_attempted"] = False
    aux_info["extraction_success"] = False
    aux_info["accepted_with_unknown_status"] = False
    if solver_exception is not None and valid_solution:
        raise solver_exception

    aux_info["result"] = result
    params["aux_info"] = aux_info

    with open(log_path, "a", encoding="utf-8") as f:
        result_str = str(result) if isinstance(result, list) else f"{float(np.asarray(result).squeeze()):.20f}"
        f.write(
            f"\nResult={result_str}, operation time={elapsed:.5f}, "
            f"mosek time={aux_info.get('mosek_time', 0):.5f}\n"
        )

    # Clique rank ordering for ordered_extract_CS.
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_remapped = []
        averages = []

        for clique in aux_info["cliques"]:
            remapped = [params["ids_remap"][i - 1] for i in clique]
            cliques_remapped.append(remapped)
            averages.append(np.mean(remapped))

        params["cliques_rank"] = np.argsort(averages)
    else:
        params["cliques_rank"] = []

    # Markdown visualization.
    try:
        clique_supp_list = []
        clique_coeff_list = []
        kappa_width = 2 * kappa

        if "cliques" in aux_info and aux_info["cliques"]:
            for ii in params["cliques_rank"]:
                sorted_vars = sorted(aux_info["cliques"][ii])
                supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
                for idx_v, j in enumerate(sorted_vars):
                    supp[idx_v, -1] = j
                clique_supp_list.append(supp)
                clique_coeff_list.append(np.ones(len(sorted_vars)))

        md_path = PROJECT_ROOT / "markdown" / prefix / "opt_problem.md"

        with open(md_path, "w", encoding="utf-8") as md:
            md.write("equality constraints:\n")
            numpoly_visualize(aux_info["supp_rpt_h"], aux_info["coeff_h"], var_mapping, md)

            md.write("\ninequality constraints:\n")
            numpoly_visualize(aux_info["supp_rpt_g"], aux_info["coeff_g"], var_mapping, md)

            md.write("\nobjective:\n")
            numpoly_visualize([aux_info["supp_rpt_f"]], [aux_info["coeff_f"]], var_mapping, md)

            md.write("\ncliques:\n")
            numpoly_visualize(clique_supp_list, clique_coeff_list, var_mapping, md)

    except Exception as exc:
        print(f"Markdown visualization skipped: {exc}")

    params_to_save = {k: v for k, v in params.items() if not callable(v)}

    with open(PROJECT_ROOT / "data" / prefix / "params.pkl", "wb") as f:
        pickle.dump(params_to_save, f)

    with open(PROJECT_ROOT / "data" / prefix / "res.pkl", "wb") as f:
        pickle.dump(
            {
                "result": result,
                "res": res,
                "coeff_info": coeff_info,
                "aux_info": aux_info,
            },
            f,
        )

    def failure_result(reason: str, total_time: float) -> Dict[str, Any]:
        aux_info["rejection_reason"] = reason
        return {
            "params": params,
            "result": result,
            "res": res,
            "coeff_info": coeff_info,
            "aux_info": aux_info,
            "solutions": {},
            "extracted_vectors": {},
            "extraction_info": {},
            "errors_by_method": {},
            "first_control": None,
            "preferred_extraction": None,
            "prefix": prefix,
            "Xs": [],
            "problem_status": problem_status,
            "solution_status": solution_status,
            "infeasible_status": infeasible_status,
            "unknown_status": unknown_status,
            "extraction_attempted": bool(aux_info.get("extraction_attempted", False)),
            "extraction_success": False,
            "accepted_with_unknown_status": False,
            "control_applied": False,
            "rejection_reason": reason,
            "f0_step_bound_diagnostic": f0_step_bound_diagnostic,
            "total_time": total_time,
        }

    if not params.get("if_solve", True):
        total_time = time.time() - total_start
        print("Debugging mode: problem constructed, but if_solve=False.")
        print(f"Total time: {total_time:.5f} s")

        return {
            "params": params,
            "result": result,
            "res": res,
            "coeff_info": coeff_info,
            "aux_info": aux_info,
            "solutions": {},
            "extracted_vectors": {},
            "first_control": None,
            "problem_status": problem_status,
            "solution_status": solution_status,
            "infeasible_status": infeasible_status,
            "unknown_status": unknown_status,
            "extraction_attempted": False,
            "extraction_success": False,
            "accepted_with_unknown_status": False,
            "control_applied": False,
            "f0_step_bound_diagnostic": f0_step_bound_diagnostic,
            "prefix": prefix,
            "Xs": [],
        }

    if not extraction_allowed:
        total_time = time.time() - total_start
        if infeasible_status:
            rejection_reason = (
                "Infeasible solver status; no SDP variables or control extracted"
            )
        elif solver_exception is not None:
            rejection_reason = (
                "Solver backend exception and no usable extracted control"
            )
        else:
            rejection_reason = (
                "UNKNOWN or non-optimal solver status and no usable extracted control"
            )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nSDP solve not accepted; no SDP variables or control extracted.\n")
            f.write(f"problem_status: {problem_status}\n")
            f.write(f"solution_status: {solution_status}\n")
            f.write(f"infeasible_status: {infeasible_status}\n")
            f.write(f"unknown_status: {unknown_status}\n")
            f.write(f"rejection_reason: {rejection_reason}\n")
            if solver_exception is not None:
                f.write(f"backend exception after solve: {solver_exception}\n")
        print(
            "SDP solve not accepted; no control extracted: "
            f"problem_status={problem_status}, solution_status={solution_status}; "
            f"{rejection_reason}"
        )
        return failure_result(rejection_reason, total_time)

    aux_info["extraction_attempted"] = True
    try:
        # Extraction is permitted for optimal solves and non-infeasible UNKNOWN solves.
        if params["relax_mode"] == "MOMENT":
            Xs = res["Xopt"]
        elif params["relax_mode"] == "SOS":
            Xs = [-S for S in res["Sopt"]]
        else:
            Xs = res.get("Xopt", [])

        ts_info = aux_info["ts_info"]
        cliques_aux = aux_info["cliques"]
        mon_rpt = aux_info["mon_rpt"]

        mom_mat_num = sum(len(ts_info[i]) for i in range(len(cliques_aux)))
        mom_mat_rpt = [None] * mom_mat_num

        idx = 0
        for i in range(len(cliques_aux)):
            for j in range(len(ts_info[i])):
                rpt = mon_rpt[i][ts_info[i][j], :]
                rpt = np.hstack([np.zeros_like(rpt), rpt])
                mom_mat_rpt[idx] = rpt
                idx += 1

        extracted_vectors = {}
        solutions = {}
        extraction_info = {}

        v_opt_naive, output_info_naive = naive_extract(Xs, mon_rpt, ts_info, total_var_num)
        extracted_vectors["naive"] = v_opt_naive
        extraction_info["naive"] = output_info_naive
        solutions["naive"] = extract_solution_variables(v_opt_naive, params)

        with open(PROJECT_ROOT / "data" / prefix / "v_opt_naive.pkl", "wb") as f:
            pickle.dump(v_opt_naive, f)

        if params["ts_mode"] == "NON":
            v_opt_robust, output_info_robust = robust_extract_CS(
                Xs,
                mom_mat_rpt,
                total_var_num,
                1e-2,
            )
            extracted_vectors["robust"] = v_opt_robust
            extraction_info["robust"] = output_info_robust
            solutions["robust"] = extract_solution_variables(v_opt_robust, params)

            with open(PROJECT_ROOT / "data" / prefix / "v_opt_robust.pkl", "wb") as f:
                pickle.dump(v_opt_robust, f)

            v_opt_ordered, output_info_ordered = ordered_extract_CS(
                Xs,
                mom_mat_rpt,
                total_var_num,
                1e-2,
                params.get("cliques_rank", []),
            )
            extracted_vectors["ordered"] = v_opt_ordered
            extraction_info["ordered"] = output_info_ordered
            solutions["ordered"] = extract_solution_variables(v_opt_ordered, params)

            with open(PROJECT_ROOT / "data" / prefix / "v_opt_ordered.pkl", "wb") as f:
                pickle.dump(v_opt_ordered, f)

        preferred = "ordered" if "ordered" in solutions else "robust" if "robust" in solutions else "naive"
        first_control = get_first_mpc_control(solutions[preferred])

        errors_by_method = {
            name: compute_SO2_errors(sol, int(params["N"]))
            for name, sol in solutions.items()
        }
    except Exception as exc:
        total_time = time.time() - total_start
        aux_info["extraction_exception"] = str(exc)
        if unknown_status or not valid_solution:
            rejection_reason = (
                "UNKNOWN or non-optimal solver status and no usable extracted control"
            )
        else:
            rejection_reason = "Extraction failed and no usable extracted control"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nSDP extraction failed; no usable control extracted.\n")
            f.write(f"problem_status: {problem_status}\n")
            f.write(f"solution_status: {solution_status}\n")
            f.write(f"infeasible_status: {infeasible_status}\n")
            f.write(f"unknown_status: {unknown_status}\n")
            f.write(f"extraction exception: {exc}\n")
            f.write(f"rejection_reason: {rejection_reason}\n")
        print(
            "SDP extraction failed; no usable control extracted: "
            f"problem_status={problem_status}, solution_status={solution_status}; "
            f"{rejection_reason}"
        )
        return failure_result(rejection_reason, total_time)

    if not is_finite_control(first_control):
        total_time = time.time() - total_start
        rejection_reason = (
            "UNKNOWN or non-optimal solver status and no usable extracted control"
            if unknown_status or not valid_solution
            else "Extracted first control is not finite"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nSDP extraction did not produce a finite first control.\n")
            f.write(f"problem_status: {problem_status}\n")
            f.write(f"solution_status: {solution_status}\n")
            f.write(f"first_control: {first_control}\n")
            f.write(f"rejection_reason: {rejection_reason}\n")
        return failure_result(rejection_reason, total_time)

    first_control = float(first_control)
    aux_info["extraction_success"] = True
    aux_info["accepted_with_unknown_status"] = bool(unknown_status and not valid_solution)
    aux_info["first_control"] = first_control

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXTRACTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"preferred extraction: {preferred}\n")
        f.write(f"first MPC control u_1: {first_control:+.12e}\n")
        f.write(f"infeasible_status: {infeasible_status}\n")
        f.write(f"unknown_status: {unknown_status}\n")
        f.write("extraction_attempted: True\n")
        f.write("extraction_success: True\n")
        f.write(
            "accepted_with_unknown_status: "
            f"{bool(unknown_status and not valid_solution)}\n"
        )
        if unknown_status and not valid_solution:
            f.write("UNKNOWN status accepted because extraction produced finite control.\n")

        for name, err in errors_by_method.items():
            f.write(f"\n{name.upper()} SO(2) max errors:\n")
            for key, vals in err.items():
                if vals:
                    f.write(f"  {key}: {max(vals):.12e}\n")

        f.write("\nPredicted trajectory summary:\n")
        sol = solutions[preferred]
        for k in range(int(params["N"]) + 1):
            f.write(
                f"k={k:3d}: "
                f"thetaR1={sol['thetaR1'][k]:+.8f}, "
                f"thetaR2={sol['thetaR2'][k]:+.8f}"
            )
            if 1 <= k < int(params["N"]):
                f.write(f", u={sol['u'][k]:+.8f}")
            f.write("\n")

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("SDP SOLVE FINISHED")
    print("=" * 80)
    print(f"preferred extraction: {preferred}")
    print(f"first MPC control u_1: {first_control:+.12e}")
    print(f"total time: {total_time:.3f} s")
    print("=" * 80)

    return {
        "params": params,
        "result": result,
        "res": res,
        "coeff_info": coeff_info,
        "aux_info": aux_info,
        "solutions": solutions,
        "extracted_vectors": extracted_vectors,
        "extraction_info": extraction_info,
        "errors_by_method": errors_by_method,
        "first_control": first_control,
        "problem_status": problem_status,
        "solution_status": solution_status,
        "infeasible_status": infeasible_status,
        "unknown_status": unknown_status,
        "extraction_attempted": True,
        "extraction_success": True,
        "accepted_with_unknown_status": bool(unknown_status and not valid_solution),
        "control_applied": True,
        "f0_step_bound_diagnostic": f0_step_bound_diagnostic,
        "preferred_extraction": preferred,
        "prefix": prefix,
        # These are the actual clique moment matrices used by the extraction
        # routines above.  Logging code saves them in compressed NumPy form.
        "Xs": Xs,
    }


def solve_from_yaml(
    yaml_path: str | Path = PROJECT_ROOT / "config" / "acrobot_physical.yaml",
    mpc_initial: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Load YAML config, build params, optionally insert MPC initial state, solve SDP.
    """
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)

    if mpc_initial is not None:
        params.update(mpc_initial)

    return solve_sdp(params)


if __name__ == "__main__":
    solve_from_yaml()
