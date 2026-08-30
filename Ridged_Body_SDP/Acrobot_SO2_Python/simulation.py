from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from config.config_loader import load_yaml_config, build_common_params
from Numerical_Simulation.lie_group_so2 import angle_from_R, orth_error_so2, det_error_so2
from Numerical_Simulation.solver_lgvi_acrobot import (
    AcrobotReducedState,
    LGVISolveError,
    make_model_from_params,
    make_initial_state_from_params,
    simulate_one_control_interval_from_params,
    convert_state_to_sdp_initial_scalars,
    diagnostics_lgvi,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"
SIM_RESULTS_DIR = RESULTS_ROOT / "Results-Open_Loop-Simulation"
MPC_RESULTS_DIR = RESULTS_ROOT / "Results-MPC"


# -----------------------------------------------------------------------------
# JSON / CSV helpers
# -----------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = []
            for key in keys:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.16e}")
                else:
                    vals.append(str(val))
            f.write(",".join(vals) + "\n")


# -----------------------------------------------------------------------------
# Geometry and error helpers
# -----------------------------------------------------------------------------


def acrobot_points_from_angles(theta1: float, theta2: float, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Full-link points using your absolute-angle convention:
    # x = l sin(theta), y = -l cos(theta)
    
    p0 = np.asarray(params.get("p_0", params.get("p0", [0.0, 0.0])), dtype=float).reshape(2)
    l1 = float(params["l1"])
    l2 = float(params["l2"])

    p1 = p0 + np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)], dtype=float)
    p2 = p1 + np.array([l2 * np.sin(theta2), -l2 * np.cos(theta2)], dtype=float)
    return p0, p1, p2


def acrobot_points_from_state(state: AcrobotReducedState, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return acrobot_points_from_angles(
        theta1=float(angle_from_R(state.R1)),
        theta2=float(angle_from_R(state.R2)),
        params=params,
    )


def state_to_row(
    state: AcrobotReducedState,
    params: Mapping[str, Any],
    time_value: float,
    k: int,
    u_value: float,
) -> Dict[str, float]:
    thetaR1 = float(angle_from_R(state.R1))
    thetaR2 = float(angle_from_R(state.R2))
    thetaF1 = float(angle_from_R(state.F1_prev))
    thetaF2 = float(angle_from_R(state.F2_prev))
    dt_sim = float(params["dt_sim"])
    if dt_sim <= 0.0:
        raise ValueError("dt_sim must be positive when computing angular velocity")
    omega1_rad_s = thetaF1 / dt_sim
    omega2_rad_s = thetaF2 / dt_sim
    p0, p1, p2 = acrobot_points_from_angles(thetaR1, thetaR2, params)

    return {
        "k": float(k),
        "t": float(time_value),
        "u": float(u_value),
        "thetaR1_rad": thetaR1,
        "thetaR2_rad": thetaR2,
        "thetaR1_deg": float(np.rad2deg(thetaR1)),
        "thetaR2_deg": float(np.rad2deg(thetaR2)),
        "thetaF1_rad": thetaF1,
        "thetaF2_rad": thetaF2,
        "thetaF1_deg": float(np.rad2deg(thetaF1)),
        "thetaF2_deg": float(np.rad2deg(thetaF2)),
        "omega1_rad_s": float(omega1_rad_s),
        "omega2_rad_s": float(omega2_rad_s),
        "omega1_deg_s": float(np.rad2deg(omega1_rad_s)),
        "omega2_deg_s": float(np.rad2deg(omega2_rad_s)),
        "angular_velocity_norm_rad_s": float(np.hypot(omega1_rad_s, omega2_rad_s)),
        "angular_velocity_norm_deg_s": float(
            np.rad2deg(np.hypot(omega1_rad_s, omega2_rad_s))
        ),
        "R1_orth_error": float(orth_error_so2(state.R1)),
        "R2_orth_error": float(orth_error_so2(state.R2)),
        "F1_orth_error": float(orth_error_so2(state.F1_prev)),
        "F2_orth_error": float(orth_error_so2(state.F2_prev)),
        "R1_det_error": float(det_error_so2(state.R1)),
        "R2_det_error": float(det_error_so2(state.R2)),
        "F1_det_error": float(det_error_so2(state.F1_prev)),
        "F2_det_error": float(det_error_so2(state.F2_prev)),
        "base_x": float(p0[0]),
        "base_y": float(p0[1]),
        "elbow_x": float(p1[0]),
        "elbow_y": float(p1[1]),
        "tip_x": float(p2[0]),
        "tip_y": float(p2[1]),
    }


def target_errors_from_row(row: Mapping[str, float], params: Mapping[str, Any]) -> Dict[str, float]:
    thetaR1_des = float(params["thetaR1_des"])
    thetaR2_des = float(params["thetaR2_des"])
    thetaF1_des = float(params.get("thetaF1_des", 0.0))
    thetaF2_des = float(params.get("thetaF2_des", 0.0))

    def wrap(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    eR1 = wrap(float(row["thetaR1_rad"]) - thetaR1_des)
    eR2 = wrap(float(row["thetaR2_rad"]) - thetaR2_des)
    eF1 = wrap(float(row["thetaF1_rad"]) - thetaF1_des)
    eF2 = wrap(float(row["thetaF2_rad"]) - thetaF2_des)

    dt_sdp = float(params["dt_sdp"])
    if dt_sdp <= 0.0:
        raise ValueError("dt_sdp must be positive when computing target angular velocity")
    omega1_des = thetaF1_des / dt_sdp
    omega2_des = thetaF2_des / dt_sdp
    omega1 = float(row["omega1_rad_s"])
    omega2 = float(row["omega2_rad_s"])
    e_omega1 = omega1 - omega1_des
    e_omega2 = omega2 - omega2_des
    e_omega_norm = float(np.hypot(e_omega1, e_omega2))

    return {
        "target_angle_error_norm_rad": float(np.sqrt(eR1 * eR1 + eR2 * eR2)),
        "target_angle_error_norm_deg": float(np.rad2deg(np.sqrt(eR1 * eR1 + eR2 * eR2))),
        "target_step_error_norm_rad": float(np.sqrt(eF1 * eF1 + eF2 * eF2)),
        "target_step_error_norm_deg": float(np.rad2deg(np.sqrt(eF1 * eF1 + eF2 * eF2))),
        "target_error_R1_deg": float(np.rad2deg(eR1)),
        "target_error_R2_deg": float(np.rad2deg(eR2)),
        "target_error_F1_deg": float(np.rad2deg(eF1)),
        "target_error_F2_deg": float(np.rad2deg(eF2)),
        "target_angular_velocity_error_norm_rad_s": e_omega_norm,
        "target_angular_velocity_error_norm_deg_s": float(np.rad2deg(e_omega_norm)),
        "target_omega1_error_rad_s": float(e_omega1),
        "target_omega2_error_rad_s": float(e_omega2),
        "target_omega1_error_deg_s": float(np.rad2deg(e_omega1)),
        "target_omega2_error_deg_s": float(np.rad2deg(e_omega2)),
    }


# -----------------------------------------------------------------------------
# Fine rotation history for the coarse SDP incoming step
# -----------------------------------------------------------------------------


def _copy_rotation_history_state(
    R1: np.ndarray,
    R2: np.ndarray,
) -> AcrobotReducedState:
    # Create a history-only state; previous F is irrelevant for this use
    return AcrobotReducedState(
        R1=np.asarray(R1, dtype=float).reshape(2, 2).copy(),
        R2=np.asarray(R2, dtype=float).reshape(2, 2).copy(),
        F1_prev=np.eye(2),
        F2_prev=np.eye(2),
    )


def _append_fine_rotation_history(
    rotation_history: list[tuple[float, np.ndarray, np.ndarray]],
    sim: Mapping[str, Any],
    interval_start_time: float,
) -> None:
    # Append every new fine-simulation node to the global rotation history.
    local_t = np.asarray(sim["t"], dtype=float).reshape(-1)
    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    if len(local_t) != len(R1) or len(local_t) != len(R2):
        raise ValueError("Simulation time and rotation arrays have inconsistent lengths")

    tol = 1.0e-12
    for k in range(1, len(local_t)):
        global_time = float(interval_start_time) + float(local_t[k])
        if rotation_history and global_time <= rotation_history[-1][0] + tol:
            if abs(global_time - rotation_history[-1][0]) <= tol:
                rotation_history[-1] = (global_time, R1[k].copy(), R2[k].copy())
                continue
            raise ValueError("Rotation history times must be strictly increasing")
        rotation_history.append((global_time, R1[k].copy(), R2[k].copy()))


def _rotation_history_state_at(
    rotation_history: list[tuple[float, np.ndarray, np.ndarray]],
    target_time: float,
) -> AcrobotReducedState:
    # Read or geodesically interpolate an SO(2) state
    if not rotation_history:
        raise ValueError("Rotation history is empty")

    times = [sample[0] for sample in rotation_history]
    tol = 1.0e-10
    if target_time < times[0] - tol or target_time > times[-1] + tol:
        raise ValueError(
            f"Requested history time {target_time} is outside "
            f"[{times[0]}, {times[-1]}]"
        )

    idx = bisect_left(times, target_time)
    if idx < len(times) and abs(times[idx] - target_time) <= tol:
        _, R1, R2 = rotation_history[idx]
        return _copy_rotation_history_state(R1, R2)
    if idx > 0 and abs(times[idx - 1] - target_time) <= tol:
        _, R1, R2 = rotation_history[idx - 1]
        return _copy_rotation_history_state(R1, R2)

    if idx == 0 or idx >= len(times):
        raise ValueError("Could not bracket requested rotation-history time")

    t0, R1_0, R2_0 = rotation_history[idx - 1]
    t1, R1_1, R2_1 = rotation_history[idx]
    alpha = float((target_time - t0) / (t1 - t0))

    theta1 = float(angle_from_R(R1_0.T @ R1_1))
    theta2 = float(angle_from_R(R2_0.T @ R2_1))
    R1_interp = R1_0 @ np.array(
        [[np.cos(alpha * theta1), -np.sin(alpha * theta1)],
         [np.sin(alpha * theta1),  np.cos(alpha * theta1)]],
        dtype=float,
    )
    R2_interp = R2_0 @ np.array(
        [[np.cos(alpha * theta2), -np.sin(alpha * theta2)],
         [np.sin(alpha * theta2),  np.cos(alpha * theta2)]],
        dtype=float,
    )
    return _copy_rotation_history_state(R1_interp, R2_interp)


def _select_sdp_history_reference(
    rotation_history: list[tuple[float, np.ndarray, np.ndarray]],
    current_time: float,
    dt_sdp: float,
    thetaF1_initial_sdp: float = 0.0,
    thetaF2_initial_sdp: float = 0.0,
) -> tuple[AcrobotReducedState, float, str, float]:
    
    # Select R(t-dt_sdp), with a virtual startup history when t < dt_sdp.

    if dt_sdp <= 0.0:
        raise ValueError("dt_sdp must be positive")
    if not rotation_history:
        raise ValueError("Rotation history is empty")

    first_time = float(rotation_history[0][0])
    target_time = float(current_time) - float(dt_sdp)
    tol = 1.0e-10 * max(1.0, abs(current_time), abs(dt_sdp))

    if target_time >= first_time - tol:
        reference_state = _rotation_history_state_at(rotation_history, target_time)
        return reference_state, float(dt_sdp), "full_dt_sdp_history", target_time

    available_duration = float(current_time) - first_time
    if available_duration <= 0.0:
        raise ValueError("No positive measured history is available for startup")
    missing_duration = float(dt_sdp) - available_duration
    if missing_duration < -tol:
        raise ValueError("Startup history selection produced a negative missing duration")
    missing_fraction = max(0.0, missing_duration / float(dt_sdp))

    _, R1_first, R2_first = rotation_history[0]
    F1_missing = np.array(
        [[np.cos(missing_fraction * thetaF1_initial_sdp),
          -np.sin(missing_fraction * thetaF1_initial_sdp)],
         [np.sin(missing_fraction * thetaF1_initial_sdp),
          np.cos(missing_fraction * thetaF1_initial_sdp)]],
        dtype=float,
    )
    F2_missing = np.array(
        [[np.cos(missing_fraction * thetaF2_initial_sdp),
          -np.sin(missing_fraction * thetaF2_initial_sdp)],
         [np.sin(missing_fraction * thetaF2_initial_sdp),
          np.cos(missing_fraction * thetaF2_initial_sdp)]],
        dtype=float,
    )

    # R_first = R_virtual @ F_missing, hence
    # R_virtual = R_first @ F_missing.T.
    reference_state = _copy_rotation_history_state(
        R1_first @ F1_missing.T,
        R2_first @ F2_missing.T,
    )
    return (
        reference_state,
        float(dt_sdp),
        "startup_virtual_initial_velocity_history",
        target_time,
    )


# -----------------------------------------------------------------------------
# Simulation logging
# -----------------------------------------------------------------------------


def simulation_to_rows(
    sim: Mapping[str, Any],
    params: Mapping[str, Any],
    initial_state: Optional[AcrobotReducedState] = None,
) -> list[dict[str, float]]:
    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    F1 = np.asarray(sim["F1"], dtype=float)
    F2 = np.asarray(sim["F2"], dtype=float)
    t = np.asarray(sim["t"], dtype=float)
    u = np.asarray(sim["u"], dtype=float).reshape(-1)
    X = np.asarray(sim.get("X", []), dtype=float)

    rows: list[dict[str, float]] = []
    n_nodes = R1.shape[0]
    for k in range(n_nodes):
        if k == 0:
            if initial_state is not None:
                F1_prev = initial_state.F1_prev
                F2_prev = initial_state.F2_prev
            elif "initial_F1_prev" in sim and "initial_F2_prev" in sim:
                F1_prev = np.asarray(sim["initial_F1_prev"], dtype=float).reshape(2, 2)
                F2_prev = np.asarray(sim["initial_F2_prev"], dtype=float).reshape(2, 2)
            else:
                F1_prev = np.eye(2)
                F2_prev = np.eye(2)
            state = AcrobotReducedState(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1_prev,
                F2_prev=F2_prev,
            )
            u_value = float(u[0]) if len(u) else float("nan")
        else:
            state = AcrobotReducedState(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1[k - 1],
                F2_prev=F2[k - 1],
            )
            u_value = float(u[min(k - 1, len(u) - 1)]) if len(u) else float("nan")

        row = state_to_row(state, params=params, time_value=float(t[k]), k=k, u_value=u_value)
        if X.ndim == 2 and X.shape[0] > k and X.shape[1] >= 4:
            row["x1"] = float(X[k, 0])
            row["y1"] = float(X[k, 1])
            row["x2"] = float(X[k, 2])
            row["y2"] = float(X[k, 3])
        else:
            row["x1"] = float("nan")
            row["y1"] = float("nan")
            row["x2"] = float("nan")
            row["y2"] = float("nan")
        row.update(target_errors_from_row(row, params))
        rows.append(row)

    return rows


def write_simulation_log(
    sim: Mapping[str, Any],
    final_state: AcrobotReducedState,
    params: Mapping[str, Any],
    run_dir: Path,
    mpc_iteration: Optional[int] = None,
    sdp_initial_next: Optional[Dict[str, float]] = None,
    initial_state: Optional[AcrobotReducedState] = None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = simulation_to_rows(sim, params, initial_state=initial_state)
    write_csv(run_dir / "simulation_trajectory.csv", rows)

    # Save arrays for separate plotting/debugging scripts.
    np.savez_compressed(
        run_dir / "simulation_arrays.npz",
        t=np.asarray(sim.get("t", []), dtype=float),
        X=np.asarray(sim.get("X", []), dtype=float),
        R1=np.asarray(sim.get("R1", []), dtype=float),
        R2=np.asarray(sim.get("R2", []), dtype=float),
        F1=np.asarray(sim.get("F1", []), dtype=float),
        F2=np.asarray(sim.get("F2", []), dtype=float),
        thetaR=np.asarray(sim.get("thetaR", []), dtype=float),
        thetaF=np.asarray(sim.get("thetaF", []), dtype=float),
        u=np.asarray(sim.get("u", []), dtype=float),
        residual_inf=np.asarray(sim.get("residual_inf", []), dtype=float),
        solver_success=np.asarray(sim.get("solver_success", []), dtype=bool),
        accepted_by_residual=np.asarray(
            sim.get("accepted_by_residual", []), dtype=bool
        ),
    )

    final_row = state_to_row(
        final_state,
        params=params,
        time_value=float(rows[-1]["t"]) if rows else 0.0,
        k=int(rows[-1]["k"]) if rows else 0,
        u_value=float(rows[-1]["u"]) if rows else float("nan"),
    )
    final_row.update(target_errors_from_row(final_row, params))

    accepted_mask = np.asarray(sim.get("accepted_by_residual", []), dtype=bool)
    residuals = np.asarray(sim.get("residual_inf", []), dtype=float)
    accepted_steps = np.flatnonzero(accepted_mask)
    accepted_residuals_log = [
        {"local_step": int(k), "residual_inf": float(residuals[k])}
        for k in accepted_steps
    ]
    if accepted_steps.size:
        accepted_residuals = residuals[accepted_steps]
        max_accepted_offset = int(np.argmax(accepted_residuals))
        max_accepted_residual = float(accepted_residuals[max_accepted_offset])
        max_accepted_residual_step: Optional[int] = int(
            accepted_steps[max_accepted_offset]
        )
    else:
        max_accepted_residual = 0.0
        max_accepted_residual_step = None

    summary = {
        "mpc_iteration": mpc_iteration,
        "n_substeps": int(len(sim.get("u", []))),
        "n_nodes": int(len(sim.get("t", []))),
        "max_residual_inf": float(np.max(np.asarray(sim.get("residual_inf", [0.0]), dtype=float))),
        "lgvi_accepted_failure_count": int(accepted_steps.size),
        "max_accepted_residual": max_accepted_residual,
        "max_accepted_residual_local_step": max_accepted_residual_step,
        "accepted_residuals": accepted_residuals_log,
        "hard_failure_occurred": False,
        "final_state": final_row,
        "next_sdp_initial": sdp_initial_next or {},
    }

    if sdp_initial_next:
        default_max_step = float(
            params.get(
                "max_step_angle",
                np.deg2rad(float(params.get("max_step_angle_deg", 20.0))),
            )
        )
        max_step1 = float(params.get("max_step_angle1", default_max_step))
        max_step2 = float(params.get("max_step_angle2", default_max_step))
        theta1 = float(sdp_initial_next["thetaF1_prev"])
        theta2 = float(sdp_initial_next["thetaF2_prev"])
        summary["thetaF1_next_sdp_F0_rad"] = theta1
        summary["thetaF1_next_sdp_F0_deg"] = float(np.rad2deg(theta1))
        summary["thetaF2_next_sdp_F0_rad"] = theta2
        summary["thetaF2_next_sdp_F0_deg"] = float(np.rad2deg(theta2))
        summary["max_step_angle_deg_link1"] = float(np.rad2deg(max_step1))
        summary["max_step_angle_deg_link2"] = float(np.rad2deg(max_step2))
        summary["next_sdp_initial_f1_within_max_step"] = bool(
            abs(theta1) <= max_step1 + 1.0e-12
        )
        summary["next_sdp_initial_f2_within_max_step"] = bool(
            abs(theta2) <= max_step2 + 1.0e-12
        )
        summary["next_sdp_initial_f_within_max_step"] = bool(
            summary["next_sdp_initial_f1_within_max_step"]
            and summary["next_sdp_initial_f2_within_max_step"]
        )

    try:
        diag = diagnostics_lgvi(model=make_model_from_params(params), sim=dict(sim))
        summary["max_constraint_norm"] = float(np.max(diag.get("phi_norm", np.array([np.nan]))))
    except Exception:
        summary["max_constraint_norm"] = float("nan")

    with open(run_dir / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    with open(run_dir / "simulation_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("SIMULATION LOG\n")
        f.write("=" * 80 + "\n")
        if mpc_iteration is not None:
            f.write(f"MPC iteration: {mpc_iteration}\n")
        f.write(f"n_substeps: {summary['n_substeps']}\n")
        f.write(f"n_nodes: {summary['n_nodes']}\n")
        f.write(f"max residual inf: {summary['max_residual_inf']:.12e}\n")
        f.write(f"max constraint norm: {summary['max_constraint_norm']:.12e}\n")
        f.write(
            "LGVI solve failures accepted by residual tolerance: "
            f"{summary['lgvi_accepted_failure_count']}\n"
        )
        f.write(
            f"maximum accepted residual: {summary['max_accepted_residual']:.12e}\n"
        )
        f.write(
            "local simulation step of maximum accepted residual: "
            f"{summary['max_accepted_residual_local_step']}\n"
        )
        f.write(f"hard failure occurred: {summary['hard_failure_occurred']}\n")
        if accepted_steps.size:
            f.write("\nWARNING: accepted near-converged LGVI solves:\n")
            for local_step in accepted_steps:
                f.write(
                    f"  local step {int(local_step)}: solver success=False, "
                    f"residual_inf={residuals[local_step]:.12e}\n"
                )
        f.write("\nFinal state:\n")
        for key, val in final_row.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nF0 generated for next SDP iteration:\n")
        if mpc_iteration is not None:
            f.write(
                "  This value will be used as F0 at MPC iteration "
                f"{mpc_iteration + 1}.\n"
            )
        if sdp_initial_next:
            f.write(
                "  thetaF1_next_sdp_F0_rad: "
                f"{summary['thetaF1_next_sdp_F0_rad']}\n"
            )
            f.write(
                "  thetaF1_next_sdp_F0_deg: "
                f"{summary['thetaF1_next_sdp_F0_deg']}\n"
            )
            f.write(
                "  thetaF2_next_sdp_F0_rad: "
                f"{summary['thetaF2_next_sdp_F0_rad']}\n"
            )
            f.write(
                "  thetaF2_next_sdp_F0_deg: "
                f"{summary['thetaF2_next_sdp_F0_deg']}\n"
            )
            f.write(
                "  f0_history_method: "
                f"{sdp_initial_next.get('f0_history_method')}\n"
            )
            f.write(
                "  f0_history_duration: "
                f"{sdp_initial_next.get('f0_history_duration')} s\n"
            )
            f.write(
                "  f0_history_scale: "
                f"{sdp_initial_next.get('f0_history_scale')}\n"
            )
            f.write(
                "  max_step_angle_deg_link1: "
                f"{summary['max_step_angle_deg_link1']}\n"
            )
            f.write(
                "  max_step_angle_deg_link2: "
                f"{summary['max_step_angle_deg_link2']}\n"
            )
            f.write(
                "  next_sdp_initial_f1_within_max_step: "
                f"{summary['next_sdp_initial_f1_within_max_step']}\n"
            )
            f.write(
                "  next_sdp_initial_f2_within_max_step: "
                f"{summary['next_sdp_initial_f2_within_max_step']}\n"
            )
        f.write("\nRaw next SDP initial values:\n")
        for key, val in (sdp_initial_next or {}).items():
            f.write(f"  {key}: {val}\n")

    return summary


def write_simulation_hard_failure_log(
    run_dir: Path,
    mpc_iteration: Optional[int],
    exc: LGVISolveError,
) -> None:
    # Persist LGVI hard-failure diagnostics before propagating the exception
    run_dir.mkdir(parents=True, exist_ok=True)
    accepted = exc.accepted_failures_before_hard_failure
    if accepted:
        max_step, max_residual = max(accepted, key=lambda item: item[1])
    else:
        max_step, max_residual = None, 0.0

    summary = {
        "mpc_iteration": mpc_iteration,
        "lgvi_accepted_failure_count": len(accepted),
        "accepted_residuals": [
            {"local_step": int(step), "residual_inf": float(residual)}
            for step, residual in accepted
        ],
        "max_accepted_residual": float(max_residual),
        "max_accepted_residual_local_step": max_step,
        "hard_failure_occurred": True,
        "hard_failure_local_step": exc.local_sim_step,
        "hard_failure_residual_inf": exc.residual_inf,
        "hard_failure_nfev": exc.nfev,
        "hard_failure_message": exc.solver_message,
    }
    with open(run_dir / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    with open(run_dir / "simulation_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("SIMULATION LOG - HARD FAILURE\n")
        f.write("=" * 80 + "\n")
        if mpc_iteration is not None:
            f.write(f"MPC iteration: {mpc_iteration}\n")
        f.write(
            "LGVI solve failures accepted by residual tolerance before failure: "
            f"{len(accepted)}\n"
        )
        f.write(f"maximum accepted residual: {max_residual:.12e}\n")
        f.write(f"local simulation step where it occurred: {max_step}\n")
        f.write("hard failure occurred: True\n")
        f.write(f"hard failure local simulation step: {exc.local_sim_step}\n")
        f.write(f"hard failure residual_inf: {exc.residual_inf:.12e}\n")
        f.write(f"hard failure nfev: {exc.nfev}\n")
        f.write(f"hard failure message: {exc.solver_message}\n")
        if accepted:
            f.write("\nWARNING: accepted near-converged LGVI solves before failure:\n")
            for local_step, residual in accepted:
                f.write(
                    f"  local step {int(local_step)}: solver success=False, "
                    f"residual_inf={float(residual):.12e}\n"
                )


def simulate_and_log_control(
    params: Mapping[str, Any],
    model: Any,
    state: AcrobotReducedState,
    u_value: float,
    run_dir: Path,
    mpc_iteration: Optional[int] = None,
    rotation_history: Optional[list[tuple[float, np.ndarray, np.ndarray]]] = None,
    interval_start_time: Optional[float] = None,
) -> Tuple[AcrobotReducedState, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    
    # Apply one MPC control input, log the simulation, and return the next SDP initial data.
    
    try:
        interval_start_state = AcrobotReducedState(
            R1=state.R1.copy(), R2=state.R2.copy(),
            F1_prev=state.F1_prev.copy(), F2_prev=state.F2_prev.copy(),
        )
        final_state, sim_raw = simulate_one_control_interval_from_params(
            params=params,
            model=model,
            state=state,
            u_j=float(u_value),
            root_tol=1e-10,
            lgvi_maxfev=int(params.get("lgvi_maxfev", 2000)),
            normalized=False,
            accept_residual=bool(params.get("accept_residual", True)),
            accept_residual_tol=float(params.get("accept_residual_tol", 1.0e-3)),
        )
        sim = dict(sim_raw)
    except LGVISolveError as exc:
        write_simulation_hard_failure_log(run_dir, mpc_iteration, exc)
        raise

    dt_sdp = float(params["dt_sdp"])
    if rotation_history is not None:
        if interval_start_time is None:
            raise ValueError(
                "interval_start_time is required when rotation_history is supplied"
            )
        _append_fine_rotation_history(
            rotation_history=rotation_history,
            sim=sim,
            interval_start_time=float(interval_start_time),
        )
        current_time = float(interval_start_time) + float(sim["t"][-1])
        history_reference_state, history_duration, history_method, history_target_time = (
            _select_sdp_history_reference(
                rotation_history=rotation_history,
                current_time=current_time,
                dt_sdp=dt_sdp,
                thetaF1_initial_sdp=float(params.get("thetaF1_0", 0.0)),
                thetaF2_initial_sdp=float(params.get("thetaF2_0", 0.0)),
            )
        )
    else:
        # Compatibility for non-MPC callers.  The main MPC path always supplies
        # the fine rotation history.
        history_reference_state = interval_start_state
        history_duration = dt_sdp
        history_method = "legacy_interval_reference"
        history_target_time = None

    sdp_initial_next = convert_state_to_sdp_initial_scalars(
        state=final_state,
        model=model,
        dt_physical=float(params["dt_sim"]),
        dt_sdp=dt_sdp,
        interval_start_state=history_reference_state,
        history_duration=history_duration,
        history_method=history_method,
        history_target_time=history_target_time,
    )
    sim["initial_F1_prev"] = interval_start_state.F1_prev.copy()
    sim["initial_F2_prev"] = interval_start_state.F2_prev.copy()

    summary = write_simulation_log(
        sim=sim,
        final_state=final_state,
        params=params,
        run_dir=run_dir,
        mpc_iteration=mpc_iteration,
        sdp_initial_next=sdp_initial_next,
        initial_state=interval_start_state,
    )

    return final_state, sim, sdp_initial_next, summary


# -----------------------------------------------------------------------------
# Final MPC plotting and GIF
# -----------------------------------------------------------------------------


def plot_mpc_results(history_rows: list[Mapping[str, float]], params: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history_rows:
        return

    t = np.array([row["t"] for row in history_rows], dtype=float)
    u = np.array([row["u"] for row in history_rows], dtype=float)
    thetaR1 = np.array([row["thetaR1_deg"] for row in history_rows], dtype=float)
    thetaR2 = np.array([row["thetaR2_deg"] for row in history_rows], dtype=float)
    thetaF1 = np.array([row["thetaF1_deg"] for row in history_rows], dtype=float)
    thetaF2 = np.array([row["thetaF2_deg"] for row in history_rows], dtype=float)
    target_norm = np.array([row["target_angle_error_norm_deg"] for row in history_rows], dtype=float)

    R_orth = {
        "R1": np.array([row["R1_orth_error"] for row in history_rows], dtype=float),
        "R2": np.array([row["R2_orth_error"] for row in history_rows], dtype=float),
        "F1": np.array([row["F1_orth_error"] for row in history_rows], dtype=float),
        "F2": np.array([row["F2_orth_error"] for row in history_rows], dtype=float),
    }

    plt.figure()
    plt.plot(t, u, marker="o")
    plt.xlabel("time [s]")
    plt.ylabel("applied control u_1")
    plt.title("MPC applied controls")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "control_inputs.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, thetaR1, marker="o", label="thetaR1")
    plt.plot(t, thetaR2, marker="o", label="thetaR2")
    plt.axhline(np.rad2deg(float(params["thetaR1_des"])), linestyle="--", label="thetaR1_des")
    plt.axhline(np.rad2deg(float(params["thetaR2_des"])), linestyle=":", label="thetaR2_des")
    plt.xlabel("time [s]")
    plt.ylabel("absolute angle [deg]")
    plt.title("Absolute link angles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "target_position_angles.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, target_norm, marker="o")
    plt.xlabel("time [s]")
    plt.ylabel("norm error to target R_des [deg]")
    plt.title("Target rotation error norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "target_rotation_error_norm.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, thetaF1, marker="o", label="thetaF1 prev")
    plt.plot(t, thetaF2, marker="o", label="thetaF2 prev")
    plt.xlabel("time [s]")
    plt.ylabel("step angle of F [deg]")
    plt.title("Step rotation / velocity proxy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "velocity_F_step_angles.png", dpi=200)
    plt.close()

    plt.figure()
    for name, vals in R_orth.items():
        plt.semilogy(t, np.maximum(vals, 1e-18), marker="o", label=name)
    plt.xlabel("time [s]")
    plt.ylabel("SO(2) orthogonality error")
    plt.title("SO(2) errors for R and F")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "SO2_errors_R_F.png", dpi=200)
    plt.close()

    elbow_x = np.array([row["elbow_x"] for row in history_rows], dtype=float)
    elbow_y = np.array([row["elbow_y"] for row in history_rows], dtype=float)
    tip_x = np.array([row["tip_x"] for row in history_rows], dtype=float)
    tip_y = np.array([row["tip_y"] for row in history_rows], dtype=float)

    plt.figure()
    plt.plot(elbow_x, elbow_y, marker="o", label="elbow")
    plt.plot(tip_x, tip_y, marker="o", label="tip")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Acrobot trajectories")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories_xy.png", dpi=200)
    plt.close()


def create_acrobot_gif(
    history_rows: list[Mapping[str, float]],
    out_path: Path,
    show_trajectory_trace: bool = False,
) -> None:
    if len(history_rows) < 2:
        return

    base_x = np.array([row["base_x"] for row in history_rows], dtype=float)
    base_y = np.array([row["base_y"] for row in history_rows], dtype=float)
    elbow_x = np.array([row["elbow_x"] for row in history_rows], dtype=float)
    elbow_y = np.array([row["elbow_y"] for row in history_rows], dtype=float)
    tip_x = np.array([row["tip_x"] for row in history_rows], dtype=float)
    tip_y = np.array([row["tip_y"] for row in history_rows], dtype=float)

    all_x = np.concatenate([base_x, elbow_x, tip_x])
    all_y = np.concatenate([base_y, elbow_y, tip_y])
    margin = 0.2

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
    ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
    ax.grid(True)
    ax.set_title("MPC-SDP acrobot simulation")

    line, = ax.plot([], [], marker="o", linewidth=3)
    trace = ax.plot([], [], color="orange", linewidth=1)[0] if show_trajectory_trace else None

    def init():
        line.set_data([], [])
        if trace is not None:
            trace.set_data([], [])
            return line, trace
        return (line,)

    def update(frame: int):
        xs = [base_x[frame], elbow_x[frame], tip_x[frame]]
        ys = [base_y[frame], elbow_y[frame], tip_y[frame]]
        line.set_data(xs, ys)
        if trace is not None:
            trace.set_data(tip_x[: frame + 1], tip_y[: frame + 1])
            return line, trace
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(history_rows), init_func=init, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Standalone simulation test
# -----------------------------------------------------------------------------


def main() -> None:
    yaml_path = PROJECT_ROOT / "config" / "acrobot_physical.yaml"
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)

    model, state0 = make_initial_state_from_params(params, h_key="dt_sim")
    u_test = 0.0

    final_state, sim, sdp_next, summary = simulate_and_log_control(
        params=params,
        model=model,
        state=state0,
        u_value=u_test,
        run_dir=SIM_RESULTS_DIR / "standalone_simulation_test",
        mpc_iteration=None,
    )

    print("Standalone simulation finished.")
    print("Results saved to:", SIM_RESULTS_DIR / "standalone_simulation_test")
    print("Next SDP initial:")
    for k, v in sdp_next.items():
        print(f"  {k}: {v:+.12e}")


if __name__ == "__main__":
    main()
