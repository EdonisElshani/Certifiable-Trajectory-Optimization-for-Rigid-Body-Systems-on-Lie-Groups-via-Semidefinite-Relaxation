from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config.config_loader import load_yaml_config, build_common_params
from SDP.solve import solve_sdp
from Numerical_Simulation.solver_lgvi_acrobot import make_model_from_params, make_initial_state_from_params

from open_loop_sdp import SDP_RESULTS_DIR, write_sdp_run_logs
from simulation import (
    MPC_RESULTS_DIR,
    SIM_RESULTS_DIR,
    simulate_and_log_control,
    simulation_to_rows,
    state_to_row,
    target_errors_from_row,
    write_csv,
    plot_mpc_results,
    create_acrobot_gif,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Results"


# -----------------------------------------------------------------------------
# Helpers
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


def get_mpc_settings(params: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    
    # Read MPC settings from YAML.

    mpc_cfg = cfg.get("mpc", {}) if isinstance(cfg.get("mpc", {}), Mapping) else {}

    stop_step_angle_tol_deg = float(
        params.get(
            "mpc_stop_step_angle_tol_deg",
            mpc_cfg.get("stop_step_angle_tol_deg", 2.0),
        )
    )
    dt_sdp = float(params["dt_sdp"])
    if dt_sdp <= 0.0:
        raise ValueError("dt_sdp must be positive")
    stop_angular_velocity_tol_deg_s = float(
        params.get(
            "mpc_stop_angular_velocity_tol_deg_s",
            mpc_cfg.get(
                "stop_angular_velocity_tol_deg_s",
                stop_step_angle_tol_deg / dt_sdp,
            ),
        )
    )
    if stop_angular_velocity_tol_deg_s <= 0.0:
        raise ValueError("mpc.stop_angular_velocity_tol_deg_s must be positive")

    return {
        "max_iterations": int(params.get("mpc_max_iterations", mpc_cfg.get("max_iterations", 20))),
        "stop_angle_tol_deg": float(
            params.get("mpc_stop_angle_tol_deg", mpc_cfg.get("stop_angle_tol_deg", 2.0))
        ),
        "stop_step_angle_tol_deg": stop_step_angle_tol_deg,
        "stop_angular_velocity_tol_deg_s": stop_angular_velocity_tol_deg_s,
        "stable_steps_required": int(
            params.get(
                "mpc_stable_steps_required",
                mpc_cfg.get("stable_steps_required", 3),
            )
        ),
        "cleanup_solver_artifacts": bool(
            params.get(
                "cleanup_solver_artifacts",
                mpc_cfg.get("cleanup_solver_artifacts", True),
            )
        ),
    }


def _status_text(x: Any) -> str:
    return str(x).upper() if x is not None else ""


def _is_infeasible_status(problem_status: Any, solution_status: Any) -> bool:
    statuses = [_status_text(problem_status), _status_text(solution_status)]
    return any("INFEASIBLE" in status for status in statuses)


def _is_unknown_status(problem_status: Any, solution_status: Any) -> bool:
    statuses = [_status_text(problem_status), _status_text(solution_status)]
    return any("UNKNOWN" in status for status in statuses)


def _is_finite_control(u: Any) -> bool:
    try:
        return u is not None and np.isfinite(float(u))
    except Exception:
        return False


def write_mpc_summary(
    out_dir: Path,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    history_rows: list[Mapping[str, float]],
    sdp_summaries: list[Mapping[str, Any]],
    sim_summaries: list[Mapping[str, Any]],
    stopped: bool,
    stop_reason: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "mpc_history.csv", history_rows)

    final_state = history_rows[-1] if history_rows else {}
    stable_counter = int(final_state.get("stable_counter", 0))
    final_angle_error = final_state.get("target_angle_error_norm_deg")
    final_step_error = final_state.get("target_step_error_norm_deg")
    final_angular_velocity_error = final_state.get(
        "target_angular_velocity_error_norm_deg_s"
    )
    if "extraction_failure" in stop_reason:
        termination = "extraction_failure"
    elif "infeasible solver status" in stop_reason:
        termination = "sdp_failure"
    elif stopped:
        termination = "stabilization"
    elif stop_reason == "max_iterations reached":
        termination = "max_iterations"
    else:
        termination = "running"

    compact = {
        "stopped": stopped,
        "termination": termination,
        "stop_reason": stop_reason,
        "num_mpc_iterations": max(0, len(history_rows) - 1),
        "stable_counter": stable_counter,
        "stable_steps_required": int(settings["stable_steps_required"]),
        "stop_angle_tol_deg": float(settings["stop_angle_tol_deg"]),
        "stop_step_angle_tol_deg": float(settings["stop_step_angle_tol_deg"]),
        "stop_angular_velocity_tol_deg_s": float(
            settings["stop_angular_velocity_tol_deg_s"]
        ),
        "target_angle_error_norm_deg": final_angle_error,
        "target_step_error_norm_deg": final_step_error,
        "target_angular_velocity_error_norm_deg_s": final_angular_velocity_error,
        "settings": _json_safe(settings),
        "params": _json_safe({k: v for k, v in params.items() if not callable(v)}),
        "sdp_summaries": _json_safe(sdp_summaries),
        "simulation_summaries": _json_safe(sim_summaries),
        "final_state": _json_safe(final_state),
    }

    with open(out_dir / "mpc_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    with open(out_dir / "mpc_readable_log.txt", "w", encoding="utf-8") as f:
        f.write("MPC-SDP FINAL RESULT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Stopped: {stopped}\n")
        f.write(f"Termination: {termination}\n")
        f.write(f"Reason: {stop_reason}\n")
        f.write(f"Iterations: {max(0, len(history_rows) - 1)}\n")
        f.write(f"Stable counter: {stable_counter}\n")
        f.write(f"Stable steps required: {int(settings['stable_steps_required'])}\n")
        f.write(f"Final target angle error norm: {final_angle_error} deg\n")
        f.write(f"Final target step error norm: {final_step_error} deg\n")
        f.write(
            "Final target angular velocity error norm: "
            f"{final_angular_velocity_error} deg/s\n"
        )
        f.write("\nMPC settings:\n")
        f.write(f"  mpc_max_iterations: {int(settings['max_iterations'])}\n")
        f.write(f"  mpc_stop_angle_tol_deg: {float(settings['stop_angle_tol_deg'])}\n")
        f.write(
            "  mpc_stop_step_angle_tol_deg: "
            f"{float(settings['stop_step_angle_tol_deg'])}\n"
        )
        f.write(
            "  mpc_stop_angular_velocity_tol_deg_s: "
            f"{float(settings['stop_angular_velocity_tol_deg_s'])}\n"
        )
        f.write(
            "  mpc_stable_steps_required: "
            f"{int(settings['stable_steps_required'])}\n"
        )
        f.write(
            "  cleanup_solver_artifacts: "
            f"{bool(settings['cleanup_solver_artifacts'])}\n"
        )
        f.write("\nApplied controls:\n")
        for row in history_rows[1:]:
            f.write(
                f"  j={int(row['mpc_iteration'])}: "
                f"t={row['t']:.6f}, u_1={row['u']:+.12e}, "
                f"thetaR1={row['thetaR1_deg']:+.6f} deg, "
                f"thetaR2={row['thetaR2_deg']:+.6f} deg, "
                f"target_norm={row['target_angle_error_norm_deg']:.6f} deg\n"
            )
        f.write("\nPer-iteration diagnostics:\n")
        for idx, sdp_summary in enumerate(sdp_summaries):
            f.write(f"  MPC iteration {idx}:\n")
            f.write(f"    problem_status: {sdp_summary.get('problem_status', 'UNKNOWN')}\n")
            f.write(f"    solution_status: {sdp_summary.get('solution_status', 'UNKNOWN')}\n")
            f.write(
                "    infeasible_status: "
                f"{sdp_summary.get('infeasible_status', False)}\n"
            )
            f.write(
                "    unknown_status: "
                f"{sdp_summary.get('unknown_status', False)}\n"
            )
            f.write(
                "    extraction_attempted: "
                f"{sdp_summary.get('extraction_attempted', False)}\n"
            )
            f.write(
                "    extraction_success: "
                f"{sdp_summary.get('extraction_success', False)}\n"
            )
            f.write(
                "    accepted_with_unknown_status: "
                f"{sdp_summary.get('accepted_with_unknown_status', False)}\n"
            )
            f.write(f"    control_applied: {sdp_summary.get('control_applied', False)}\n")
            if sdp_summary.get("extraction_success", False):
                f.write(
                    "    first_block_rank_proxy: "
                    f"{sdp_summary.get('first_block_rank_proxy')}\n"
                )
                f.write(
                    "    first_block_dominant_weight: "
                    f"{sdp_summary.get('first_block_dominant_weight')}\n"
                )
                f.write(
                    "    first_step_SO2_error: "
                    f"{sdp_summary.get('first_step_SO2_error')}\n"
                )
                f.write(
                    "    first_step_kinematic_error: "
                    f"{sdp_summary.get('first_step_kinematic_error')}\n"
                )
                f.write(
                    "    max_SO2_error: "
                    f"{sdp_summary.get('max_so2_error')}\n"
                )
                f.write(
                    "    max_kinematic_error: "
                    f"{sdp_summary.get('max_kinematic_error')}\n"
                )
                for key in [
                    "max_dynamic_residual",
                    "max_dynamic_residual_inf",
                    "dynamic_residual_inf",
                ]:
                    if key in sdp_summary:
                        f.write(f"    {key}: {sdp_summary.get(key)}\n")
            f0_diag = sdp_summary.get("f0_step_bound_diagnostic", {})
            if f0_diag:
                thetaF1_current_rad = f0_diag.get(
                    "thetaF1_current_sdp_F0_rad",
                    f0_diag.get("thetaF1_prev_rad"),
                )
                thetaF1_current_deg = f0_diag.get(
                    "thetaF1_current_sdp_F0_deg",
                    f0_diag.get("thetaF1_prev_deg"),
                )
                thetaF2_current_rad = f0_diag.get(
                    "thetaF2_current_sdp_F0_rad",
                    f0_diag.get("thetaF2_prev_rad"),
                )
                thetaF2_current_deg = f0_diag.get(
                    "thetaF2_current_sdp_F0_deg",
                    f0_diag.get("thetaF2_prev_deg"),
                )
                max_step_angle_deg_link1 = f0_diag.get(
                    "max_step_angle_deg_link1",
                    f0_diag.get("max_step_angle1_deg", f0_diag.get("max_step_angle_deg")),
                )
                max_step_angle_deg_link2 = f0_diag.get(
                    "max_step_angle_deg_link2",
                    f0_diag.get("max_step_angle2_deg", f0_diag.get("max_step_angle_deg")),
                )
                f.write("    Fixed F0 used by current SDP iteration:\n")
                f.write(
                    "      thetaF1_current_sdp_F0_rad = "
                    f"{thetaF1_current_rad}\n"
                )
                f.write(
                    "      thetaF1_current_sdp_F0_deg = "
                    f"{thetaF1_current_deg}\n"
                )
                f.write(
                    "      thetaF2_current_sdp_F0_rad = "
                    f"{thetaF2_current_rad}\n"
                )
                f.write(
                    "      thetaF2_current_sdp_F0_deg = "
                    f"{thetaF2_current_deg}\n"
                )
                f.write(
                    "      max_step_angle_deg_link1 = "
                    f"{max_step_angle_deg_link1}\n"
                )
                f.write(
                    "      max_step_angle_deg_link2 = "
                    f"{max_step_angle_deg_link2}\n"
                )
                f.write(
                    "      F1_0_step_bound_warning = "
                    f"{f0_diag.get('F1_0_step_bound_warning')}\n"
                )
                f.write(
                    "      F2_0_step_bound_warning = "
                    f"{f0_diag.get('F2_0_step_bound_warning')}\n"
                )
                f.write(
                    "      F0_step_bound_warning = "
                    f"{f0_diag.get('F0_step_bound_warning')}\n"
                )
                f.write(
                    "      Note: F0 is fixed from measured rotation history on the "
                    "dt_sdp time scale and is not constrained by the future "
                    "step-angle bound.\n"
                )
            if idx < len(sim_summaries):
                sim_summary = sim_summaries[idx]
                nxt = sim_summary.get("next_sdp_initial", {})
                thetaF1_next_rad = sim_summary.get(
                    "thetaF1_next_sdp_F0_rad",
                    nxt.get("thetaF1_prev"),
                )
                thetaF2_next_rad = sim_summary.get(
                    "thetaF2_next_sdp_F0_rad",
                    nxt.get("thetaF2_prev"),
                )
                thetaF1_next_deg = sim_summary.get(
                    "thetaF1_next_sdp_F0_deg",
                    None if thetaF1_next_rad is None else float(np.rad2deg(thetaF1_next_rad)),
                )
                thetaF2_next_deg = sim_summary.get(
                    "thetaF2_next_sdp_F0_deg",
                    None if thetaF2_next_rad is None else float(np.rad2deg(thetaF2_next_rad)),
                )
                f.write("    F0 generated for next SDP iteration:\n")
                f.write(
                    "      This value will be used as F0 at MPC iteration "
                    f"{idx + 1}.\n"
                )
                f.write(
                    "      thetaF1_next_sdp_F0_rad = "
                    f"{thetaF1_next_rad}\n"
                )
                f.write(
                    "      thetaF1_next_sdp_F0_deg = "
                    f"{thetaF1_next_deg}\n"
                )
                f.write(
                    "      thetaF2_next_sdp_F0_rad = "
                    f"{thetaF2_next_rad}\n"
                )
                f.write(
                    "      thetaF2_next_sdp_F0_deg = "
                    f"{thetaF2_next_deg}\n"
                )
                f.write(
                    "      f0_history_method = "
                    f"{nxt.get('f0_history_method')}\n"
                )
                f.write(
                    "      f0_history_duration = "
                    f"{nxt.get('f0_history_duration')} s\n"
                )
                f.write(
                    "      f0_history_scale = "
                    f"{nxt.get('f0_history_scale')}\n"
                )
                f.write(
                    "      thetaF_last_fine comparison only, not the next SDP F0: "
                    f"[{nxt.get('thetaF1_last_fine')}, {nxt.get('thetaF2_last_fine')}] rad\n"
                )
                f.write(
                    "      next SDP initial F1 satisfies link-1 max-step bound: "
                    f"{sim_summary.get('next_sdp_initial_f1_within_max_step')}\n"
                )
                f.write(
                    "      next SDP initial F2 satisfies link-2 max-step bound: "
                    f"{sim_summary.get('next_sdp_initial_f2_within_max_step')}\n"
                )
                f.write(
                    "      next SDP initial F satisfies max-step bound: "
                    f"{sim_summary.get('next_sdp_initial_f_within_max_step')}\n"
                )
                for accepted in sim_summary.get("accepted_residuals", []):
                    f.write(
                        "    accepted LGVI residual: "
                        f"local_step={accepted['local_step']}, "
                        f"residual_inf={accepted['residual_inf']:.12e}\n"
                    )
            elif sdp_summary.get("failure_reason"):
                f.write(f"    rejection reason: {sdp_summary['failure_reason']}\n")
            elif sdp_summary.get("rejection_reason"):
                f.write(f"    rejection reason: {sdp_summary['rejection_reason']}\n")
        f.write("\nFinal state:\n")
        if history_rows:
            for key, val in history_rows[-1].items():
                f.write(f"  {key}: {val}\n")


# -----------------------------------------------------------------------------
# Main MPC loop
# -----------------------------------------------------------------------------


def run_mpc_sdp(
    yaml_path: str | Path = PROJECT_ROOT / "config" / "acrobot_physical.yaml",
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_yaml_config(yaml_path)
    params = build_common_params(cfg)
    settings = get_mpc_settings(params, cfg)
    if int(settings["stable_steps_required"]) < 1:
        raise ValueError("mpc.stable_steps_required must be at least 1")

    if run_name is None:
        # Salt the timestamp with the SLURM job ID (or PID) so concurrent
        # runs sharing this project directory never collide on run_name.
        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        run_name = datetime.now().strftime("mpc_%Y-%m-%d_%H-%M-%S") + f"_{job_id}"

    mpc_dir = MPC_RESULTS_DIR / run_name
    sdp_parent = SDP_RESULTS_DIR / run_name
    sim_parent = SIM_RESULTS_DIR / run_name

    mpc_dir.mkdir(parents=True, exist_ok=True)
    sdp_parent.mkdir(parents=True, exist_ok=True)
    sim_parent.mkdir(parents=True, exist_ok=True)

    model, state = make_initial_state_from_params(params, h_key="dt_sim")

    history_rows: list[dict[str, float]] = []
    sdp_summaries: list[dict[str, Any]] = []
    sim_summaries: list[dict[str, Any]] = []
    full_trajectory_rows: list[dict[str, Any]] = []

    # Fine measured rotation history used to construct
    # F0 = R(t-dt_sdp).T @ R(t).  Keeping fine nodes also supports a dt_sdp
    # that is not an integer multiple of the control interval.
    rotation_history: list[tuple[float, np.ndarray, np.ndarray]] = [
        (0.0, state.R1.copy(), state.R2.copy())
    ]

    # Initial row before applying any MPC control.
    initial_row = state_to_row(state, params=params, time_value=0.0, k=0, u_value=float("nan"))
    initial_row.update(target_errors_from_row(initial_row, params))
    initial_row["mpc_iteration"] = -1.0
    initial_row["angle_ok"] = False
    initial_row["step_ok"] = False
    initial_row["angular_velocity_ok"] = False
    initial_row["stop_angular_velocity_tol_deg_s"] = float(
        settings["stop_angular_velocity_tol_deg_s"]
    )
    initial_row["stable_counter"] = 0
    history_rows.append(initial_row)

    mpc_initial: Optional[Dict[str, float]] = None
    stopped = False
    stop_reason = "max_iterations reached"
    stable_counter = 0

    for j in range(int(settings["max_iterations"])):
        print("\n" + "=" * 80)
        print(f"MPC iteration {j}")
        print("=" * 80)

        params_sdp = dict(params)
        if mpc_initial is not None:
            params_sdp.update(mpc_initial)

        # Solve SDP
        # write_sdp_run_logs copies compact information into Results and then deletes old artifacts
        out = solve_sdp(params_sdp)

        sdp_run_dir = sdp_parent / f"mpc_{j:04d}"
        sdp_summary = write_sdp_run_logs(
            out,
            run_dir=sdp_run_dir,
            mpc_iteration=j,
            cleanup_solver_artifacts_enabled=bool(settings["cleanup_solver_artifacts"]),
        )
        sdp_summaries.append(sdp_summary)

        problem_status = str(out.get("problem_status", "UNKNOWN"))
        solution_status = str(out.get("solution_status", "UNKNOWN"))
        infeasible_sdp = _is_infeasible_status(problem_status, solution_status)
        unknown_sdp = _is_unknown_status(problem_status, solution_status)
        u1_candidate = out.get("first_control")
        extraction_success = _is_finite_control(u1_candidate)
        if infeasible_sdp or not extraction_success:
            stopped = True
            if infeasible_sdp:
                stop_reason = (
                    f"infeasible solver status at MPC iteration {j}: "
                    f"problem_status={problem_status}, solution_status={solution_status}; "
                    "control was not applied and the next interval was not simulated"
                )
            else:
                stop_reason = (
                    f"extraction_failure at MPC iteration {j}: "
                    "UNKNOWN or non-optimal solver status and no usable extracted control; "
                    f"problem_status={problem_status}, solution_status={solution_status}; "
                    "control was not applied and the next interval was not simulated"
                )
            write_mpc_summary(
                out_dir=mpc_dir, params=params, settings=settings,
                history_rows=history_rows, sdp_summaries=sdp_summaries,
                sim_summaries=sim_summaries, stopped=stopped,
                stop_reason=stop_reason,
            )
            del out
            gc.collect()
            print(stop_reason)
            break
        if unknown_sdp:
            print(
                "UNKNOWN status accepted because extraction produced finite control: "
                f"problem_status={problem_status}, solution_status={solution_status}"
            )

        u_apply = float(u1_candidate)

        del out
        gc.collect()

        sim_run_dir = sim_parent / f"mpc_{j:04d}"
        interval_start_time = j * float(
            params.get("control_interval", params["dt_sdp"])
        )
        state, sim, mpc_initial, sim_summary = simulate_and_log_control(
            params=params,
            model=model,
            state=state,
            u_value=u_apply,
            run_dir=sim_run_dir,
            mpc_iteration=j,
            rotation_history=rotation_history,
            interval_start_time=interval_start_time,
        )
        sim_summaries.append(sim_summary)

        interval_rows = simulation_to_rows(sim, params)
        # Adjacent intervals share their boundary node
        if j > 0:
            interval_rows = interval_rows[1:]
        for sim_row in interval_rows:
            full_row: dict[str, Any] = {
                "global_time": interval_start_time + float(sim_row["t"]),
                "mpc_iteration": j,
                "local_sim_step": int(sim_row["k"]),
                "u_applied": float(sim_row["u"]),
            }
            full_row.update(sim_row)
            full_trajectory_rows.append(full_row)
        write_csv(mpc_dir / "mpc_full_trajectory.csv", full_trajectory_rows)

        current_time = (j + 1) * float(params.get("control_interval", params["dt_sdp"]))
        row = state_to_row(state, params=params, time_value=current_time, k=j + 1, u_value=u_apply)
        row.update(target_errors_from_row(row, params))
        row["mpc_iteration"] = float(j)

        target_angle_error_norm_deg = float(row["target_angle_error_norm_deg"])
        target_angular_velocity_error_norm_deg_s = float(
            row["target_angular_velocity_error_norm_deg_s"]
        )
        stop_angle_tol_deg = float(settings["stop_angle_tol_deg"])
        stop_angular_velocity_tol_deg_s = float(
            settings["stop_angular_velocity_tol_deg_s"]
        )
        stable_steps_required = int(settings["stable_steps_required"])

        angle_ok = target_angle_error_norm_deg < stop_angle_tol_deg
        angular_velocity_ok = (
            target_angular_velocity_error_norm_deg_s
            < stop_angular_velocity_tol_deg_s
        )
        if angle_ok and angular_velocity_ok:
            stable_counter += 1
        else:
            stable_counter = 0

        row["angle_ok"] = angle_ok
        row["step_ok"] = angular_velocity_ok
        row["angular_velocity_ok"] = angular_velocity_ok
        row["stop_angular_velocity_tol_deg_s"] = stop_angular_velocity_tol_deg_s
        row["stable_counter"] = stable_counter
        history_rows.append(row)

        if stable_counter >= stable_steps_required:
            stopped = True
            stop_reason = (
                f"stabilized for {stable_counter} consecutive MPC steps "
                f"(angle error < {stop_angle_tol_deg} deg and "
                f"angular velocity error < "
                f"{stop_angular_velocity_tol_deg_s} deg/s)"
            )

        write_mpc_summary(
            out_dir=mpc_dir,
            params=params,
            settings=settings,
            history_rows=history_rows,
            sdp_summaries=sdp_summaries,
            sim_summaries=sim_summaries,
            stopped=stopped,
            stop_reason=stop_reason if stopped else "running",
        )

        if stopped:
            print("Stopping condition reached.")
            break

        del sim
        gc.collect()

    write_mpc_summary(
        out_dir=mpc_dir,
        params=params,
        settings=settings,
        history_rows=history_rows,
        sdp_summaries=sdp_summaries,
        sim_summaries=sim_summaries,
        stopped=stopped,
        stop_reason=stop_reason,
    )

    if bool(params.get("plot_results", False)):
        plot_mpc_results(history_rows, params=params, out_dir=mpc_dir / "figures")
        if bool(params.get("generate_mpc_gif", float(params["dt_sim"]) > 1.0e-3)):
            gif_rows = full_trajectory_rows if full_trajectory_rows else history_rows
            gif_stride = max(1, int(params.get("gif_stride", 1)))
            gif_rows_strided = gif_rows[::gif_stride]
            if gif_rows_strided and gif_rows_strided[-1] is not gif_rows[-1]:
                gif_rows_strided.append(gif_rows[-1])
            create_acrobot_gif(
                gif_rows_strided,
                out_path=mpc_dir / "acrobot_mpc.gif",
                show_trajectory_trace=False,
            )
        else:
            print(
                "Skipping MPC GIF generation "
                f"(dt_sim={float(params['dt_sim']):g}; enable with "
                "simulation.generate_mpc_gif: true)."
            )
    else:
        print(
            "Skipping all MPC figures and GIF generation "
            "(simulation.plot_results: false)."
        )

    print("\n" + "=" * 80)
    print("MPC-SDP LOOP FINISHED")
    print("=" * 80)
    print(f"Stopped: {stopped}")
    print(f"Reason: {stop_reason}")
    print(f"Stable counter: {stable_counter}/{int(settings['stable_steps_required'])}")
    if history_rows:
        print(
            "Final errors: "
            f"angle={history_rows[-1]['target_angle_error_norm_deg']:.6f} deg, "
            "angular_velocity="
            f"{history_rows[-1]['target_angular_velocity_error_norm_deg_s']:.6f} deg/s"
        )
    print(f"Results folder: {mpc_dir}")

    return {
        "stopped": stopped,
        "stop_reason": stop_reason,
        "results_dir": str(mpc_dir),
        "history_rows": history_rows,
        "sdp_summaries": sdp_summaries,
        "sim_summaries": sim_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "acrobot_physical.yaml",
        help="Path to the YAML config file (default: config/acrobot_physical.yaml).",
    )
    args = parser.parse_args()

    run_mpc_sdp(yaml_path=args.config)


if __name__ == "__main__":
    main()
