# config/config_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config file.

    Example:
        cfg = load_yaml_config("config/acrobot_physical.yaml")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    return cfg


def _get_block(cfg: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """
    Safely get a YAML block.
    """
    block = cfg.get(key, {})
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise TypeError(f"YAML block '{key}' must be a mapping/dictionary.")
    return dict(block)


def _as_float(block: Mapping[str, Any], key: str, default: float) -> float:
    return float(block.get(key, default))


def _as_int(block: Mapping[str, Any], key: str, default: int) -> int:
    return int(block.get(key, default))


def _as_vec2(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != 2:
        raise ValueError(f"{name} must be a 2-vector, got shape {arr.shape}.")
    return arr


def build_common_params(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert YAML config into one flat params dictionary used by:

        - Numerical_Simulation
        - SDP/solve.py
        - MPC loop

    Important convention:

        thetaR_i:
            absolute link orientation angle of R_i

        thetaF_i:
            step rotation angle of F_i

    Therefore:

        R_i = [[cos(thetaR_i), -sin(thetaR_i)],
               [sin(thetaR_i),  cos(thetaR_i)]]

        F_i = [[cos(thetaF_i), -sin(thetaF_i)],
               [sin(thetaF_i),  cos(thetaF_i)]]
    """

    params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # YAML blocks
    # ------------------------------------------------------------------
    system = _get_block(cfg, "system")
    physical = _get_block(cfg, "physical")
    time = _get_block(cfg, "time")
    initial = _get_block(cfg, "initial")
    target = _get_block(cfg, "target")
    bounds = _get_block(cfg, "bounds")
    sdp = _get_block(cfg, "sdp")
    cost = _get_block(cfg, "cost")
    mpc = _get_block(cfg, "mpc")
    simulation = _get_block(cfg, "simulation")

    # ------------------------------------------------------------------
    # System metadata
    # ------------------------------------------------------------------
    params["system_name"] = system.get("name", "acrobot_so2_absolute")
    params["angle_convention"] = system.get(
        "angle_convention",
        "absolute body-frame SO(2) angles",
    )

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    params["m1"] = _as_float(physical, "m1", 1.0)
    params["m2"] = _as_float(physical, "m2", 1.0)

    params["l1"] = _as_float(physical, "l1", 0.5)
    params["l2"] = _as_float(physical, "l2", 0.5)

    params["deltaJ1"] = _as_float(physical, "deltaJ1", 0.05)
    params["deltaJ2"] = _as_float(physical, "deltaJ2", 0.05)

    params["g"] = _as_float(physical, "g", 9.81)

    # Base position.
    if "p_0" in physical:
        params["p_0"] = _as_vec2(physical["p_0"], "physical.p_0")
    elif "p0" in physical:
        params["p_0"] = _as_vec2(physical["p0"], "physical.p0")
    else:
        params["p_0"] = np.array([0.0, 0.0], dtype=float)

    # Alias, because some older code used p0.
    params["p0"] = params["p_0"].copy()

    # Geometry vectors.
    #
    # Thesis convention:
    #   rho_10  : link-1 COM -> base
    #   rho_112 : link-1 COM -> elbow
    #   rho_212 : link-2 COM -> elbow
    #
    # For uniform rods:
    #   rho_10  = [0,  l1/2]
    #   rho_112 = [0, -l1/2]
    #   rho_212 = [0,  l2/2]
    if "rho_10" in physical:
        params["rho_10"] = _as_vec2(physical["rho_10"], "physical.rho_10")
    else:
        params["rho_10"] = np.array([0.0, 0.5 * params["l1"]], dtype=float)

    if "rho_112" in physical:
        params["rho_112"] = _as_vec2(physical["rho_112"], "physical.rho_112")
    else:
        params["rho_112"] = np.array([0.0, -0.5 * params["l1"]], dtype=float)

    if "rho_212" in physical:
        params["rho_212"] = _as_vec2(physical["rho_212"], "physical.rho_212")
    else:
        params["rho_212"] = np.array([0.0, 0.5 * params["l2"]], dtype=float)

    # ------------------------------------------------------------------
    # Time parameters
    # ------------------------------------------------------------------
    params["dt_sdp"] = _as_float(time, "dt_sdp", 0.1)
    params["dt_sim"] = _as_float(time, "dt_sim", 0.001)
    params["control_interval"] = _as_float(
        time,
        "control_interval",
        params["dt_sdp"],
    )

    # Some SDP code expects params["dt"].
    params["dt"] = params["dt_sdp"]

    # ------------------------------------------------------------------
    # Fine LGVI simulation settings
    # ------------------------------------------------------------------
    params["lgvi_maxfev"] = _as_int(simulation, "lgvi_maxfev", 2000)
    params["accept_residual"] = bool(simulation.get("accept_residual", True))
    params["accept_residual_tol"] = _as_float(
        simulation, "accept_residual_tol", 1.0e-3
    )
    # Master switch for all in-run visual output. Numerical logs are always saved.
    params["plot_results"] = bool(simulation.get("plot_results", False))
    params["generate_mpc_gif"] = bool(
        simulation.get("generate_mpc_gif", params["dt_sim"] > 1.0e-3)
    )
    params["gif_stride"] = _as_int(
        simulation,
        "gif_stride",
        max(1, int(round(0.02 / params["dt_sim"]))),
    )

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    # Old-code convention:
    #
    #   thetaR_i,0 defines R_i at the current physical state.
    #   thetaF_i,0 defines previous/current step rotation F_i,0.
    #
    # The YAML initial step angles define F_1,0 and F_2,0.
    params["thetaR1_0"] = np.deg2rad(_as_float(initial, "thetaR1_deg", 0.0))
    params["thetaR2_0"] = np.deg2rad(_as_float(initial, "thetaR2_deg", 0.0))

    params["thetaF1_0"] = np.deg2rad(_as_float(initial, "thetaF1_deg", 0.0))
    params["thetaF2_0"] = np.deg2rad(_as_float(initial, "thetaF2_deg", 0.0))

    # Direct scalar values for initial/current R.
    params["c1_current"] = float(np.cos(params["thetaR1_0"]))
    params["s1_current"] = float(np.sin(params["thetaR1_0"]))
    params["c2_current"] = float(np.cos(params["thetaR2_0"]))
    params["s2_current"] = float(np.sin(params["thetaR2_0"]))

    # Previous step F from thetaF.
    params["a1_prev"] = float(np.cos(params["thetaF1_0"]))
    params["b1_prev"] = float(np.sin(params["thetaF1_0"]))
    params["a2_prev"] = float(np.cos(params["thetaF2_0"]))
    params["b2_prev"] = float(np.sin(params["thetaF2_0"]))

    # Backward-compatible names.
    #
    # Warning:
    #   In the MPC setup, these c1_0 names mean "current physical state",
    #   which the SDP then places at node 1.
    params["c1_0"] = params["c1_current"]
    params["s1_0"] = params["s1_current"]
    params["c2_0"] = params["c2_current"]
    params["s2_0"] = params["s2_current"]

    # ------------------------------------------------------------------
    # Target state
    # ------------------------------------------------------------------
    params["thetaR1_des"] = np.deg2rad(_as_float(target, "thetaR1_deg", 0.0))
    params["thetaR2_des"] = np.deg2rad(_as_float(target, "thetaR2_deg", 0.0))

    # Desired terminal step rotation from YAML.
    params["thetaF1_des"] = np.deg2rad(_as_float(target, "thetaF1_deg", 0.0))
    params["thetaF2_des"] = np.deg2rad(_as_float(target, "thetaF2_deg", 0.0))

    params["c1_des"] = float(np.cos(params["thetaR1_des"]))
    params["s1_des"] = float(np.sin(params["thetaR1_des"]))
    params["c2_des"] = float(np.cos(params["thetaR2_des"]))
    params["s2_des"] = float(np.sin(params["thetaR2_des"]))

    params["a1_des"] = float(np.cos(params["thetaF1_des"]))
    params["b1_des"] = float(np.sin(params["thetaF1_des"]))
    params["a2_des"] = float(np.cos(params["thetaF2_des"]))
    params["b2_des"] = float(np.sin(params["thetaF2_des"]))

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    params["u_max"] = _as_float(bounds, "u_max", 15.0)
    params["lambda_max"] = _as_float(bounds, "lambda_max", 100.0)

    global_step_deg = _as_float(bounds, "max_step_angle_deg", 20.0)
    step1_deg = float(bounds.get("max_step_angle_deg_link1", global_step_deg))
    step2_deg = float(bounds.get("max_step_angle_deg_link2", global_step_deg))

    # Legacy global value is retained for old callers/configs; constraints and
    # diagnostics below use the link-specific values.
    params["max_step_angle_deg"] = global_step_deg
    params["max_step_angle"] = np.deg2rad(global_step_deg)
    params["max_step_angle1_deg"] = step1_deg
    params["max_step_angle2_deg"] = step2_deg
    params["max_step_angle_deg_link1"] = step1_deg
    params["max_step_angle_deg_link2"] = step2_deg
    params["max_step_angle1"] = np.deg2rad(step1_deg)
    params["max_step_angle2"] = np.deg2rad(step2_deg)

    # SDP inequality uses a >= cos(theta_step_max).
    params["a1_min"] = float(np.cos(params["max_step_angle1"]))
    params["a2_min"] = float(np.cos(params["max_step_angle2"]))

    # ------------------------------------------------------------------
    # SDP parameters
    # ------------------------------------------------------------------
    params["N"] = _as_int(sdp, "horizon_N", 4)

    params["kappa"] = _as_int(
        sdp,
        "relaxation_order",
        _as_int(sdp, "kappa", 2),
    )

    # SPOT options.
    params["relax_mode"] = sdp.get("relax_mode", "SOS")
    params["cs_mode"] = sdp.get("cs_mode", sdp.get("clique_mode", "SELF"))
    params["ts_mode"] = sdp.get("ts_mode", "NON")
    params["ts_mom_mode"] = sdp.get("ts_mom_mode", "NON")
    params["ts_eq_mode"] = sdp.get("ts_eq_mode", "NON")

    params["if_solve"] = bool(sdp.get("if_solve", True))
    params["if_mex"] = bool(sdp.get("if_mex", True))

    # ------------------------------------------------------------------
    # Cost parameters
    # ------------------------------------------------------------------
    params["rho_R"] = _as_float(cost, "rho_R", 100.0)
    params["rho_F"] = _as_float(cost, "rho_F", 5.0)

    params["alpha_R"] = _as_float(cost, "alpha_R", 0.01)
    params["alpha_F"] = _as_float(cost, "alpha_F", 0.005)

    params["gamma"] = _as_float(cost, "gamma", 50.0)

    # Optional lambda regularization.
    params["alpha_lam"] = _as_float(cost, "alpha_lam", 0.0)

    # ------------------------------------------------------------------
    # MPC settings
    # ------------------------------------------------------------------
    params["mpc_max_iterations"] = _as_int(mpc, "max_iterations", 20)
    params["mpc_stop_angle_tol_deg"] = _as_float(mpc, "stop_angle_tol_deg", 2.0)
    params["mpc_stop_step_angle_tol_deg"] = _as_float(mpc, "stop_step_angle_tol_deg", 2.0)
    if "stop_angular_velocity_tol_deg_s" in mpc:
        params["mpc_stop_angular_velocity_tol_deg_s"] = _as_float(
            mpc,
            "stop_angular_velocity_tol_deg_s",
            params["mpc_stop_step_angle_tol_deg"] / params["dt_sdp"],
        )
    params["mpc_stable_steps_required"] = _as_int(mpc, "stable_steps_required", 3)
    params["cleanup_solver_artifacts"] = bool(mpc.get("cleanup_solver_artifacts", True))

    # Solver normalization metadata.  The SDP decision variables u and lambda
    # are normalized to [-1, 1].  The physical/control quantities are obtained as
    #   u_phys        = u_max * u_solver,
    #   lambda_bar    = lambda_max * lambda_solver.
    # With lambda_max=40, lambda_bar lies in [-40, 40], matching the old
    # two-factor scale 4 * 10 without a second scale factor.
    params["u_solver_scale"] = params["u_max"]
    params["lambda_solver_scale"] = params["lambda_max"]
    params["lambda_bar_scale"] = params["lambda_max"]

    # ------------------------------------------------------------------
    # Nonstandard SO(2) rotational inertia
    # ------------------------------------------------------------------
    # The YAML stores only the diagonal value of each nonstandard
    # inertia matrix:
    #
    #   Jd_i = diag([deltaJ_i, deltaJ_i])
    #
    # The reduced scalar rotational coefficient is trace(Jd_i).
    params["Jd1"] = np.diag([params["deltaJ1"], params["deltaJ1"]])
    params["Jd2"] = np.diag([params["deltaJ2"], params["deltaJ2"]])

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if params["N"] < 2:
        raise ValueError("SDP horizon_N must be at least 2 because u_k starts at k=1.")

    if params["dt_sdp"] <= 0.0:
        raise ValueError("dt_sdp must be positive.")

    if params["dt_sim"] <= 0.0:
        raise ValueError("dt_sim must be positive.")

    if params["control_interval"] <= 0.0:
        raise ValueError("control_interval must be positive.")

    if params["lgvi_maxfev"] < 1:
        raise ValueError("simulation.lgvi_maxfev must be at least 1.")

    if params["accept_residual_tol"] < 0.0:
        raise ValueError("simulation.accept_residual_tol must be nonnegative.")

    if params["gif_stride"] < 1:
        raise ValueError("simulation.gif_stride must be at least 1.")

    return params


def print_params_summary(params: Mapping[str, Any]) -> None:
    """
    Small debug helper.
    """
    print("=" * 80)
    print("COMMON PARAMS SUMMARY")
    print("=" * 80)

    keys = [
        "m1",
        "m2",
        "l1",
        "l2",
        "deltaJ1",
        "deltaJ2",
        "g",
        "dt_sdp",
        "dt_sim",
        "control_interval",
        "thetaR1_0",
        "thetaR2_0",
        "thetaF1_0",
        "thetaF2_0",
        "thetaR1_des",
        "thetaR2_des",
        "thetaF1_des",
        "thetaF2_des",
        "N",
        "u_max",
        "lambda_max",
        "max_step_angle1_deg",
        "max_step_angle2_deg",
        "a1_min",
        "a2_min",
        "rho_R",
        "rho_F",
        "alpha_R",
        "alpha_F",
        "gamma",
    ]

    for key in keys:
        if key in params:
            value = params[key]
            if isinstance(value, float):
                print(f"{key:20s}: {value:+.12e}")
            else:
                print(f"{key:20s}: {value}")

    print("rho_10             :", params["rho_10"])
    print("rho_112            :", params["rho_112"])
    print("rho_212            :", params["rho_212"])
    print("Jd1                :", params["Jd1"])
    print("Jd2                :", params["Jd2"])
    print("=" * 80)
