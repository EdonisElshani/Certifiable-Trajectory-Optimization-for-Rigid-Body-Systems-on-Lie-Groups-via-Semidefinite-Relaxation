from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import root

try:
    from Numerical_Simulation.lie_group_so2 import (
        F_from_delta,
        F_from_cayley,
        cayley_from_R,
        angle_from_R,
        orth_error_so2,
        det_error_so2,
    )

    from Numerical_Simulation.Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
        AcrobotSO2Model,
    )

except ImportError:
    # Allows running this file from inside Numerical_Simulation.
    from lie_group_so2 import (
        F_from_delta,
        F_from_cayley,
        cayley_from_R,
        angle_from_R,
        orth_error_so2,
        det_error_so2,
    )

    from Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import (
        AcrobotSO2Model,
    )


@dataclass
class AcrobotReducedState:

    R1: np.ndarray
    R2: np.ndarray
    F1_prev: np.ndarray
    F2_prev: np.ndarray

    def __post_init__(self) -> None:
        self.R1 = np.asarray(self.R1, dtype=float).reshape(2, 2)
        self.R2 = np.asarray(self.R2, dtype=float).reshape(2, 2)
        self.F1_prev = np.asarray(self.F1_prev, dtype=float).reshape(2, 2)
        self.F2_prev = np.asarray(self.F2_prev, dtype=float).reshape(2, 2)


# Backward-compatible name.
AcrobotLGVIState = AcrobotReducedState


@dataclass
class LGVIStepInfo:

    success: bool
    residual_inf: float
    nfev: int
    message: str
    accepted_by_residual: bool = False


class LGVISolveError(RuntimeError):

    def __init__(self, residual_inf: float, message: str, nfev: int) -> None:
        self.residual_inf = float(residual_inf)
        self.solver_message = str(message)
        self.nfev = int(nfev)
        self.local_sim_step: Optional[int] = None
        self.accepted_failures_before_hard_failure: List[Tuple[int, float]] = []
        super().__init__(
            "Reduced LGVI one-step solve failed: "
            f"success=False, ||r||_inf={self.residual_inf:.3e}, "
            f"message={self.solver_message}"
        )


def _require_option_b_model(model: AcrobotSO2Model) -> None:

    required = [
        "reduced_step_residual",
        "initial_step_guess",
        "advance_reduced_state",
        "reconstruct_positions_from_rotations",
        "scalars_from_step_rotation",
        "scalars_from_rotation",
        "angles_from_rotations",
    ]

    missing = [name for name in required if not hasattr(model, name)]

    if missing:
        raise AttributeError(
            "AcrobotSO2Model is missing Option-B method(s): "
            + ", ".join(missing)
            + ". Update model_acrobot_so2.py first."
        )


def make_model_from_params(params: Mapping[str, Any]) -> AcrobotSO2Model:

    return AcrobotSO2Model.from_params_dict(params)


def _get_time_step(params: Mapping[str, Any], h_key: str) -> float:

    if h_key in params:
        return float(params[h_key])

    if "time" in params and isinstance(params["time"], Mapping):
        if h_key in params["time"]:
            return float(params["time"][h_key])

    if "dt" in params:
        return float(params["dt"])

    raise KeyError(
        f"Could not find time step '{h_key}', 'time.{h_key}', or fallback key 'dt'."
    )


def make_reduced_state_from_absolute(
    model: AcrobotSO2Model,
    h: float,
    thetaR: np.ndarray,
    thetaF: np.ndarray,
) -> AcrobotReducedState:

    _require_option_b_model(model)

    thetaR = np.asarray(thetaR, dtype=float).reshape(2)
    thetaF = np.asarray(thetaF, dtype=float).reshape(2)

    R1, R2 = model.rotations_from_angles(thetaR[0], thetaR[1])

    F1_prev = F_from_delta(thetaF[0])
    F2_prev = F_from_delta(thetaF[1])

    return AcrobotReducedState(
        R1=R1,
        R2=R2,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
    )


def make_initial_state_from_params(
    params: Mapping[str, Any],
    model: Optional[AcrobotSO2Model] = None,
    h_key: str = "dt_sim",
) -> Tuple[AcrobotSO2Model, AcrobotReducedState]:

    if model is None:
        model = make_model_from_params(params)

    _require_option_b_model(model)

    h = _get_time_step(params, h_key)

    thetaR = np.array(
        [
            float(params["thetaR1_0"]),
            float(params["thetaR2_0"]),
        ],
        dtype=float,
    )

    thetaF_sdp = np.array(
        [
            float(params["thetaF1_0"]),
            float(params["thetaF2_0"]),
        ],
        dtype=float,
    )

    h_ref = float(params.get("dt_sdp", h))
    thetaF_for_h = thetaF_sdp * (h / h_ref)

    state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=thetaR,
        thetaF=thetaF_for_h,
    )

    return model, state


def reconstruct_X_from_R(
    model: AcrobotSO2Model,
    R1: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray:

    return model.reconstruct_positions_from_rotations(R1, R2)


def _normalize_reduced_residual(
    residual: np.ndarray,
    model: AcrobotSO2Model,
) -> np.ndarray:

    residual = np.asarray(residual, dtype=float).reshape(8)

    trans_scale = max(
        1.0,
        abs(model.m1 * model.d1_com),
        abs(model.m2 * model.d1_elbow),
        abs(model.m2 * model.d2_com),
    )

    rot_scale = max(
        1.0,
        abs(model.rot_inertia_1),
        abs(model.rot_inertia_2),
    )

    scale = np.array(
        [
            trans_scale,
            trans_scale,
            trans_scale,
            trans_scale,
            rot_scale,
            rot_scale,
            1.0,
            1.0,
        ],
        dtype=float,
    )

    return residual / scale


def acrobot_reduced_step_residual(
    z: np.ndarray,
    model: AcrobotSO2Model,
    h: float,
    R1_k: np.ndarray,
    R2_k: np.ndarray,
    F1_prev: np.ndarray,
    F2_prev: np.ndarray,
    u_k: float,
    normalized: bool = False,
) -> np.ndarray:

    _require_option_b_model(model)

    residual = model.reduced_step_residual(
        z=z,
        R1_k=R1_k,
        R2_k=R2_k,
        F1_prev=F1_prev,
        F2_prev=F2_prev,
        u_k=u_k,
        h=h,
    )

    if normalized:
        residual = _normalize_reduced_residual(residual, model)

    return residual


def initial_guess_from_previous(
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    previous_z: Optional[np.ndarray] = None,
) -> np.ndarray:

    if previous_z is not None:
        previous_z = np.asarray(previous_z, dtype=float).reshape(8)
        return previous_z.copy()

    return model.initial_step_guess(
        F1_prev=state.F1_prev,
        F2_prev=state.F2_prev,
    )


def _z_from_cayley_y(model: AcrobotSO2Model, y: np.ndarray) -> np.ndarray:
    #Convert 6D Cayley unknowns to old 8D z = [a,b,a,b,lambda]
    y = np.asarray(y, dtype=float).reshape(6)
    a1, b1 = F_from_cayley(float(y[0]))[0, 0], F_from_cayley(float(y[0]))[1, 0]
    a2, b2 = F_from_cayley(float(y[1]))[0, 0], F_from_cayley(float(y[1]))[1, 0]
    return np.array([a1, b1, a2, b2, y[2], y[3], y[4], y[5]], dtype=float)


def _cayley_y_from_z_or_state(state: AcrobotReducedState, z_guess: Optional[np.ndarray] = None) -> np.ndarray:
    #Build a 6D Cayley initial guess from previous step or previous 8D solution
    if z_guess is not None:
        z_guess = np.asarray(z_guess, dtype=float).reshape(8)
        F1_guess = np.array([[z_guess[0], -z_guess[1]], [z_guess[1], z_guess[0]]], dtype=float)
        F2_guess = np.array([[z_guess[2], -z_guess[3]], [z_guess[3], z_guess[2]]], dtype=float)
        return np.array(
            [
                cayley_from_R(F1_guess),
                cayley_from_R(F2_guess),
                z_guess[4],
                z_guess[5],
                z_guess[6],
                z_guess[7],
            ],
            dtype=float,
        )
    return np.array(
        [
            cayley_from_R(state.F1_prev),
            cayley_from_R(state.F2_prev),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def lgvi_one_step(
    model: AcrobotSO2Model,
    h: float,
    state: AcrobotReducedState,
    u_k: float,
    z_guess: Optional[np.ndarray] = None,
    root_tol: float = 1e-10,
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Tuple[AcrobotReducedState, LGVIStepInfo, np.ndarray]:

    _require_option_b_model(model)

    h = float(h)
    u_k = float(u_k)

    y0 = _cayley_y_from_z_or_state(state, z_guess=z_guess)

    def full_residual_from_y(y: np.ndarray) -> np.ndarray:
        z = _z_from_cayley_y(model, y)
        r = model.reduced_step_residual(
            z=z,
            R1_k=state.R1,
            R2_k=state.R2,
            F1_prev=state.F1_prev,
            F2_prev=state.F2_prev,
            u_k=u_k,
            h=h,
        )
        # First six equations are 4 translational + 2 rotational dynamics.
        # The last two SO(2) constraints are identically zero by Cayley.
        return np.asarray(r[:6], dtype=float).reshape(6)

    def fun(y: np.ndarray) -> np.ndarray:
        r = full_residual_from_y(y)
        if normalized:
            # Normalization originally expected 8 equations.  For the Cayley
            # solve, use the first six scalings only.
            r8 = np.r_[r, 0.0, 0.0]
            return _normalize_reduced_residual(r8, model)[:6]
        return r

    # deterministic warm starts 
    starts = [y0]
    offsets = (0.0, 0.05, -0.05, 0.15, -0.15, 0.35, -0.35, 0.75, -0.75, 1.25, -1.25)
    for dq1 in offsets:
        for dq2 in offsets:
            if dq1 == 0.0 and dq2 == 0.0:
                continue
            y_alt = y0.copy()
            y_alt[0] += dq1
            y_alt[1] += dq2
            starts.append(y_alt)
    y_zero = y0.copy()
    y_zero[0] = 0.0
    y_zero[1] = 0.0
    starts.append(y_zero)

    best = None
    best_residual_inf = np.inf
    best_raw_success = False
    best_message = "no solve attempted"
    best_nfev = 0

    for start in starts:
        sol = root(
            fun,
            np.asarray(start, dtype=float).reshape(6),
            method="hybr",
            options={"xtol": root_tol, "maxfev": int(lgvi_maxfev)},
        )
        y_candidate = np.asarray(sol.x, dtype=float).reshape(6)
        residual = fun(y_candidate)
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        if np.isfinite(residual_inf) and residual_inf < best_residual_inf:
            best = y_candidate
            best_residual_inf = residual_inf
            best_raw_success = bool(sol.success)
            best_message = str(sol.message)
            best_nfev = int(sol.nfev)
        if bool(sol.success) or residual_inf <= float(accept_residual_tol):
            break

    if best is None:
        raise LGVISolveError(residual_inf=np.inf, message="all root attempts failed", nfev=0)

    accepted_by_residual = bool(
        not best_raw_success
        and accept_residual
        and np.isfinite(best_residual_inf)
        and best_residual_inf <= float(accept_residual_tol)
    )

    info = LGVIStepInfo(
        success=bool(best_raw_success),
        residual_inf=best_residual_inf,
        nfev=int(best_nfev),
        message=best_message,
        accepted_by_residual=accepted_by_residual,
    )

    if not best_raw_success and not accepted_by_residual:
        raise LGVISolveError(
            residual_inf=best_residual_inf,
            message=best_message,
            nfev=int(best_nfev),
        )

    z = _z_from_cayley_y(model, best)

    R1_next, R2_next, F1_k, F2_k, _, _ = model.advance_reduced_state(
        R1_k=state.R1,
        R2_k=state.R2,
        z=z,
    )

    next_state = AcrobotReducedState(
        R1=R1_next,
        R2=R2_next,
        F1_prev=F1_k,
        F2_prev=F2_k,
    )

    return next_state, info, z


def rollout_lgvi_controls(
    model: AcrobotSO2Model,
    h: float,
    initial_state: AcrobotReducedState,
    u_sequence: np.ndarray,
    root_tol: float = 1e-10,
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Dict[str, Any]:

    _require_option_b_model(model)

    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    num_steps = int(len(u_sequence))

    R1 = np.zeros((num_steps + 1, 2, 2), dtype=float)
    R2 = np.zeros((num_steps + 1, 2, 2), dtype=float)

    F1 = np.zeros((num_steps, 2, 2), dtype=float)
    F2 = np.zeros((num_steps, 2, 2), dtype=float)

    X = np.zeros((num_steps + 1, 4), dtype=float)

    thetaR = np.zeros((num_steps + 1, 2), dtype=float)
    thetaF = np.zeros((num_steps, 2), dtype=float)

    lam0 = np.zeros((num_steps, 2), dtype=float)
    lam12 = np.zeros((num_steps, 2), dtype=float)

    residual_inf = np.zeros(num_steps, dtype=float)

    infos: List[LGVIStepInfo] = []
    z_solutions: List[np.ndarray] = []

    state = initial_state

    R1[0] = state.R1
    R2[0] = state.R2
    X[0] = reconstruct_X_from_R(model, state.R1, state.R2)
    thetaR[0] = model.angles_from_rotations(state.R1, state.R2)

    z_guess: Optional[np.ndarray] = None

    for k, u_k in enumerate(u_sequence):
        try:
            state_next, info, z = lgvi_one_step(
                model=model,
                h=h,
                state=state,
                u_k=float(u_k),
                z_guess=z_guess,
                root_tol=root_tol,
                lgvi_maxfev=lgvi_maxfev,
                normalized=normalized,
                accept_residual=accept_residual,
                accept_residual_tol=accept_residual_tol,
            )
        except LGVISolveError as exc:
            exc.local_sim_step = k
            exc.accepted_failures_before_hard_failure = [
                (i, step_info.residual_inf)
                for i, step_info in enumerate(infos)
                if step_info.accepted_by_residual
            ]
            raise

        F1_k, F2_k, lam0_k, lam12_k = model.unpack_reduced_solution(z)

        R1[k + 1] = state_next.R1
        R2[k + 1] = state_next.R2

        F1[k] = F1_k
        F2[k] = F2_k

        X[k + 1] = reconstruct_X_from_R(model, state_next.R1, state_next.R2)

        thetaR[k + 1] = model.angles_from_rotations(
            state_next.R1,
            state_next.R2,
        )

        thetaF[k, 0] = angle_from_R(F1_k)
        thetaF[k, 1] = angle_from_R(F2_k)

        lam0[k] = lam0_k
        lam12[k] = lam12_k

        residual_inf[k] = info.residual_inf

        infos.append(info)
        z_solutions.append(z)

        # Warm-start next root solve with current solution.
        z_guess = z.copy()
        state = state_next

    return {
        "t": np.arange(num_steps + 1, dtype=float) * float(h),
        "X": X,
        "R1": R1,
        "R2": R2,
        "F1": F1,
        "F2": F2,
        "thetaR": thetaR,
        "thetaF": thetaF,
        "lambda0": lam0,
        "lambda12": lam12,
        "u": u_sequence,
        "residual_inf": residual_inf,
        "solver_success": np.asarray([info.success for info in infos], dtype=bool),
        "accepted_by_residual": np.asarray(
            [info.accepted_by_residual for info in infos], dtype=bool
        ),
        "infos": infos,
        "z_solutions": z_solutions,
        "final_state": state,
    }


def simulate_one_control_interval(
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    u_j: float,
    dt_control: float,
    dt_sim: float,
    root_tol: float = 1e-10,
    lgvi_maxfev: int = 2000,
    normalized: bool = False,
    accept_residual: bool = True,
    accept_residual_tol: float = 1e-3,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:

    _require_option_b_model(model)

    dt_control = float(dt_control)
    dt_sim = float(dt_sim)

    ratio = dt_control / dt_sim
    n_substeps = int(round(ratio))

    if abs(ratio - n_substeps) > 1e-10:
        raise ValueError(
            f"dt_control / dt_sim must be an integer. "
            f"Got {dt_control} / {dt_sim} = {ratio}"
        )

    u_sequence = np.full(n_substeps, float(u_j), dtype=float)

    sim = rollout_lgvi_controls(
        model=model,
        h=dt_sim,
        initial_state=state,
        u_sequence=u_sequence,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
    )

    return sim["final_state"], sim


def simulate_one_control_interval_from_params(
    params: Mapping[str, Any],
    model: AcrobotSO2Model,
    state: AcrobotReducedState,
    u_j: float,
    root_tol: float = 1e-10,
    lgvi_maxfev: Optional[int] = None,
    normalized: bool = False,
    accept_residual: Optional[bool] = None,
    accept_residual_tol: Optional[float] = None,
) -> Tuple[AcrobotReducedState, Dict[str, Any]]:

    # Convenience wrapper using YAML-derived params.

    if "dt_sim" in params:
        dt_sim = float(params["dt_sim"])
    else:
        dt_sim = float(params["time"]["dt_sim"])

    if "control_interval" in params:
        dt_control = float(params["control_interval"])
    else:
        dt_control = float(params["time"].get("control_interval", params["time"]["dt_sdp"]))

    if lgvi_maxfev is None:
        lgvi_maxfev = int(params.get("lgvi_maxfev", 2000))
    if accept_residual is None:
        accept_residual = bool(params.get("accept_residual", True))
    if accept_residual_tol is None:
        accept_residual_tol = float(params.get("accept_residual_tol", 1e-3))

    return simulate_one_control_interval(
        model=model,
        state=state,
        u_j=u_j,
        dt_control=dt_control,
        dt_sim=dt_sim,
        root_tol=root_tol,
        lgvi_maxfev=lgvi_maxfev,
        normalized=normalized,
        accept_residual=accept_residual,
        accept_residual_tol=accept_residual_tol,
    )


def simulate_lgvi_acrobot(
    model: AcrobotSO2Model,
    h: float,
    steps: int,
    alpha0: np.ndarray,
    thetaF0: Optional[np.ndarray] = None,
    u_fun: Optional[Any] = None,
    first_step: str = "reduced",
    root_tol: float = 1e-10,
    maxfev: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:

    if steps < 1:
        raise ValueError("steps must be at least 1")

    if first_step.lower() not in {"reduced", "euler", "rk4"} and verbose:
        print(
            f"[simulate_lgvi_acrobot] first_step='{first_step}' ignored. "
            "Using reduced Option-B initialization."
        )

    if thetaF0 is None:
        thetaF0 = np.zeros(2, dtype=float)

    initial_state = make_reduced_state_from_absolute(
        model=model,
        h=h,
        thetaR=np.asarray(alpha0, dtype=float).reshape(2),
        thetaF=np.asarray(thetaF0, dtype=float).reshape(2),
    )

    if u_fun is None:
        u_sequence = np.zeros(steps, dtype=float)
    else:
        u_sequence = np.array(
            [float(u_fun(k * h)) for k in range(steps)],
            dtype=float,
        )

    sim = rollout_lgvi_controls(
        model=model,
        h=h,
        initial_state=initial_state,
        u_sequence=u_sequence,
        root_tol=root_tol,
        lgvi_maxfev=maxfev,
    )

    if verbose:
        print(
            "[simulate_lgvi_acrobot] max residual:",
            float(np.max(sim["residual_inf"])) if len(sim["residual_inf"]) else np.nan,
        )

    return sim


def get_absolute_angles_and_step_angles(
    state: AcrobotReducedState,
) -> Dict[str, float]:

    # Extract absolute R angles and previous F step angles from reduced state.
    
    thetaR1 = float(angle_from_R(state.R1))
    thetaR2 = float(angle_from_R(state.R2))

    thetaF1_prev = float(angle_from_R(state.F1_prev))
    thetaF2_prev = float(angle_from_R(state.F2_prev))

    return {
        "thetaR1": thetaR1,
        "thetaR2": thetaR2,
        "thetaF1_prev": thetaF1_prev,
        "thetaF2_prev": thetaF2_prev,
    }


def convert_state_to_sdp_initial(
    state: AcrobotReducedState,
    dt_physical: float,
    dt_sdp: float,
    interval_start_state: Optional[AcrobotReducedState] = None,
    history_duration: Optional[float] = None,
    history_method: str = "full_dt_sdp_history",
    history_target_time: Optional[float] = None,
) -> Dict[str, Any]:

    # Convert a fine simulation state into an SDP-compatible initial state.

    dt_physical = float(dt_physical)
    dt_sdp = float(dt_sdp)

    if dt_physical <= 0.0:
        raise ValueError("dt_physical must be positive")
    if dt_sdp <= 0.0:
        raise ValueError("dt_sdp must be positive")

    values = get_absolute_angles_and_step_angles(state)
    thetaF1_last_fine = values["thetaF1_prev"]
    thetaF2_last_fine = values["thetaF2_prev"]

    if interval_start_state is None:
        raise ValueError("A historical reference state is required for MPC SDP conversion")

    if history_duration is None:
        # Backward-compatible behavior for callers that already provide a
        # reference state separated by one complete SDP interval.
        history_duration = dt_sdp
    history_duration = float(history_duration)
    if history_duration <= 0.0:
        raise ValueError("history_duration must be positive")

    F1_history = interval_start_state.R1.T @ state.R1
    F2_history = interval_start_state.R2.T @ state.R2

    # When a complete SDP interval is available, keep the measured relative
    # rotation itself.  In particular, if control_interval == dt_sdp this is
    # exactly the former implementation R_start.T @ R_end.
    full_history_tol = 1.0e-10 * max(1.0, abs(dt_sdp))
    has_full_history = abs(history_duration - dt_sdp) <= full_history_tol
    if has_full_history:
        history_scale = 1.0
        F1_prev_sdp = F1_history
        F2_prev_sdp = F2_history
    else:
        history_scale = dt_sdp / history_duration
        thetaF1_history = float(angle_from_R(F1_history))
        thetaF2_history = float(angle_from_R(F2_history))
        F1_prev_sdp = F_from_delta(history_scale * thetaF1_history)
        F2_prev_sdp = F_from_delta(history_scale * thetaF2_history)

    thetaF1_sdp = float(angle_from_R(F1_prev_sdp))
    thetaF2_sdp = float(angle_from_R(F2_prev_sdp))

    return {
        "R1": state.R1.copy(),
        "R2": state.R2.copy(),
        "F1_prev": F1_prev_sdp,
        "F2_prev": F2_prev_sdp,
        "thetaR1": values["thetaR1"],
        "thetaR2": values["thetaR2"],
        "thetaF1_prev": thetaF1_sdp,
        "thetaF2_prev": thetaF2_sdp,
        "thetaF1_last_fine": thetaF1_last_fine,
        "thetaF2_last_fine": thetaF2_last_fine,
        "omega1_last_fine_rad_s": float(thetaF1_last_fine / dt_physical),
        "omega2_last_fine_rad_s": float(thetaF2_last_fine / dt_physical),
        "f0_history_duration": history_duration,
        "f0_history_scale": float(history_scale),
        "f0_history_has_full_dt_sdp": bool(has_full_history),
        "f0_history_method": str(history_method),
        "f0_history_target_time": (
            None if history_target_time is None else float(history_target_time)
        ),
    }


def convert_state_to_sdp_initial_scalars(
    state: AcrobotReducedState,
    model: AcrobotSO2Model,
    dt_physical: float,
    dt_sdp: float,
    interval_start_state: Optional[AcrobotReducedState] = None,
    history_duration: Optional[float] = None,
    history_method: str = "full_dt_sdp_history",
    history_target_time: Optional[float] = None,
) -> Dict[str, Any]:

    # Convert reduced state to scalar values useful for fixing SDP initial data.

    converted = convert_state_to_sdp_initial(
        state=state,
        dt_physical=dt_physical,
        dt_sdp=dt_sdp,
        interval_start_state=interval_start_state,
        history_duration=history_duration,
        history_method=history_method,
        history_target_time=history_target_time,
    )

    R1 = np.asarray(converted["R1"], dtype=float).reshape(2, 2)
    R2 = np.asarray(converted["R2"], dtype=float).reshape(2, 2)

    F1_prev = np.asarray(converted["F1_prev"], dtype=float).reshape(2, 2)
    F2_prev = np.asarray(converted["F2_prev"], dtype=float).reshape(2, 2)

    c1_0, s1_0 = model.scalars_from_rotation(R1)
    c2_0, s2_0 = model.scalars_from_rotation(R2)

    a1_prev, b1_prev = model.scalars_from_step_rotation(F1_prev)
    a2_prev, b2_prev = model.scalars_from_step_rotation(F2_prev)

    return {

        "c1_0": float(c1_0),
        "s1_0": float(s1_0),
        "c2_0": float(c2_0),
        "s2_0": float(s2_0),

        # Previous step F for the next SDP.
        "a1_prev": float(a1_prev),
        "b1_prev": float(b1_prev),
        "a2_prev": float(a2_prev),
        "b2_prev": float(b2_prev),

        # Diagnostics only.
        "thetaR1": float(converted["thetaR1"]),
        "thetaR2": float(converted["thetaR2"]),
        "thetaF1_prev": float(converted["thetaF1_prev"]),
        "thetaF2_prev": float(converted["thetaF2_prev"]),
        "thetaF1_last_fine": float(converted["thetaF1_last_fine"]),
        "thetaF2_last_fine": float(converted["thetaF2_last_fine"]),
        "omega1_last_fine_rad_s": float(converted["omega1_last_fine_rad_s"]),
        "omega2_last_fine_rad_s": float(converted["omega2_last_fine_rad_s"]),
        "f0_history_duration": float(converted["f0_history_duration"]),
        "f0_history_scale": float(converted["f0_history_scale"]),
        "f0_history_has_full_dt_sdp": bool(
            converted["f0_history_has_full_dt_sdp"]
        ),
        "f0_history_method": str(converted["f0_history_method"]),
        "f0_history_target_time": converted["f0_history_target_time"],
    }


def diagnostics_lgvi(
    model: AcrobotSO2Model,
    sim: Dict[str, Any],
) -> Dict[str, np.ndarray]:

    R1 = np.asarray(sim["R1"], dtype=float)
    R2 = np.asarray(sim["R2"], dtype=float)
    F1 = np.asarray(sim["F1"], dtype=float)
    F2 = np.asarray(sim["F2"], dtype=float)
    X = np.asarray(sim["X"], dtype=float)

    num_nodes = R1.shape[0]
    num_steps = max(0, num_nodes - 1)

    phi_norm = np.zeros(num_nodes, dtype=float)
    phi0_norm = np.zeros(num_nodes, dtype=float)
    phi12_norm = np.zeros(num_nodes, dtype=float)

    orth_R1 = np.zeros(num_nodes, dtype=float)
    orth_R2 = np.zeros(num_nodes, dtype=float)

    det_R1 = np.zeros(num_nodes, dtype=float)
    det_R2 = np.zeros(num_nodes, dtype=float)

    thetaR = np.zeros((num_nodes, 2), dtype=float)

    for k in range(num_nodes):
        phi = model.constraints(X[k], R1[k], R2[k])

        phi0_norm[k] = float(np.linalg.norm(phi[0:2]))
        phi12_norm[k] = float(np.linalg.norm(phi[2:4]))
        phi_norm[k] = float(np.linalg.norm(phi))

        orth_R1[k] = float(orth_error_so2(R1[k]))
        orth_R2[k] = float(orth_error_so2(R2[k]))

        det_R1[k] = float(det_error_so2(R1[k]))
        det_R2[k] = float(det_error_so2(R2[k]))

        thetaR[k] = model.angles_from_rotations(R1[k], R2[k])

    thetaF = np.zeros((num_steps, 2), dtype=float)
    energy = np.full(num_steps, np.nan, dtype=float)

    for k in range(num_steps):
        thetaF[k, 0] = float(angle_from_R(F1[k]))
        thetaF[k, 1] = float(angle_from_R(F2[k]))

        if hasattr(model, "energy_from_reduced_state"):
            energy[k] = model.energy_from_reduced_state(
                R1=R1[k],
                R2=R2[k],
                F1_prev=F1[k],
                F2_prev=F2[k],
                h=float(sim["t"][1] - sim["t"][0]) if len(sim.get("t", [])) > 1 else 1.0,
            )

    if num_steps > 0 and np.isfinite(energy[0]):
        energy_error = energy - energy[0]
    else:
        energy_error = energy.copy()

    return {
        "phi_norm": phi_norm,
        "phi0_norm": phi0_norm,
        "phi12_norm": phi12_norm,
        "orth_R1": orth_R1,
        "orth_R2": orth_R2,
        "det_R1": det_R1,
        "det_R2": det_R2,
        "thetaR": thetaR,
        "thetaF": thetaF,
        "energy": energy,
        "energy_error": energy_error,
    }


def print_step_summary(
    state: AcrobotReducedState,
    model: AcrobotSO2Model,
    h: float,
    label: str = "state",
) -> None:
    
    # Small debugging helper.

    X = reconstruct_X_from_R(model, state.R1, state.R2)
    theta = model.angles_from_rotations(state.R1, state.R2)
    step = get_absolute_angles_and_step_angles(state)

    print(f"[{label}]")
    print(f"  thetaR1 = {theta[0]: .6f} rad, {np.rad2deg(theta[0]): .3f} deg")
    print(f"  thetaR2 = {theta[1]: .6f} rad, {np.rad2deg(theta[1]): .3f} deg")
    print(f"  thetaF1_prev = {step['thetaF1_prev']: .6f} rad")
    print(f"  thetaF2_prev = {step['thetaF2_prev']: .6f} rad")
    print(f"  X = {X}")
    print(f"  constraint norm = {model.constraint_norm(X, state.R1, state.R2):.3e}")
