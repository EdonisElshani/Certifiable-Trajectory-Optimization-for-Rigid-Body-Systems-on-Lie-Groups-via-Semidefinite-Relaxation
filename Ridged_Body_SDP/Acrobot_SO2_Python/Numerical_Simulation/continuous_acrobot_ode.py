"""continuous_acrobot_ode.py

Continuous minimal-coordinate Acrobot reference model.

The implemented state is in ABSOLUTE link angles to match the LGVI codes:

    y = [theta1_abs, theta2_abs, omega1_abs, omega2_abs].

The standard Underactuated Robotics Acrobot page writes the dynamics with
q=[theta1, theta2_rel], where theta2_rel is the elbow/relative angle and the
actuator acts at the elbow.  Here theta2_abs = theta1 + theta2_rel, so the same
elbow torque maps to absolute generalized forces [-u, +u].

This is a continuous ODE benchmark.  It is not a variational integrator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

try:
    from .lie_group_so2 import angle_diff_deg
    from .Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model
except ImportError:
    from lie_group_so2 import angle_diff_deg
    from Discrete_Mechanical_Models_Lie_Groups.model_acrobot_so2 import AcrobotSO2Model


def _constants(model: AcrobotSO2Model) -> Dict[str, float]:
    lc1 = float(np.linalg.norm(model.rho10))
    lc2 = float(np.linalg.norm(model.rho212))
    l1 = float(np.linalg.norm(model.rho10 - model.rho112))
    return {
        "m1": float(model.m1),
        "m2": float(model.m2),
        "l1": l1,
        "lc1": lc1,
        "lc2": lc2,
        "J1": float(model.rot_inertia_1),
        "J2": float(model.rot_inertia_2),
        "g": float(model.g),
    }


def mass_matrix_absolute(model: AcrobotSO2Model, theta: np.ndarray) -> np.ndarray:
    c = _constants(model)
    th = np.asarray(theta, dtype=float).reshape(2)
    th1, th2 = float(th[0]), float(th[1])
    A = c["m1"] * c["lc1"] ** 2 + c["J1"] + c["m2"] * c["l1"] ** 2
    B = c["m2"] * c["lc2"] ** 2 + c["J2"]
    C = c["m2"] * c["l1"] * c["lc2"] * np.cos(th1 - th2)
    return np.array([[A, C], [C, B]], dtype=float)


def bias_absolute(model: AcrobotSO2Model, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Return h(q,qdot)+G(q) for M qddot + h + G = Q."""
    c = _constants(model)
    th = np.asarray(theta, dtype=float).reshape(2)
    w = np.asarray(omega, dtype=float).reshape(2)
    th1, th2 = float(th[0]), float(th[1])
    w1, w2 = float(w[0]), float(w[1])
    Cc = c["m2"] * c["l1"] * c["lc2"]
    d = th1 - th2
    coriolis = np.array(
        [
            Cc * np.sin(d) * w2 * w2,
            -Cc * np.sin(d) * w1 * w1,
        ],
        dtype=float,
    )
    gravity = np.array(
        [
            (c["m1"] * c["lc1"] + c["m2"] * c["l1"]) * c["g"] * np.sin(th1),
            c["m2"] * c["lc2"] * c["g"] * np.sin(th2),
        ],
        dtype=float,
    )
    return coriolis + gravity


def generalized_force_absolute(u: float, torque_mode: str = "elbow") -> np.ndarray:
    u = float(u)
    mode = str(torque_mode).lower()
    if mode == "elbow":
        return np.array([-u, +u], dtype=float)
    if mode == "base":
        return np.array([+u, 0.0], dtype=float)
    raise ValueError("torque_mode must be 'elbow' or 'base'.")


def rhs_absolute(
    t: float,
    y: np.ndarray,
    model: AcrobotSO2Model,
    u_fun: Callable[[float], float] | float,
    torque_mode: str = "elbow",
) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(4)
    theta = y[0:2]
    omega = y[2:4]
    u = float(u_fun(t) if callable(u_fun) else u_fun)
    M = mass_matrix_absolute(model, theta)
    rhs = generalized_force_absolute(u, torque_mode=torque_mode) - bias_absolute(model, theta, omega)
    omega_dot = np.linalg.solve(M, rhs)
    return np.r_[omega, omega_dot].astype(float)


def rollout_continuous_acrobot(
    model: AcrobotSO2Model,
    h: float,
    theta0: np.ndarray,
    omega0: np.ndarray,
    u_sequence: np.ndarray,
    torque_mode: str = "elbow",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: Optional[float] = None,
) -> Dict[str, Any]:
    """Integrate the continuous ODE with piecewise-constant controls."""
    u_sequence = np.asarray(u_sequence, dtype=float).reshape(-1)
    n = int(len(u_sequence))
    h = float(h)
    y = np.r_[np.asarray(theta0, dtype=float).reshape(2), np.asarray(omega0, dtype=float).reshape(2)].astype(float)
    Y = np.zeros((n + 1, 4), dtype=float)
    Y[0] = y
    success = True
    message = "ok"
    nfev_total = 0
    if max_step is None:
        max_step = h
    for k, u_k in enumerate(u_sequence):
        sol = solve_ivp(
            lambda t, yy: rhs_absolute(t, yy, model=model, u_fun=float(u_k), torque_mode=torque_mode),
            t_span=(0.0, h),
            y0=y,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=float(max_step),
        )
        nfev_total += int(sol.nfev)
        if not sol.success:
            success = False
            message = str(sol.message)
            Y[k + 1 :] = np.nan
            break
        y = sol.y[:, -1]
        Y[k + 1] = y
    thetaR = Y[:, 0:2]
    theta_dot = Y[:, 2:4]
    thetaF = np.diff(thetaR, axis=0) if n > 0 else np.zeros((0, 2), dtype=float)
    return {
        "method": "continuous_underactuated_absolute_ode",
        "t": np.arange(n + 1, dtype=float) * h,
        "Y": Y,
        "thetaR": thetaR,
        "theta_dot": theta_dot,
        "thetaF": thetaF,
        "u": u_sequence,
        "success": bool(success),
        "message": message,
        "nfev": nfev_total,
    }


def final_angle_error_deg(sim: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, float]:
    a = np.rad2deg(np.asarray(sim["thetaR"][-1], dtype=float).reshape(2))
    b = np.rad2deg(np.asarray(ref["thetaR"][-1], dtype=float).reshape(2))
    err = np.asarray(angle_diff_deg(a, b), dtype=float).reshape(2)
    return {
        "err_theta1_deg": float(err[0]),
        "err_theta2_deg": float(err[1]),
        "err_norm_deg": float(np.linalg.norm(err)),
    }


def rollout_continuous_constant_control(
    model: AcrobotSO2Model,
    h: float,
    theta0: np.ndarray,
    omega0: np.ndarray,
    u: float,
    n_steps: int,
    torque_mode: str = "elbow",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: Optional[float] = None,
) -> Dict[str, Any]:
    """Fast continuous rollout for one constant control over n_steps*h.

    This is used by comparison scripts where each case has a constant input.
    It avoids calling solve_ivp separately for every small LGVI step, because
    apparently even computers object to needless bureaucracy.
    """
    h = float(h)
    tf = float(n_steps) * h
    if max_step is None:
        max_step = min(1e-3, h)
    y0 = np.r_[np.asarray(theta0, dtype=float).reshape(2), np.asarray(omega0, dtype=float).reshape(2)]
    t_eval = np.arange(int(n_steps) + 1, dtype=float) * h
    sol = solve_ivp(
        lambda t, yy: rhs_absolute(t, yy, model=model, u_fun=float(u), torque_mode=torque_mode),
        t_span=(0.0, tf),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=float(max_step),
    )
    if sol.success:
        Y = sol.y.T
    else:
        Y = np.full((int(n_steps) + 1, 4), np.nan, dtype=float)
        Y[0] = y0
    thetaR = Y[:, 0:2]
    theta_dot = Y[:, 2:4]
    thetaF = np.diff(thetaR, axis=0) if int(n_steps) > 0 else np.zeros((0, 2), dtype=float)
    return {
        "method": "continuous_underactuated_absolute_ode_constant_u",
        "t": t_eval,
        "Y": Y,
        "thetaR": thetaR,
        "theta_dot": theta_dot,
        "thetaF": thetaF,
        "u": np.full(int(n_steps), float(u), dtype=float),
        "success": bool(sol.success),
        "message": str(sol.message),
        "nfev": int(sol.nfev),
    }
