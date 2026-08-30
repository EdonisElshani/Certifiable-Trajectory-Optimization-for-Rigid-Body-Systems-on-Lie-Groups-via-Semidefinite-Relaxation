# SDP/constraints.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _get_param(params: Dict[str, Any], key: str, default=None):
    if key in params:
        return params[key]
    if default is not None:
        return default
    raise KeyError(f"Missing parameter '{key}'.")


def _u_phys(u_var, params: Dict[str, Any]):
    # Physical control from normalized SDP variable
    return float(params["u_max"]) * u_var


def _lambda_bar(lam_var, params: Dict[str, Any]):
    # Scaled physical multiplier used inside the dynamics.
    return float(params["lambda_max"]) * lam_var


def compute_node0_from_node1_and_F0(params: Dict[str, Any]) -> Dict[str, float]:

    c1_1 = float(params.get("c1_1", params.get("c1_current", np.cos(params["thetaR1_0"]))))
    s1_1 = float(params.get("s1_1", params.get("s1_current", np.sin(params["thetaR1_0"]))))

    c2_1 = float(params.get("c2_1", params.get("c2_current", np.cos(params["thetaR2_0"]))))
    s2_1 = float(params.get("s2_1", params.get("s2_current", np.sin(params["thetaR2_0"]))))

    a1_0 = float(
        params.get(
            "a1_0",
            params.get("a1_prev", np.cos(params["thetaF1_0"])),
        )
    )
    b1_0 = float(
        params.get(
            "b1_0",
            params.get("b1_prev", np.sin(params["thetaF1_0"])),
        )
    )

    a2_0 = float(
        params.get(
            "a2_0",
            params.get("a2_prev", np.cos(params["thetaF2_0"])),
        )
    )
    b2_0 = float(
        params.get(
            "b2_0",
            params.get("b2_prev", np.sin(params["thetaF2_0"])),
        )
    )

    # R0 = R1 F0^T.
    c1_0 = c1_1 * a1_0 + s1_1 * b1_0
    s1_0 = s1_1 * a1_0 - c1_1 * b1_0

    c2_0 = c2_1 * a2_0 + s2_1 * b2_0
    s2_0 = s2_1 * a2_0 - c2_1 * b2_0

    return {
        "c1_0": c1_0,
        "s1_0": s1_0,
        "c2_0": c2_0,
        "s2_0": s2_0,
        "a1_0": a1_0,
        "b1_0": b1_0,
        "a2_0": a2_0,
        "b2_0": b2_0,
        "c1_1": c1_1,
        "s1_1": s1_1,
        "c2_1": c2_1,
        "s2_1": s2_1,
    }


def get_init_constraints(
    c1_0,
    s1_0,
    c2_0,
    s2_0,
    a1_0,
    b1_0,
    a2_0,
    b2_0,
    c1_1,
    s1_1,
    c2_1,
    s2_1,
    params,
):

    init = compute_node0_from_node1_and_F0(params)

    eqs = [
        c1_0 - init["c1_0"],
        s1_0 - init["s1_0"],
        c2_0 - init["c2_0"],
        s2_0 - init["s2_0"],

        a1_0 - init["a1_0"],
        b1_0 - init["b1_0"],
        a2_0 - init["a2_0"],
        b2_0 - init["b2_0"],

        c1_1 - init["c1_1"],
        s1_1 - init["s1_1"],
        c2_1 - init["c2_1"],
        s2_1 - init["s2_1"],

        c1_0**2 + s1_0**2 - 1,
        c2_0**2 + s2_0**2 - 1,
        a1_0**2 + b1_0**2 - 1,
        a2_0**2 + b2_0**2 - 1,
        c1_1**2 + s1_1**2 - 1,
        c2_1**2 + s2_1**2 - 1,
    ]

    # First 12 are fixed scalar constraints
    eq_mask = [1] * 12 + [0, 0, 1, 1, 0, 0]

    return eqs, [], eq_mask


def get_rotational_kinematics_link1(
    c1_km1,
    s1_km1,
    c1_k,
    s1_k,
    a1_km1,
    b1_km1,
    params,
):

    # R_{1,k} = R_{1,k-1} F_{1,k-1}.
    
    eqs = [
        c1_k - c1_km1 * a1_km1 + s1_km1 * b1_km1,
        s1_k - s1_km1 * a1_km1 - c1_km1 * b1_km1,
    ]
    return eqs, [], [1, 1]


def get_rotational_kinematics_link2(
    c2_km1,
    s2_km1,
    c2_k,
    s2_k,
    a2_km1,
    b2_km1,
    params,
):
    
    # R_{2,k} = R_{2,k-1} F_{2,k-1}.

    eqs = [
        c2_k - c2_km1 * a2_km1 + s2_km1 * b2_km1,
        s2_k - s2_km1 * a2_km1 - c2_km1 * b2_km1,
    ]
    return eqs, [], [1, 1]


def get_SO2_orthogonality_constraint_rotation_R(c1_k, s1_k, c2_k, s2_k, params):
    eqs = [
        c1_k**2 + s1_k**2 - 1,
        c2_k**2 + s2_k**2 - 1,
    ]
    return eqs, [], [0, 0]


def get_SO2_orthogonality_constraint_rotation_F(a1_k, b1_k, a2_k, b2_k, params):
    eqs = [
        a1_k**2 + b1_k**2 - 1,
        a2_k**2 + b2_k**2 - 1,
    ]
    return eqs, [], [1, 1]


def get_step_angle_bound_constraint_link_1(a1_k, params):
    return [], [a1_k - params["a1_min"]], []


def get_step_angle_bound_constraint_link_2(a2_k, params):
    return [], [a2_k - params["a2_min"]], []


def _model2_delta_hat(c_k, s_k, a_prev, b_prev, a_k, b_k):

    delta_s = s_k * (a_k + a_prev - 2.0) + c_k * (b_k - b_prev)
    delta_c = c_k * (a_k + a_prev - 2.0) - s_k * (b_k - b_prev)
    return delta_s, delta_c


def get_translational_dynamics_link1(
    c1_k,
    s1_k,
    a1_km1,
    b1_km1,
    a1_k,
    b1_k,
    lam0x_k,
    lam0y_k,
    lam12x_k,
    lam12y_k,
    params,
):
    
    # Reduced Model 2 translational DEL for link 1.

    h = params["dt"]
    m1 = params["m1"]
    l1 = params["l1"]
    g = params["g"]
    ds1, dc1 = _model2_delta_hat(
        c_k=c1_k,
        s_k=s1_k,
        a_prev=a1_km1,
        b_prev=b1_km1,
        a_k=a1_k,
        b_k=b1_k,
    )

    eqs = [
        (m1 * l1 / 2.0) * ds1
        - h**2 * (_lambda_bar(lam0x_k, params) + _lambda_bar(lam12x_k, params)),

        -(m1 * l1 / 2.0) * dc1
        + h**2 * m1 * g
        - h**2 * (_lambda_bar(lam0y_k, params) + _lambda_bar(lam12y_k, params)),
    ]

    return eqs, [], [1, 1]


def get_translational_dynamics_link2(
    c1_k,
    s1_k,
    a1_km1,
    b1_km1,
    a1_k,
    b1_k,
    c2_k,
    s2_k,
    a2_km1,
    b2_km1,
    a2_k,
    b2_k,
    lam12x_k,
    lam12y_k,
    params,
):
    
    # Reduced Model 2 translational DEL for link 2.

    h = params["dt"]
    m2 = params["m2"]
    l1 = params["l1"]
    l2 = params["l2"]
    g = params["g"]
    ds1, dc1 = _model2_delta_hat(
        c_k=c1_k,
        s_k=s1_k,
        a_prev=a1_km1,
        b_prev=b1_km1,
        a_k=a1_k,
        b_k=b1_k,
    )
    ds2, dc2 = _model2_delta_hat(
        c_k=c2_k,
        s_k=s2_k,
        a_prev=a2_km1,
        b_prev=b2_km1,
        a_k=a2_k,
        b_k=b2_k,
    )

    eqs = [
        m2 * (l1 * ds1 + (l2 / 2.0) * ds2)
        + h**2 * _lambda_bar(lam12x_k, params),

        -m2 * (l1 * dc1 + (l2 / 2.0) * dc2)
        + h**2 * m2 * g
        + h**2 * _lambda_bar(lam12y_k, params),
    ]

    return eqs, [], [1, 1]

def get_rotational_dynamics_link1(
    b1_km1,
    b1_k,
    c1_k,
    s1_k,
    lam0x_k,
    lam0y_k,
    lam12x_k,
    lam12y_k,
    u_k,
    params,
):

    # Reduced rotational dynamics for link 1.

    h = params["dt"]

    Jd1 = np.asarray(params["Jd1"], dtype=float).reshape(2, 2)
    rot_inertia_1 = float(np.trace(Jd1))

    rho_10 = params["rho_10"]
    rho_112 = params["rho_112"]

    rho_10_x, rho_10_y = rho_10[0], rho_10[1]
    rho_112_x, rho_112_y = rho_112[0], rho_112[1]

    Rt_lam0_x = c1_k * _lambda_bar(lam0x_k, params) + s1_k * _lambda_bar(lam0y_k, params)
    Rt_lam0_y = -s1_k * _lambda_bar(lam0x_k, params) + c1_k * _lambda_bar(lam0y_k, params)

    Rt_lam12_x = c1_k * _lambda_bar(lam12x_k, params) + s1_k * _lambda_bar(lam12y_k, params)
    Rt_lam12_y = -s1_k * _lambda_bar(lam12x_k, params) + c1_k * _lambda_bar(lam12y_k, params)

    mu_10 = rho_10_x * Rt_lam0_y - rho_10_y * Rt_lam0_x
    mu_112 = rho_112_x * Rt_lam12_y - rho_112_y * Rt_lam12_x

    eqs = [
        rot_inertia_1 * (b1_km1 - b1_k)
        + h**2 * (mu_10 + mu_112 - _u_phys(u_k, params))
    ]

    return eqs, [], [1]


def get_rotational_dynamics_link2(
    b2_km1,
    b2_k,
    c2_k,
    s2_k,
    lam12x_k,
    lam12y_k,
    u_k,
    params,
):
    
    # Reduced rotational dynamics for link 2.

    h = params["dt"]

    Jd2 = np.asarray(params["Jd2"], dtype=float).reshape(2, 2)
    rot_inertia_2 = float(np.trace(Jd2))

    rho_212 = params["rho_212"]
    rho_212_x, rho_212_y = rho_212[0], rho_212[1]

    Rt_lam12_x = c2_k * _lambda_bar(lam12x_k, params) + s2_k * _lambda_bar(lam12y_k, params)
    Rt_lam12_y = -s2_k * _lambda_bar(lam12x_k, params) + c2_k * _lambda_bar(lam12y_k, params)

    mu_212 = rho_212_x * Rt_lam12_y - rho_212_y * Rt_lam12_x

    eqs = [
        rot_inertia_2 * (b2_km1 - b2_k)
        + h**2 * (_u_phys(u_k, params) - mu_212)
    ]

    return eqs, [], [1]


def get_control_bounds(u_k, params):
    # u_k is normalized;  Physical control is u_max * u_k
    return [], [1.0 - u_k**2], []


def get_lambda_bounds(lam0x_k, lam0y_k, lam12x_k, lam12y_k, params):
    # Lambda variables are normalized; physical dynamics use lambda_max * lambda_var
    ineqs = [
        1.0 - lam0x_k**2,
        1.0 - lam0y_k**2,
        1.0 - lam12x_k**2,
        1.0 - lam12y_k**2,
    ]

    return [], ineqs, []


def reconstruct_positions_from_cs(c1_k, s1_k, c2_k, s2_k, params):
    
    # Reconstruct COM positions from rotations.

    p0 = params["p_0"]
    l1 = params["l1"]
    l2 = params["l2"]

    x1 = np.array(
        [
            p0[0] + (l1 / 2.0) * s1_k,
            p0[1] - (l1 / 2.0) * c1_k,
        ],
        dtype=object,
    )

    x2 = np.array(
        [
            p0[0] + l1 * s1_k + (l2 / 2.0) * s2_k,
            p0[1] - l1 * c1_k - (l2 / 2.0) * c2_k,
        ],
        dtype=object,
    )

    return x1, x2
