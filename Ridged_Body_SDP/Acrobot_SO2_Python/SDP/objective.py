from __future__ import annotations

import numpy as np


def rotation_tracking_cost(c, s, c_des, s_des, weight):
    """
    ||R - R_des||_F^2 for SO(2).

    R = [[c,-s],[s,c]]

    ||R-Rd||_F^2 = 2[(c-cd)^2 + (s-sd)^2].
    """
    return weight * 2.0 * ((c - c_des) ** 2 + (s - s_des) ** 2)


def step_tracking_cost(a, b, a_des, b_des, weight):
    """
    ||F - F_des||_F^2 for SO(2).
    """
    return weight * 2.0 * ((a - a_des) ** 2 + (b - b_des) ** 2)


def build_objective(v, params):
    """
    Build reduced Acrobot objective.

    Matches old style:
        terminal R tracking
        terminal F tracking
        stage R tracking
        stage F tracking
        normalized control effort
        optional normalized lambda regularization
    """
    N = int(params["N"])
    idf = params["id"]

    rho_R = float(params["rho_R"])
    rho_F = float(params["rho_F"])
    alpha_R = float(params["alpha_R"])
    alpha_F = float(params["alpha_F"])
    gamma = float(params["gamma"])
    alpha_lam = float(params.get("alpha_lam", 0.0))

    c1_des = float(params.get("c1_des", np.cos(params["thetaR1_des"])))
    s1_des = float(params.get("s1_des", np.sin(params["thetaR1_des"])))

    c2_des = float(params.get("c2_des", np.cos(params["thetaR2_des"])))
    s2_des = float(params.get("s2_des", np.sin(params["thetaR2_des"])))

    a1_des = float(params.get("a1_des", np.cos(params["thetaF1_des"])))
    b1_des = float(params.get("b1_des", np.sin(params["thetaF1_des"])))

    a2_des = float(params.get("a2_des", np.cos(params["thetaF2_des"])))
    b2_des = float(params.get("b2_des", np.sin(params["thetaF2_des"])))

    obj = 0.0

    # Terminal R tracking.
    obj += rotation_tracking_cost(
        v("c1", N),
        v("s1", N),
        c1_des,
        s1_des,
        rho_R,
    )
    obj += rotation_tracking_cost(
        v("c2", N),
        v("s2", N),
        c2_des,
        s2_des,
        rho_R,
    )

    # Terminal F tracking.
    obj += step_tracking_cost(
        v("a1", N - 1),
        v("b1", N - 1),
        a1_des,
        b1_des,
        rho_F,
    )
    obj += step_tracking_cost(
        v("a2", N - 1),
        v("b2", N - 1),
        a2_des,
        b2_des,
        rho_F,
    )

    # Stage R tracking.
    for k in range(N):
        obj += rotation_tracking_cost(
            v("c1", k),
            v("s1", k),
            c1_des,
            s1_des,
            alpha_R,
        )
        obj += rotation_tracking_cost(
            v("c2", k),
            v("s2", k),
            c2_des,
            s2_des,
            alpha_R,
        )

    # Stage F regularization/tracking.
    # Old code used k = 0,...,N-2 and terminal F separately.
    for k in range(N - 1):
        obj += step_tracking_cost(
            v("a1", k),
            v("b1", k),
            a1_des,
            b1_des,
            alpha_F,
        )
        obj += step_tracking_cost(
            v("a2", k),
            v("b2", k),
            a2_des,
            b2_des,
            alpha_F,
        )

    # Control effort. The SDP variable is normalized, and this objective
    # intentionally penalizes normalized control rather than physical control.
    for k in range(1, N):
        obj += (1.0 / gamma) * v("u", k) ** 2

    # Optional lambda regularization. This intentionally penalizes normalized
    # lambda variables rather than raw multipliers or lambda_bar values.
    for k in range(1, N):
        obj += alpha_lam * (
            v("lam0x", k) ** 2
            + v("lam0y", k) ** 2
            + v("lam12x", k) ** 2
            + v("lam12y", k) ** 2
        )

    return obj


def evaluate_objective_from_vector(v_opt, params):
    """
    Numeric objective evaluation for extracted candidates.

    The extracted vector is still in normalized SDP coordinates, so this
    evaluates the same normalized-control/normalized-lambda objective used by
    build_objective().
    """
    N = int(params["N"])
    idf = params["id"]

    def vv(prefix, k):
        return float(v_opt[idf(prefix, k) - 1])

    rho_R = float(params["rho_R"])
    rho_F = float(params["rho_F"])
    alpha_R = float(params["alpha_R"])
    alpha_F = float(params["alpha_F"])
    gamma = float(params["gamma"])
    alpha_lam = float(params.get("alpha_lam", 0.0))

    c1_des = float(params.get("c1_des", np.cos(params["thetaR1_des"])))
    s1_des = float(params.get("s1_des", np.sin(params["thetaR1_des"])))
    c2_des = float(params.get("c2_des", np.cos(params["thetaR2_des"])))
    s2_des = float(params.get("s2_des", np.sin(params["thetaR2_des"])))

    a1_des = float(params.get("a1_des", np.cos(params["thetaF1_des"])))
    b1_des = float(params.get("b1_des", np.sin(params["thetaF1_des"])))
    a2_des = float(params.get("a2_des", np.cos(params["thetaF2_des"])))
    b2_des = float(params.get("b2_des", np.sin(params["thetaF2_des"])))

    obj = 0.0

    obj += float(rotation_tracking_cost(vv("c1", N), vv("s1", N), c1_des, s1_des, rho_R))
    obj += float(rotation_tracking_cost(vv("c2", N), vv("s2", N), c2_des, s2_des, rho_R))

    obj += float(step_tracking_cost(vv("a1", N - 1), vv("b1", N - 1), a1_des, b1_des, rho_F))
    obj += float(step_tracking_cost(vv("a2", N - 1), vv("b2", N - 1), a2_des, b2_des, rho_F))

    for k in range(N):
        obj += float(rotation_tracking_cost(vv("c1", k), vv("s1", k), c1_des, s1_des, alpha_R))
        obj += float(rotation_tracking_cost(vv("c2", k), vv("s2", k), c2_des, s2_des, alpha_R))

    for k in range(N - 1):
        obj += float(step_tracking_cost(vv("a1", k), vv("b1", k), a1_des, b1_des, alpha_F))
        obj += float(step_tracking_cost(vv("a2", k), vv("b2", k), a2_des, b2_des, alpha_F))

    for k in range(1, N):
        obj += (1.0 / gamma) * vv("u", k) ** 2
        obj += alpha_lam * (
            vv("lam0x", k) ** 2
            + vv("lam0y", k) ** 2
            + vv("lam12x", k) ** 2
            + vv("lam12y", k) ** 2
        )

    return float(obj)
