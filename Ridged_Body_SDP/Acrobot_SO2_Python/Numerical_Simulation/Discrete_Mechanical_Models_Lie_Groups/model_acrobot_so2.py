# Numerical_Simulation/Discrete_Mechanical_Models_Lie_Groups/model_acrobot_so2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

try:
    from Numerical_Simulation.lie_group_so2 import R_so2, angle_from_R, cross2
except ImportError:
    from lie_group_so2 import R_so2, angle_from_R, cross2


@dataclass(frozen=True)
class AcrobotSO2Params:
    """
    Physical parameters for the reduced absolute-angle SO(2) acrobot.

    No default physical values are stored here.
    The YAML file is the single source of truth.

    Thesis convention:
        R maps body frame to inertial frame.

        thetaR = 0 means the link hangs downward.

        The local negative y-axis points along the physical link from
        proximal joint to distal joint.

    Body-frame vectors:
        rho10:
            link-1 COM -> base joint

        rho112:
            link-1 COM -> elbow joint

        rho212:
            link-2 COM -> elbow joint
    """

    m1: float
    m2: float

    l1: float
    l2: float

    deltaJ1: float
    deltaJ2: float

    g: float
    p0: Tuple[float, float]

    rho10: Tuple[float, float]
    rho112: Tuple[float, float]
    rho212: Tuple[float, float]

    @classmethod
    def from_params_dict(cls, params: Mapping[str, Any]) -> "AcrobotSO2Params":
        """
        Build parameters from the YAML-derived dictionary.

        Supports both:

            flattened params from config_loader.py:
                params["m1"], params["rho_10"], ...

            raw YAML:
                cfg["physical"]["m1"], cfg["physical"]["rho_10"], ...

        If rho vectors are not explicitly given, they are generated from
        the thesis convention:

            rho_10  = [0,  l1/2]
            rho_112 = [0, -l1/2]
            rho_212 = [0,  l2/2]
        """
        physical = params.get("physical", params)

        required_keys = [
            "m1",
            "m2",
            "l1",
            "l2",
            "deltaJ1",
            "deltaJ2",
            "g",
        ]

        missing = [key for key in required_keys if key not in physical]
        if missing:
            raise KeyError(
                "Missing physical parameter(s): "
                + ", ".join(missing)
                + ". Check your YAML/config_loader."
            )

        m1 = float(physical["m1"])
        m2 = float(physical["m2"])

        l1 = float(physical["l1"])
        l2 = float(physical["l2"])

        deltaJ1 = float(physical["deltaJ1"])
        deltaJ2 = float(physical["deltaJ2"])

        g = float(physical["g"])

        p0_value = physical.get("p_0", physical.get("p0", None))
        if p0_value is None:
            raise KeyError(
                "Missing base position. Expected 'p_0' in flattened params "
                "or 'p0' in the YAML physical block."
            )

        p0 = tuple(np.asarray(p0_value, dtype=float).reshape(2))

        rho10_value = physical.get(
            "rho_10",
            physical.get("rho10", np.array([0.0, 0.5 * l1], dtype=float)),
        )

        rho112_value = physical.get(
            "rho_112",
            physical.get("rho112", np.array([0.0, -0.5 * l1], dtype=float)),
        )

        rho212_value = physical.get(
            "rho_212",
            physical.get("rho212", np.array([0.0, 0.5 * l2], dtype=float)),
        )

        rho10 = tuple(np.asarray(rho10_value, dtype=float).reshape(2))
        rho112 = tuple(np.asarray(rho112_value, dtype=float).reshape(2))
        rho212 = tuple(np.asarray(rho212_value, dtype=float).reshape(2))

        return cls(
            m1=m1,
            m2=m2,
            l1=l1,
            l2=l2,
            deltaJ1=deltaJ1,
            deltaJ2=deltaJ2,
            g=g,
            p0=p0,
            rho10=rho10,
            rho112=rho112,
            rho212=rho212,
        )


class AcrobotSO2Model:
    """
    Reduced absolute-angle SO(2) acrobot model.

    This model is designed for Option B:

        use the same reduced R/F dynamics as the SDP.

    State convention:
        R1_k, R2_k:
            absolute rotations at node k

        F1_{k-1}, F2_{k-1}:
            previous relative rotations

        F1_k, F2_k:
            current step rotations solved from reduced dynamics

    No independent x or v dynamics are used here.

    Positions X are only reconstructed for plotting and diagnostics.
    They are not independent state variables.
    """

    def __init__(self, params: AcrobotSO2Params) -> None:
        self.params = params
        p = self.params

        self.m1 = float(p.m1)
        self.m2 = float(p.m2)

        self.l1 = float(p.l1)
        self.l2 = float(p.l2)

        self.deltaJ1 = float(p.deltaJ1)
        self.deltaJ2 = float(p.deltaJ2)

        self.g = float(p.g)

        self.p0 = np.asarray(p.p0, dtype=float).reshape(2)

        self.rho10 = np.asarray(p.rho10, dtype=float).reshape(2)
        self.rho112 = np.asarray(p.rho112, dtype=float).reshape(2)
        self.rho212 = np.asarray(p.rho212, dtype=float).reshape(2)

        # Translational mass matrices for diagnostic energy only.
        self.M1 = self.m1 * np.eye(2)
        self.M2 = self.m2 * np.eye(2)

        self.M = np.block(
            [
                [self.M1, np.zeros((2, 2))],
                [np.zeros((2, 2)), self.M2],
            ]
        )

        # With the thesis convention:
        #   rho10  = [0,  l1/2]
        #   rho112 = [0, -l1/2]
        #   rho212 = [0,  l2/2]
        #
        # These three effective distances produce the reduced translational
        # dynamics used in the SDP screenshots.
        self.d1_com = float(self.rho10[1])
        self.d1_elbow = float(self.rho10[1] - self.rho112[1])
        self.d2_com = float(self.rho212[1])

        self.Jd1 = np.diag([self.deltaJ1, self.deltaJ1])
        self.Jd2 = np.diag([self.deltaJ2, self.deltaJ2])
        self.rot_inertia_1 = float(np.trace(self.Jd1))
        self.rot_inertia_2 = float(np.trace(self.Jd2))

    @classmethod
    def from_params_dict(cls, params: Mapping[str, Any]) -> "AcrobotSO2Model":
        """
        Build the model from the shared YAML-derived params dictionary.
        """
        return cls(AcrobotSO2Params.from_params_dict(params))

    # ------------------------------------------------------------------
    # Basic SO(2) helpers
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def rotation_from_scalars(c: float, s: float) -> np.ndarray:
        """
        Return SO(2)-style matrix from scalar variables.

        R = [[c, -s],
             [s,  c]]
        """
        return np.array(
            [
                [float(c), -float(s)],
                [float(s), float(c)],
            ],
            dtype=float,
        )

    @staticmethod
    def scalars_from_rotation(R: np.ndarray) -> Tuple[float, float]:
        """
        Extract c, s from:

            R = [[c, -s],
                 [s,  c]]
        """
        R = np.asarray(R, dtype=float).reshape(2, 2)
        c = float(R[0, 0])
        s = float(R[1, 0])
        return c, s

    @staticmethod
    def step_rotation_from_scalars(a: float, b: float) -> np.ndarray:
        """
        Return F from step variables:

            F = [[a, -b],
                 [b,  a]]
        """
        return np.array(
            [
                [float(a), -float(b)],
                [float(b), float(a)],
            ],
            dtype=float,
        )

    @staticmethod
    def scalars_from_step_rotation(F: np.ndarray) -> Tuple[float, float]:
        """
        Extract a, b from:

            F = [[a, -b],
                 [b,  a]]
        """
        F = np.asarray(F, dtype=float).reshape(2, 2)
        a = float(F[0, 0])
        b = float(F[1, 0])
        return a, b

    def rotations_from_angles(
        self,
        thetaR1: float,
        thetaR2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return absolute rotations from absolute body-frame angles.
        """
        R1 = R_so2(float(thetaR1))
        R2 = R_so2(float(thetaR2))
        return R1, R2

    def angles_from_rotations(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        wrap: bool = False,
    ) -> np.ndarray:
        """
        Return absolute body-frame angles [thetaR1, thetaR2].
        """
        thetaR1 = float(angle_from_R(R1))
        thetaR2 = float(angle_from_R(R2))

        if wrap:
            thetaR1 = self.wrap_angle(thetaR1)
            thetaR2 = self.wrap_angle(thetaR2)

        return np.array([thetaR1, thetaR2], dtype=float)

    # ------------------------------------------------------------------
    # Reduced kinematics
    # ------------------------------------------------------------------
    @staticmethod
    def kinematics_next_scalars(
        c: float,
        s: float,
        a: float,
        b: float,
    ) -> Tuple[float, float]:
        """
        Scalar SO(2) kinematics:

            R_{k+1} = R_k F_k

        If:

            R_k = [[c, -s],
                   [s,  c]]

            F_k = [[a, -b],
                   [b,  a]]

        then:

            c_next = c a - s b
            s_next = s a + c b
        """
        c_next = float(c) * float(a) - float(s) * float(b)
        s_next = float(s) * float(a) + float(c) * float(b)

        return c_next, s_next

    def advance_rotations(
        self,
        R1_k: np.ndarray,
        R2_k: np.ndarray,
        F1_k: np.ndarray,
        F2_k: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance rotations by:

            R1_{k+1} = R1_k F1_k
            R2_{k+1} = R2_k F2_k
        """
        R1_k = np.asarray(R1_k, dtype=float).reshape(2, 2)
        R2_k = np.asarray(R2_k, dtype=float).reshape(2, 2)

        F1_k = np.asarray(F1_k, dtype=float).reshape(2, 2)
        F2_k = np.asarray(F2_k, dtype=float).reshape(2, 2)

        return R1_k @ F1_k, R2_k @ F2_k

    # ------------------------------------------------------------------
    # Reconstructed positions for diagnostics / plotting
    # ------------------------------------------------------------------
    def reconstruct_positions_from_rotations(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct COM positions X = [x1; x2] from R1, R2.

        Constraints:

            phi0  = x1 + R1 rho10 - p0 = 0
            phi12 = x1 + R1 rho112 - x2 - R2 rho212 = 0

        Therefore:

            x1 = p0 - R1 rho10
            x2 = x1 + R1 rho112 - R2 rho212

        With the thesis rho vectors:

            x1 = [l1/2 sin(theta1), -l1/2 cos(theta1)]
            x2 = [l1 sin(theta1) + l2/2 sin(theta2),
                  -l1 cos(theta1) - l2/2 cos(theta2)]
        """
        R1 = np.asarray(R1, dtype=float).reshape(2, 2)
        R2 = np.asarray(R2, dtype=float).reshape(2, 2)

        x1 = self.p0 - R1 @ self.rho10
        x2 = x1 + R1 @ self.rho112 - R2 @ self.rho212

        return np.r_[x1, x2]

    def positions_from_angles(
        self,
        thetaR1: float,
        thetaR2: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return X, R1, R2, constraints from absolute angles.

        This is only an initializer / diagnostic helper.
        """
        R1, R2 = self.rotations_from_angles(thetaR1, thetaR2)
        X = self.reconstruct_positions_from_rotations(R1, R2)
        phi = self.constraints(X, R1, R2)

        return X, R1, R2, phi

    def constraints(
        self,
        X: np.ndarray,
        R1: np.ndarray,
        R2: np.ndarray,
    ) -> np.ndarray:
        """
        Return holonomic constraints [phi0; phi12].

        This is diagnostic only for Option B.
        The reduced dynamics do not solve for X independently.
        """
        X = np.asarray(X, dtype=float).reshape(4)
        R1 = np.asarray(R1, dtype=float).reshape(2, 2)
        R2 = np.asarray(R2, dtype=float).reshape(2, 2)

        x1 = X[0:2]
        x2 = X[2:4]

        phi0 = x1 + R1 @ self.rho10 - self.p0
        phi12 = x1 + R1 @ self.rho112 - x2 - R2 @ self.rho212

        return np.r_[phi0, phi12]

    def constraint_norm(
        self,
        X: np.ndarray,
        R1: np.ndarray,
        R2: np.ndarray,
    ) -> float:
        return float(np.linalg.norm(self.constraints(X, R1, R2)))

    # ------------------------------------------------------------------
    # Constraint moments
    # ------------------------------------------------------------------
    def constraint_moments(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        lam0: np.ndarray,
        lam12: np.ndarray,
    ) -> np.ndarray:
        """
        Constraint moments in SO(2).

        Link 1:

            mu1 = rho10 x (R1^T lam0)
                + rho112 x (R1^T lam12)

        Link 2:

            mu2 = - rho212 x (R2^T lam12)

        This sign convention matches:

            phi12 = x1 + R1 rho112 - x2 - R2 rho212
        """
        R1 = np.asarray(R1, dtype=float).reshape(2, 2)
        R2 = np.asarray(R2, dtype=float).reshape(2, 2)

        lam0 = np.asarray(lam0, dtype=float).reshape(2)
        lam12 = np.asarray(lam12, dtype=float).reshape(2)

        mu1 = cross2(self.rho10, R1.T @ lam0) + cross2(
            self.rho112,
            R1.T @ lam12,
        )

        mu2 = -cross2(self.rho212, R2.T @ lam12)

        return np.array([mu1, mu2], dtype=float)

    constraint_torques = constraint_moments

    # ------------------------------------------------------------------
    # Option B: reduced SDP-matching translational dynamics
    # ------------------------------------------------------------------
    @staticmethod
    def _model2_delta_hat(
        c_k: float,
        s_k: float,
        a_prev: float,
        b_prev: float,
        a_k: float,
        b_k: float,
    ) -> Tuple[float, float]:
        """
        Model 2 kinematic-substituted second difference.

        From R_{k+1}=R_k F_k and R_{k-1}=R_k F_{k-1}^T,

            Delta_hat s = s_k(a_k + a_{k-1} - 2) + c_k(b_k - b_{k-1}),
            Delta_hat c = c_k(a_k + a_{k-1} - 2) - s_k(b_k - b_{k-1}).
        """
        delta_s = float(s_k) * (float(a_k) + float(a_prev) - 2.0) + float(c_k) * (
            float(b_k) - float(b_prev)
        )
        delta_c = float(c_k) * (float(a_k) + float(a_prev) - 2.0) - float(s_k) * (
            float(b_k) - float(b_prev)
        )
        return float(delta_s), float(delta_c)

    # ------------------------------------------------------------------
    # Model 2: exact kinematic-substituted translational dynamics
    # ------------------------------------------------------------------
    def reduced_translational_residual_from_scalars(
        self,
        c1_k: float,
        s1_k: float,
        a1_prev: float,
        b1_prev: float,
        a1_k: float,
        b1_k: float,
        c2_k: float,
        s2_k: float,
        a2_prev: float,
        b2_prev: float,
        a2_k: float,
        b2_k: float,
        lam0: np.ndarray,
        lam12: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Reduced Model 2 translational dynamics.

        This is the exact finite LGVI second difference after substituting

            R_{k+1} = R_k F_k,
            R_{k-1} = R_k F_{k-1}^T.

        It is algebraically equivalent to the direct reconstructed-position
        reduction, but it only uses R_k, F_{k-1}, and F_k in the dynamics.
        """
        h = float(h)
        lam0 = np.asarray(lam0, dtype=float).reshape(2)
        lam12 = np.asarray(lam12, dtype=float).reshape(2)

        ds1, dc1 = self._model2_delta_hat(c1_k, s1_k, a1_prev, b1_prev, a1_k, b1_k)
        ds2, dc2 = self._model2_delta_hat(c2_k, s2_k, a2_prev, b2_prev, a2_k, b2_k)

        res1_x = self.m1 * self.d1_com * ds1 - h**2 * (lam0[0] + lam12[0])
        res1_y = -self.m1 * self.d1_com * dc1 + h**2 * self.m1 * self.g - h**2 * (
            lam0[1] + lam12[1]
        )
        res2_x = self.m2 * (self.d1_elbow * ds1 + self.d2_com * ds2) + h**2 * lam12[0]
        res2_y = -self.m2 * (self.d1_elbow * dc1 + self.d2_com * dc2) + h**2 * self.m2 * self.g + h**2 * lam12[1]

        return np.array([res1_x, res1_y, res2_x, res2_y], dtype=float)

    def reduced_translational_residual(
        self,
        R1_k: np.ndarray,
        R2_k: np.ndarray,
        F1_k: np.ndarray,
        F2_k: np.ndarray,
        R1_prev: np.ndarray,
        R2_prev: np.ndarray,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
        lam0: np.ndarray,
        lam12: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Matrix wrapper for Reduced Model 2 translational dynamics.

        R1_prev and R2_prev are kept in the signature for compatibility with
        the old simulator structure, but Model 2 does not use them explicitly.
        The previous node is represented through F_{k-1}.
        """
        c1_k, s1_k = self.scalars_from_rotation(R1_k)
        c2_k, s2_k = self.scalars_from_rotation(R2_k)

        a1_k, b1_k = self.scalars_from_step_rotation(F1_k)
        a2_k, b2_k = self.scalars_from_step_rotation(F2_k)
        a1_prev, b1_prev = self.scalars_from_step_rotation(F1_prev)
        a2_prev, b2_prev = self.scalars_from_step_rotation(F2_prev)

        return self.reduced_translational_residual_from_scalars(
            c1_k=c1_k,
            s1_k=s1_k,
            a1_prev=a1_prev,
            b1_prev=b1_prev,
            a1_k=a1_k,
            b1_k=b1_k,
            c2_k=c2_k,
            s2_k=s2_k,
            a2_prev=a2_prev,
            b2_prev=b2_prev,
            a2_k=a2_k,
            b2_k=b2_k,
            lam0=lam0,
            lam12=lam12,
            h=h,
        )

    # ------------------------------------------------------------------
    # Option B: reduced rotational dynamics
    # ------------------------------------------------------------------
    def reduced_rotational_residual(
        self,
        R1_k: np.ndarray,
        R2_k: np.ndarray,
        F1_k: np.ndarray,
        F2_k: np.ndarray,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
        lam0: np.ndarray,
        lam12: np.ndarray,
        u_k: float,
        h: float,
    ) -> np.ndarray:
        """
        Reduced rotational dynamics in the same b-variable style as the SDP.

        Approximate discrete momentum:

            Pi_i * h = trace(Jd_i) * b_i

        Dynamics:

            link 1:
                trace(Jd1)*(b1_prev - b1_k)
                + h^2*(mu1 - u_k) = 0

            link 2:
                trace(Jd2)*(b2_prev - b2_k)
                + h^2*(mu2 + u_k) = 0

        where:

            [mu1, mu2] = constraint_moments(...)
        """
        h = float(h)
        u_k = float(u_k)

        _, b1_k = self.scalars_from_step_rotation(F1_k)
        _, b2_k = self.scalars_from_step_rotation(F2_k)

        _, b1_prev = self.scalars_from_step_rotation(F1_prev)
        _, b2_prev = self.scalars_from_step_rotation(F2_prev)

        # Consistent interior-node indexing: the reduced DEL residual uses
        # F_{k-1}, F_k, lambda_k, u_k, so the constraint moments are evaluated
        # at the current node R_k.  Using R_{k+1} here would mix two shifted
        # indexing conventions.
        mu1, mu2 = self.constraint_moments(
            R1=R1_k,
            R2=R2_k,
            lam0=lam0,
            lam12=lam12,
        )

        res_rot1 = self.rot_inertia_1 * (b1_prev - b1_k) + h**2 * (mu1 - u_k)
        res_rot2 = self.rot_inertia_2 * (b2_prev - b2_k) + h**2 * (mu2 + u_k)

        return np.array([res_rot1, res_rot2], dtype=float)

    # ------------------------------------------------------------------
    # Option B: complete one-step reduced residual
    # ------------------------------------------------------------------
    def reduced_step_residual(
        self,
        z: np.ndarray,
        R1_k: np.ndarray,
        R2_k: np.ndarray,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
        u_k: float,
        h: float,
    ) -> np.ndarray:
        """
        Complete reduced one-step residual for the Option B simulator.

        Unknown vector:

            z = [
                a1_k,
                b1_k,
                a2_k,
                b2_k,
                lam0_x,
                lam0_y,
                lam12_x,
                lam12_y,
            ]

        Residuals:

            4 translational reduced dynamics
            2 rotational reduced dynamics
            2 SO(2) constraints for F1_k, F2_k

        Total: 8 equations.

        No X_next is solved here.
        After solving, update:

            R1_{k+1} = R1_k F1_k
            R2_{k+1} = R2_k F2_k
        """
        z = np.asarray(z, dtype=float).reshape(8)

        a1_k = float(z[0])
        b1_k = float(z[1])

        a2_k = float(z[2])
        b2_k = float(z[3])

        lam0 = np.array([z[4], z[5]], dtype=float)
        lam12 = np.array([z[6], z[7]], dtype=float)

        F1_k = self.step_rotation_from_scalars(a1_k, b1_k)
        F2_k = self.step_rotation_from_scalars(a2_k, b2_k)

        R1_prev = np.asarray(R1_k, dtype=float).reshape(2, 2) @ np.asarray(
            F1_prev,
            dtype=float,
        ).reshape(2, 2).T

        R2_prev = np.asarray(R2_k, dtype=float).reshape(2, 2) @ np.asarray(
            F2_prev,
            dtype=float,
        ).reshape(2, 2).T

        trans = self.reduced_translational_residual(
            R1_k=R1_k,
            R2_k=R2_k,
            F1_k=F1_k,
            F2_k=F2_k,
            R1_prev=R1_prev,
            R2_prev=R2_prev,
            F1_prev=F1_prev,
            F2_prev=F2_prev,
            lam0=lam0,
            lam12=lam12,
            h=h,
        )

        rot = self.reduced_rotational_residual(
            R1_k=R1_k,
            R2_k=R2_k,
            F1_k=F1_k,
            F2_k=F2_k,
            F1_prev=F1_prev,
            F2_prev=F2_prev,
            lam0=lam0,
            lam12=lam12,
            u_k=u_k,
            h=h,
        )

        so2 = np.array(
            [
                a1_k**2 + b1_k**2 - 1.0,
                a2_k**2 + b2_k**2 - 1.0,
            ],
            dtype=float,
        )

        return np.r_[trans, rot, so2]

    def initial_step_guess(
        self,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Initial guess for the reduced Option B root solve.

        Uses the previous step rotation as a constant-velocity guess and zero
        constraint multipliers.
        """
        a1_prev, b1_prev = self.scalars_from_step_rotation(F1_prev)
        a2_prev, b2_prev = self.scalars_from_step_rotation(F2_prev)

        return np.array(
            [
                a1_prev,
                b1_prev,
                a2_prev,
                b2_prev,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

    def unpack_reduced_solution(
        self,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert reduced solution z into F1_k, F2_k, lam0, lam12.
        """
        z = np.asarray(z, dtype=float).reshape(8)

        a1_k = float(z[0])
        b1_k = float(z[1])

        a2_k = float(z[2])
        b2_k = float(z[3])

        F1_k = self.step_rotation_from_scalars(a1_k, b1_k)
        F2_k = self.step_rotation_from_scalars(a2_k, b2_k)

        lam0 = np.array([z[4], z[5]], dtype=float)
        lam12 = np.array([z[6], z[7]], dtype=float)

        return F1_k, F2_k, lam0, lam12

    def advance_reduced_state(
        self,
        R1_k: np.ndarray,
        R2_k: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance reduced state after solving z.

        Returns:

            R1_next, R2_next, F1_k, F2_k, lam0, lam12
        """
        F1_k, F2_k, lam0, lam12 = self.unpack_reduced_solution(z)

        R1_next, R2_next = self.advance_rotations(
            R1_k=R1_k,
            R2_k=R2_k,
            F1_k=F1_k,
            F2_k=F2_k,
        )

        return R1_next, R2_next, F1_k, F2_k, lam0, lam12

    # ------------------------------------------------------------------
    # Diagnostics only
    # ------------------------------------------------------------------
    def potential_from_X(
        self,
        X: np.ndarray,
    ) -> float:
        """
        Gravitational potential from reconstructed COM coordinates.

        Diagnostic only.
        """
        X = np.asarray(X, dtype=float).reshape(4)

        x1 = X[0:2]
        x2 = X[2:4]

        return float(self.m1 * self.g * x1[1] + self.m2 * self.g * x2[1])

    def potential_from_rotations(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
    ) -> float:
        """
        Gravitational potential after reconstructing X from rotations.

        Diagnostic only.
        """
        X = self.reconstruct_positions_from_rotations(R1, R2)
        return self.potential_from_X(X)

    potential = potential_from_X

    def kinetic_from_reduced_step(
        self,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
        h: float,
    ) -> float:
        """
        Approximate rotational kinetic energy from reduced F variables.

        Diagnostic only.
        """
        h = float(h)

        _, b1 = self.scalars_from_step_rotation(F1_prev)
        _, b2 = self.scalars_from_step_rotation(F2_prev)

        omega1 = float(np.arcsin(np.clip(b1, -1.0, 1.0))) / h
        omega2 = float(np.arcsin(np.clip(b2, -1.0, 1.0))) / h

        return float(
            0.5 * self.rot_inertia_1 * omega1**2
            + 0.5 * self.rot_inertia_2 * omega2**2
        )

    def energy_from_reduced_state(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        F1_prev: np.ndarray,
        F2_prev: np.ndarray,
        h: float,
    ) -> float:
        """
        Approximate total energy from reduced state.

        Diagnostic only.
        """
        T = self.kinetic_from_reduced_step(
            F1_prev=F1_prev,
            F2_prev=F2_prev,
            h=h,
        )

        V = self.potential_from_rotations(R1, R2)

        return float(T + V)
