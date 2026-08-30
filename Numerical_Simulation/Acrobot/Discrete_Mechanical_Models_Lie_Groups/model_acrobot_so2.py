"""Planar Acrobot model in maximal coordinates on (R^2 x SO(2))^2.

Conventions
-----------
Maximal-coordinate LGVI variables use absolute link attitudes R1, R2.  The
body x-axis points along each link from the proximal joint to the distal joint.
The fixed-base and elbow constraints are

    phi0  = x1 + R1 rho10 - p0 = 0,
    phi12 = x1 + R1 rho112 - x2 - R2 rho212 = 0.

For the classical Acrobot continuous equations, we use the common relative
coordinates

    q = [theta1, theta2]^T,

where theta1 is the shoulder angle and theta2 is the elbow relative angle.  The
zero configuration has both links pointing directly downward.  Therefore the
absolute SO(2) attitude angles used by the maximal-coordinate model are

    alpha1 = theta1 - pi/2,
    alpha2 = theta1 + theta2 - pi/2.

The classical Acrobot actuation is a single scalar elbow torque u with
relative-coordinate input matrix B=[0,1]^T.  In absolute-link virtual work this
is tau=[-u,+u]^T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from Acrobot.lie_group_so2 import R_so2, angle_from_R, cross2
except ImportError:  # support running from inside Acrobot
    from lie_group_so2 import R_so2, angle_from_R, cross2


@dataclass
class AcrobotSO2Params:
    # Geometry and mass. Defaults: two uniform rods of length 1 and mass 1.
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 0.5
    l2: float = 0.5
    lc1: float = 0.25
    lc2: float = 0.25
    # Moments of inertia about each link center of mass.
    J1: float = 1.0 / 12.0 * m1 * l1**2
    J2: float = 1.0 / 12.0 * m2 * l2**2
    g: float = 9.81
    p0: Tuple[float, float] = (0.0, 0.0)


class AcrobotSO2Model:
    """Acrobot model helper for LGVI and RK4 validation."""

    def __init__(self, params: Optional[AcrobotSO2Params] = None) -> None:
        self.params = params if params is not None else AcrobotSO2Params()
        p = self.params

        self.m1 = float(p.m1)
        self.m2 = float(p.m2)
        self.l1 = float(p.l1)
        self.l2 = float(p.l2)
        self.lc1 = float(p.lc1)
        self.lc2 = float(p.lc2)
        self.J1 = float(p.J1)  # COM inertia of link 1
        self.J2 = float(p.J2)  # COM inertia of link 2
        self.g = float(p.g)
        self.p0 = np.asarray(p.p0, dtype=float).reshape(2)

        self.M1 = self.m1 * np.eye(2)
        self.M2 = self.m2 * np.eye(2)
        self.M = np.block([[self.M1, np.zeros((2, 2))], [np.zeros((2, 2)), self.M2]])

        # Lee-style nonstandard inertia for SO(2):
        # T_R = 1/2 tr(omega_hat Jd omega_hat^T) = 1/2 J omega^2.
        self.Jd1 = 0.5 * self.J1 * np.eye(2)
        self.Jd2 = 0.5 * self.J2 * np.eye(2)

        self.e2 = np.array([0.0, 1.0])  # upward world direction
        self.E = np.r_[self.e2, self.e2]

        # Body-fixed COM-to-joint vectors.
        self.rho10 = np.array([-self.lc1, 0.0])            # link 1 COM -> base joint
        self.rho112 = np.array([self.l1 - self.lc1, 0.0])  # link 1 COM -> elbow joint
        self.rho212 = np.array([-self.lc2, 0.0])            # link 2 COM -> elbow joint

        # Vectors from proximal joint to COM/distal joint.
        self.a1 = np.array([self.lc1, 0.0])
        self.b1 = np.array([self.l1, 0.0])
        self.a2 = np.array([self.lc2, 0.0])

        # Pivot inertias used by the standard Acrobot manipulator equations.
        # These correspond to I1, I2 in the equations shown in the screenshot.
        self.I1_pivot = self.J1 + self.m1 * self.lc1**2
        self.I2_pivot = self.J2 + self.m2 * self.lc2**2

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def absolute_angles_from_relative(self, theta1: float, theta2: float) -> np.ndarray:
        """Return absolute SO(2) attitude angles [alpha1, alpha2].

        theta1=theta2=0 corresponds to both links pointing downward.
        """
        return np.array([theta1 - 0.5 * np.pi, theta1 + theta2 - 0.5 * np.pi], dtype=float)

    def relative_angles_from_absolute(self, alpha1: float, alpha2: float, wrap: bool = False) -> np.ndarray:
        """Return relative Acrobot angles [theta1, theta2] from absolute attitudes."""
        theta1 = alpha1 + 0.5 * np.pi
        theta2 = alpha2 - alpha1
        if wrap:
            theta1 = self.wrap_angle(theta1)
            theta2 = self.wrap_angle(theta2)
        return np.array([theta1, theta2], dtype=float)

    def relative_angles_from_rotations(self, R1: np.ndarray, R2: np.ndarray, wrap: bool = False) -> np.ndarray:
        alpha1 = angle_from_R(R1)
        alpha2 = angle_from_R(R2)
        return self.relative_angles_from_absolute(alpha1, alpha2, wrap=wrap)

    # ------------------------------------------------------------------
    # Maximal-coordinate constraints and geometry
    # ------------------------------------------------------------------
    def constraints(self, X: np.ndarray, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """Return stacked constraints [phi0; phi12] in R^4."""
        X = np.asarray(X, dtype=float).reshape(4)
        x1 = X[:2]
        x2 = X[2:]
        phi0 = x1 + R1 @ self.rho10 - self.p0
        phi12 = x1 + R1 @ self.rho112 - x2 - R2 @ self.rho212
        return np.r_[phi0, phi12]

    def Gx_T_lambda(self, lam0: np.ndarray, lam12: np.ndarray) -> np.ndarray:
        """Return D_X phi(q)^T lambda = [lam0 + lam12; -lam12]."""
        lam0 = np.asarray(lam0, dtype=float).reshape(2)
        lam12 = np.asarray(lam12, dtype=float).reshape(2)
        return np.r_[lam0 + lam12, -lam12]

    def constraint_torques(self, R1: np.ndarray, R2: np.ndarray, lam0: np.ndarray, lam12: np.ndarray) -> np.ndarray:
        """Return scalar SO(2) constraint torque coefficients [gamma1, gamma2]."""
        lam0 = np.asarray(lam0, dtype=float).reshape(2)
        lam12 = np.asarray(lam12, dtype=float).reshape(2)
        gamma1 = cross2(self.rho10, R1.T @ lam0) + cross2(self.rho112, R1.T @ lam12)
        gamma2 = -cross2(self.rho212, R2.T @ lam12)
        return np.array([gamma1, gamma2], dtype=float)

    def positions_from_angles(self, alpha1: float, alpha2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return X, R1, R2, constraints from absolute link attitude angles."""
        R1 = R_so2(alpha1)
        R2 = R_so2(alpha2)
        x1 = self.p0 - R1 @ self.rho10
        x2 = x1 + R1 @ self.rho112 - R2 @ self.rho212
        X = np.r_[x1, x2]
        return X, R1, R2, self.constraints(X, R1, R2)

    def positions_from_relative(self, theta1: float, theta2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return X, R1, R2, constraints from standard Acrobot relative angles."""
        alpha1, alpha2 = self.absolute_angles_from_relative(theta1, theta2)
        return self.positions_from_angles(alpha1, alpha2)

    def angles_from_rotations(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """Return absolute SO(2) attitude angles [alpha1, alpha2]."""
        return np.array([angle_from_R(R1), angle_from_R(R2)], dtype=float)

    # ------------------------------------------------------------------
    # Energies
    # ------------------------------------------------------------------
    def potential(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=float).reshape(4)
        return float(self.g * self.E.T @ self.M @ X)

    def energy_from_maximal(self, X: np.ndarray, V: np.ndarray, omega1_abs: float, omega2_abs: float) -> float:
        """Continuous energy evaluated from maximal coordinates and absolute link velocities."""
        X = np.asarray(X, dtype=float).reshape(4)
        V = np.asarray(V, dtype=float).reshape(4)
        T_trans = 0.5 * float(V.T @ self.M @ V)
        T_rot = 0.5 * self.J1 * omega1_abs**2 + 0.5 * self.J2 * omega2_abs**2
        return T_trans + T_rot + self.potential(X)

    def energy_from_relative(self, y: np.ndarray) -> float:
        """Energy from y=[theta1, theta2, theta1dot, theta2dot]."""
        theta1, theta2, q1dot, q2dot = np.asarray(y, dtype=float).reshape(4)
        Mq = self.mass_matrix_relative(theta1, theta2)
        qdot = np.array([q1dot, q2dot], dtype=float)
        X, _, _, _ = self.positions_from_relative(theta1, theta2)
        return 0.5 * float(qdot.T @ Mq @ qdot) + self.potential(X)

    def energy_from_absolute(self, y_abs: np.ndarray) -> float:
        """Energy from y_abs=[alpha1, alpha2, omega1_abs, omega2_abs]."""
        alpha1, alpha2, omega1, omega2 = np.asarray(y_abs, dtype=float).reshape(4)
        theta1, theta2 = self.relative_angles_from_absolute(alpha1, alpha2)
        return self.energy_from_relative(np.array([theta1, theta2, omega1, omega2 - omega1]))

    # Keep old name for backward compatibility with the first package.
    energy_from_minimal = energy_from_absolute

    # ------------------------------------------------------------------
    # Standard Acrobot continuous dynamics in relative coordinates
    # ------------------------------------------------------------------
    def mass_matrix_relative(self, theta1: float, theta2: float) -> np.ndarray:
        """M(q) from the standard Acrobot equations q=[theta1, theta2]."""
        c2 = np.cos(theta2)
        h12 = self.m2 * self.l1 * self.lc2 * c2
        M11 = self.I1_pivot + self.I2_pivot + self.m2 * self.l1**2 + 2.0 * h12
        M12 = self.I2_pivot + h12
        M22 = self.I2_pivot
        return np.array([[M11, M12], [M12, M22]], dtype=float)

    def coriolis_matrix_relative(self, theta1: float, theta2: float, qdot: np.ndarray) -> np.ndarray:
        """C(q,qdot) from the equations in the screenshot."""
        qdot = np.asarray(qdot, dtype=float).reshape(2)
        q1dot, q2dot = qdot
        hsin = self.m2 * self.l1 * self.lc2 * np.sin(theta2)
        return np.array([
            [-2.0 * hsin * q2dot, -hsin * q2dot],
            [ hsin * q1dot,          0.0],
        ], dtype=float)

    def gravity_relative(self, theta1: float, theta2: float) -> np.ndarray:
        """tau_g(q) from the screenshot, with zero configuration downward."""
        g1 = -self.m1 * self.g * self.lc1 * np.sin(theta1) \
             -self.m2 * self.g * (self.l1 * np.sin(theta1) + self.lc2 * np.sin(theta1 + theta2))
        g2 = -self.m2 * self.g * self.lc2 * np.sin(theta1 + theta2)
        return np.array([g1, g2], dtype=float)

    def generalized_torque_relative(self, u: float) -> np.ndarray:
        """Classical Acrobot: scalar elbow torque u -> B u with B=[0,1]^T."""
        return np.array([0.0, float(u)], dtype=float)

    def generalized_torque_classical_acrobot(self, u: float) -> np.ndarray:
        """Absolute-link virtual work: scalar elbow torque u -> tau=[-u,+u]."""
        return np.array([-float(u), float(u)], dtype=float)

    def qddot_relative(self, theta1: float, theta2: float, qdot: np.ndarray, u: float = 0.0) -> np.ndarray:
        """Solve M(q) qddot + C(q,qdot) qdot = tau_g(q) + B u."""
        qdot = np.asarray(qdot, dtype=float).reshape(2)
        Mq = self.mass_matrix_relative(theta1, theta2)
        Cq = self.coriolis_matrix_relative(theta1, theta2, qdot)
        rhs = self.gravity_relative(theta1, theta2) + self.generalized_torque_relative(u) - Cq @ qdot
        return np.linalg.solve(Mq, rhs)

    def rhs_relative(self, t: float, y: np.ndarray, u_fun: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """Continuous ODE in standard relative coordinates.

        y = [theta1, theta2, theta1dot, theta2dot].
        """
        theta1, theta2, q1dot, q2dot = np.asarray(y, dtype=float).reshape(4)
        u = 0.0 if u_fun is None else float(u_fun(t))
        qddot = self.qddot_relative(theta1, theta2, np.array([q1dot, q2dot]), u)
        return np.array([q1dot, q2dot, qddot[0], qddot[1]], dtype=float)

    def rhs_absolute(self, t: float, y_abs: np.ndarray, u_fun: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """Continuous ODE in absolute link attitudes.

        y_abs = [alpha1, alpha2, omega1_abs, omega2_abs].
        Internally converts to q=[theta1,theta2].
        """
        alpha1, alpha2, omega1, omega2 = np.asarray(y_abs, dtype=float).reshape(4)
        theta1, theta2 = self.relative_angles_from_absolute(alpha1, alpha2)
        qdot = np.array([omega1, omega2 - omega1], dtype=float)
        u = 0.0 if u_fun is None else float(u_fun(t))
        qddot = self.qddot_relative(theta1, theta2, qdot, u)
        return np.array([omega1, omega2, qddot[0], qddot[0] + qddot[1]], dtype=float)

    # Backward compatible alias used by the LGVI first-step initializer.
    minimal_rhs = rhs_absolute
