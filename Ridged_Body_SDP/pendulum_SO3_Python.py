import os
import datetime
import numpy as np
import pickle
import time

import sys
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr, numpoly_visualize
from SPOT.PYTHON.naive_extract import naive_extract
from SPOT.PYTHON.robust_extract_CS import robust_extract_CS, ordered_extract_CS

###############################
# Helper Functions and Stubs  #
###############################

# Variable Groups (defined by timestep index in get_var_mapping_and_dict):
# R_k: k=0..N - (r11, r12, ..., r33)
# F_k: k=0..N-1 - (f11, f12, ..., f33)
# u_p_k: k=1..N-1 - (up1, up2, up3)

LONG_PREFIXES = ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]


def get_var_mapping_and_dict(N):

    var_start_dict = {}
    var_mapping = {}
    cnt = 1
    long_list = list(range(N + 1))  # k=0..N (N+1 values)
    f_list = list(range(N))          # F_k: k=0..N-1 (N values)
    u_list = list(range(1, N))       # u_p_k: k=1..N-1 (N-1 values) - NO control at k=0

    fmt = {
        # Rotation matrix R_k components 
        "r11": "r_{{11, {k}}}", "r12": "r_{{12, {k}}}", "r13": "r_{{13, {k}}}",
        "r21": "r_{{21, {k}}}", "r22": "r_{{22, {k}}}", "r23": "r_{{23, {k}}}",
        "r31": "r_{{31, {k}}}", "r32": "r_{{32, {k}}}", "r33": "r_{{33, {k}}}",
        
        # Step rotation matrix F_k components 
        "f11": "f_{{11, {k}}}", "f12": "f_{{12, {k}}}", "f13": "f_{{13, {k}}}",
        "f21": "f_{{21, {k}}}", "f22": "f_{{22, {k}}}", "f23": "f_{{23, {k}}}",
        "f31": "f_{{31, {k}}}", "f32": "f_{{32, {k}}}", "f33": "f_{{33, {k}}}",
        
        # Control input components
        "up1": "u_{{p,1, {k}}}", "up2": "u_{{p,2, {k}}}", "up3": "u_{{p,3, {k}}}",
    }


    oriented_prefixes = [
        # R matrix components (all k=0..N)
        ("r11", long_list), ("r12", long_list), ("r13", long_list),
        ("r21", long_list), ("r22", long_list), ("r23", long_list),
        ("r31", long_list), ("r32", long_list), ("r33", long_list),
        
        # F matrix components (all k=0..N-1)
        ("f11", f_list), ("f12", f_list), ("f13", f_list),
        ("f21", f_list), ("f22", f_list), ("f23", f_list),
        ("f31", f_list), ("f32", f_list), ("f33", f_list),
        
        # Control inputs (all k=1..N-1 ONLY - no control at initial state k=0)
        ("up1", u_list), ("up2", u_list), ("up3", u_list),
    ]

    # Build var_start_dict and var_mapping
    for prefix, klist in oriented_prefixes:
        var_start_dict[prefix] = cnt
        for k in klist:
            var_mapping[cnt] = fmt[prefix].format(k=k)
            cnt += 1

    total_var_num = cnt - 1
    var_start_dict["N"] = N
    return var_mapping, var_start_dict, total_var_num


def get_id(prefix, k, var_start_dict, prefix_k0):

    return var_start_dict[prefix] + (k - prefix_k0[prefix])


def get_remapped_ids(params):

    N = params['N']
    total_var_num = params['total_var_num']
    var_start_dict = params['var_start_dict']
    id_func = params['id']
    
    # Create remapping array: ids_remap[original_id - 1] = new_id
    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1  # New ID counter (1-indexed)
    
    # Iterate through timesteps and assign new IDs in grouped order
    for k in range(N + 1):
        # R_k components 
        ids_remap[id_func("r11", k) - 1] = idx; idx += 1
        ids_remap[id_func("r12", k) - 1] = idx; idx += 1
        ids_remap[id_func("r13", k) - 1] = idx; idx += 1
        ids_remap[id_func("r21", k) - 1] = idx; idx += 1
        ids_remap[id_func("r22", k) - 1] = idx; idx += 1
        ids_remap[id_func("r23", k) - 1] = idx; idx += 1
        ids_remap[id_func("r31", k) - 1] = idx; idx += 1
        ids_remap[id_func("r32", k) - 1] = idx; idx += 1
        ids_remap[id_func("r33", k) - 1] = idx; idx += 1
        
        # F_k components exist for k=0..N-1 
        if k < N:
            ids_remap[id_func("f11", k) - 1] = idx; idx += 1
            ids_remap[id_func("f12", k) - 1] = idx; idx += 1
            ids_remap[id_func("f13", k) - 1] = idx; idx += 1
            ids_remap[id_func("f21", k) - 1] = idx; idx += 1
            ids_remap[id_func("f22", k) - 1] = idx; idx += 1
            ids_remap[id_func("f23", k) - 1] = idx; idx += 1
            ids_remap[id_func("f31", k) - 1] = idx; idx += 1
            ids_remap[id_func("f32", k) - 1] = idx; idx += 1
            ids_remap[id_func("f33", k) - 1] = idx; idx += 1
        
        # u_p_k components exist ONLY for k=1..N-1
        if 1 <= k <= N - 1:
            ids_remap[id_func("up1", k) - 1] = idx; idx += 1
            ids_remap[id_func("up2", k) - 1] = idx; idx += 1
            ids_remap[id_func("up3", k) - 1] = idx; idx += 1
    
    return ids_remap


# ---- Constraint Functions ----

def _u_phys(up_component, params):

    return float(params['u_max']) * up_component


def get_init_constraints(r11_0, r12_0, r13_0, r21_0, r22_0, r23_0, r31_0, r32_0, r33_0, params):

    # Extract initial constraint values from R_0_init matrix
    R_0_init_val = params['R_0_init']
    
    eqs = [
        # Initial conditions: R_0 = R_0_init (9 constraints)
        r11_0 - R_0_init_val[0, 0],
        r12_0 - R_0_init_val[0, 1],
        r13_0 - R_0_init_val[0, 2],
        r21_0 - R_0_init_val[1, 0],
        r22_0 - R_0_init_val[1, 1],
        r23_0 - R_0_init_val[1, 2],
        r31_0 - R_0_init_val[2, 0],
        r32_0 - R_0_init_val[2, 1],
        r33_0 - R_0_init_val[2, 2],
    ]
    eq_mask = [1] * 9
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_init_F_constraints(f11_0, f12_0, f13_0, f21_0, f22_0, f23_0, f31_0, f32_0, f33_0, params):

    # Extract initial constraint values from F_0_init matrix
    F_0_init_val = params['F_0_init']
    
    eqs = [
        # Initial conditions: F_0 = F_0_init (9 constraints)
        f11_0 - F_0_init_val[0, 0],
        f12_0 - F_0_init_val[0, 1],
        f13_0 - F_0_init_val[0, 2],
        f21_0 - F_0_init_val[1, 0],
        f22_0 - F_0_init_val[1, 1],
        f23_0 - F_0_init_val[1, 2],
        f31_0 - F_0_init_val[2, 0],
        f32_0 - F_0_init_val[2, 1],
        f33_0 - F_0_init_val[2, 2],
    ]
    eq_mask = [1] * 9
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_discrete_euler_lagrange(r31_k, r32_k, r33_k, 
                                f12_km1, f13_km1, f21_km1, f23_km1, f32_km1, f31_km1,
                                f12_k, f13_k, f21_k, f23_k, f32_k, f31_k,
                                up1_k, up2_k, up3_k, params):

    # Extract diagonal elements of J_d inertia matrix
    delta_1 = params['J_d'][0, 0]
    delta_2 = params['J_d'][1, 1]
    delta_3 = params['J_d'][2, 2]

    # Extract physical parameters
    h = params['dt']  # Timestep (use dt from params)
    m = params['m']
    g = params['g']
    rho_c = params['rho_c']
    rho_1, rho_2, rho_3 = rho_c[0], rho_c[1], rho_c[2]

    # Compute moment M_k from R_k components (r31_k, r32_k, r33_k)
    M_1 = m * g * (rho_2 * r33_k - rho_3 * r32_k)
    M_2 = m * g * (rho_3 * r31_k - rho_1 * r33_k)
    M_3 = m * g * (rho_1 * r32_k - rho_2 * r31_k)

    # Rescale the normalized control input to physical units.
    up1_phys = _u_phys(up1_k, params)
    up2_phys = _u_phys(up2_k, params)
    up3_phys = _u_phys(up3_k, params)

    # Compute control moment u_k from R_k and physical control input u_p_k
    u_1 = r32_k * up3_phys - r33_k * up2_phys
    u_2 = r33_k * up1_phys - r31_k * up3_phys
    u_3 = r31_k * up2_phys - r32_k * up1_phys
    
    h_sq = h**2
    
    eqs = [
        # Discrete Euler-Lagrange constraint 1 (equations 7.43)
        delta_2 * f32_k - delta_3 * f23_k - delta_3 * f32_km1 + delta_2 * f23_km1 - h_sq * (M_1 + u_1),
        # Discrete Euler-Lagrange constraint 2 (equation 7.44)
        delta_3 * f13_k - delta_1 * f31_k - delta_1 * f13_km1 + delta_3 * f31_km1 - h_sq * (M_2 + u_2),
        # Discrete Euler-Lagrange constraint 3 (equation 7.45)
        delta_1 * f21_k - delta_2 * f12_k - delta_2 * f21_km1 + delta_1 * f12_km1 - h_sq * (M_3 + u_3),
    ]
    
    eq_mask = [1, 1, 1]
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_kinematic_constraints(r11_km1, r12_km1, r13_km1, r21_km1, r22_km1, r23_km1, r31_km1, r32_km1, r33_km1,
                               r11_k, r12_k, r13_k, r21_k, r22_k, r23_k, r31_k, r32_k, r33_k,
                               f11_km1, f12_km1, f13_km1, f21_km1, f22_km1, f23_km1, f31_km1, f32_km1, f33_km1, params):

    eqs = [
        # Row 1 of R_k = R_{k-1} * F_{k-1}
        r11_k - r11_km1*f11_km1 - r12_km1*f21_km1 - r13_km1*f31_km1,
        r12_k - r11_km1*f12_km1 - r12_km1*f22_km1 - r13_km1*f32_km1,
        r13_k - r11_km1*f13_km1 - r12_km1*f23_km1 - r13_km1*f33_km1,
        # Row 2 of R_k = R_{k-1} * F_{k-1}
        r21_k - r21_km1*f11_km1 - r22_km1*f21_km1 - r23_km1*f31_km1,
        r22_k - r21_km1*f12_km1 - r22_km1*f22_km1 - r23_km1*f32_km1,
        r23_k - r21_km1*f13_km1 - r22_km1*f23_km1 - r23_km1*f33_km1,
        # Row 3 of R_k = R_{k-1} * F_{k-1}
        r31_k - r31_km1*f11_km1 - r32_km1*f21_km1 - r33_km1*f31_km1,
        r32_k - r31_km1*f12_km1 - r32_km1*f22_km1 - r33_km1*f32_km1,
        r33_k - r31_km1*f13_km1 - r32_km1*f23_km1 - r33_km1*f33_km1,
    ]
    
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_SO3_orthogonality_constraints_R(r11_k, r12_k, r13_k, r21_k, r22_k, r23_k, r31_k, r32_k, r33_k, params):

    eqs = [
        # Column norms (unit vectors): 3 constraints
        r11_k**2 + r21_k**2 + r31_k**2 - 1,
        r12_k**2 + r22_k**2 + r32_k**2 - 1,
        r13_k**2 + r23_k**2 + r33_k**2 - 1,
        # Column orthogonality: 3 constraints
        r11_k*r12_k + r21_k*r22_k + r31_k*r32_k,
        r12_k*r13_k + r22_k*r23_k + r32_k*r33_k,
        r11_k*r13_k + r21_k*r23_k + r31_k*r33_k,
        # Cross product constraints (equations 7.34-7.42): 9 constraints
        r21_k*r32_k - r31_k*r22_k - r13_k,
        r31_k*r12_k - r11_k*r32_k - r23_k,
        r11_k*r22_k - r21_k*r12_k - r33_k,
        r22_k*r33_k - r32_k*r23_k - r11_k,
        r32_k*r13_k - r12_k*r33_k - r21_k,
        r12_k*r23_k - r22_k*r13_k - r31_k,
        r23_k*r31_k - r33_k*r21_k - r12_k,
        r33_k*r11_k - r13_k*r31_k - r22_k,
        r13_k*r21_k - r23_k*r11_k - r32_k,
    ]
    
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_SO3_orthogonality_constraints_F(f11_k, f12_k, f13_k, f21_k, f22_k, f23_k, f31_k, f32_k, f33_k, params):

    eqs = [
        # Column norms (unit vectors): 3 constraints
        f11_k**2 + f21_k**2 + f31_k**2 - 1,
        f12_k**2 + f22_k**2 + f32_k**2 - 1,
        f13_k**2 + f23_k**2 + f33_k**2 - 1,
        # Column orthogonality: 3 constraints
        f11_k*f12_k + f21_k*f22_k + f31_k*f32_k,
        f12_k*f13_k + f22_k*f23_k + f32_k*f33_k,
        f11_k*f13_k + f21_k*f23_k + f31_k*f33_k,
        # Cross product constraints (from 7.18.k): 9 constraints
        f21_k*f32_k - f31_k*f22_k - f13_k,
        f31_k*f12_k - f11_k*f32_k - f23_k,
        f11_k*f22_k - f21_k*f12_k - f33_k,
        f22_k*f33_k - f32_k*f23_k - f11_k,
        f32_k*f13_k - f12_k*f33_k - f21_k,
        f12_k*f23_k - f22_k*f13_k - f31_k,
        f23_k*f31_k - f33_k*f21_k - f12_k,
        f33_k*f11_k - f13_k*f31_k - f22_k,
        f13_k*f21_k - f23_k*f11_k - f32_k,
    ]
    
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ineqs = []
    
    return eqs, ineqs, eq_mask


def get_step_angle_bound_constraint(f11_k, f22_k, f33_k, params):

    f_c_min = params['f_c_min']
    
    # Step angle bound: (trace(F_k) - 1) / 2 - f_c_min >= 0
    # trace(F_k) relates to rotation angle in SO(3)
    trace_F_k = f11_k + f22_k + f33_k
    ineqs = [
        (trace_F_k - 1) / 2 - f_c_min,
    ]
    
    eqs = []
    eq_mask = []
    
    return eqs, ineqs, eq_mask


def get_control_bounds(up1_k, up2_k, up3_k, params):

    ineqs = [
        1.0 - (up1_k**2 + up2_k**2 + up3_k**2),
    ]

    eqs = []
    eq_mask = []

    return eqs, ineqs, eq_mask


# ---- Objective (same cost structure as SPOT_MPC_Acrobot/SDP/objective.py) ----

def rotation_tracking_cost_so3(r_components, r_des_components, weight):

    return weight * sum(
        (r - rd) ** 2 for r, rd in zip(r_components, r_des_components)
    )


def step_tracking_cost_so3(f_components, f_des_components, weight):

    return weight * sum(
        (f - fd) ** 2 for f, fd in zip(f_components, f_des_components)
    )


def build_objective(v, params, N):

    rho_R = float(params['rho_R'])
    rho_F = float(params['rho_F'])
    alpha_R = float(params['alpha_R'])
    alpha_F = float(params['alpha_F'])
    gamma = float(params['gamma'])

    R_des = params['R_des']
    F_des = params['F_des']
    R_des_entries = [R_des[i, j] for i in range(3) for j in range(3)]
    F_des_entries = [F_des[i, j] for i in range(3) for j in range(3)]

    def R_entries(k):
        return [v("r" + str(i + 1) + str(j + 1), k) for i in range(3) for j in range(3)]

    def F_entries(k):
        return [v("f" + str(i + 1) + str(j + 1), k) for i in range(3) for j in range(3)]

    obj = 0.0

    # Terminal R tracking: rho_R * ||R_N - R_des||_F^2.
    obj += rotation_tracking_cost_so3(R_entries(N), R_des_entries, rho_R)

    # Terminal F tracking: rho_F * ||F_{N-1} - F_des||_F^2.
    obj += step_tracking_cost_so3(F_entries(N - 1), F_des_entries, rho_F)

    # Stage R tracking (k=0..N-1).
    for k in range(N):
        obj += rotation_tracking_cost_so3(R_entries(k), R_des_entries, alpha_R)

    # Stage F tracking (k=0..N-2).
    for k in range(N - 1):
        obj += step_tracking_cost_so3(F_entries(k), F_des_entries, alpha_F)

    # Control effort on the normalized control (k=1..N-1, no control at k=0),
    # same (1/gamma) * ||u_normalized||^2 form as SPOT_MPC_Acrobot.
    for k in range(1, N):
        obj += (1.0 / gamma) * (
            v("up1", k) ** 2 + v("up2", k) ** 2 + v("up3", k) ** 2
        )

    return obj


def extract_solution_variables(v_opt, var_start_dict, N, id_func, u_max):

    solution_dict = {
        'R': {},                # Rotation matrices R_k for k=0..N
        'F': {},                # Step rotation matrices F_k for k=0..N-1
        'u_p': {},              # Physical control moment u_p_k = u_max * normalized, k=1..N-1
        'u_p_normalized': {},   # Raw normalized SDP control variable in [-1, 1], k=1..N-1
    }

    # Extract R_k (k=0..N)
    for k in range(N + 1):
        R_k = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                prefix = "r" + str(i+1) + str(j+1)
                var_id = id_func(prefix, k)
                R_k[i, j] = v_opt[var_id - 1]
        solution_dict['R'][k] = R_k

    # Extract F_k (k=0..N-1)
    for k in range(N):
        F_k = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                prefix = "f" + str(i+1) + str(j+1)
                var_id = id_func(prefix, k)
                F_k[i, j] = v_opt[var_id - 1]
        solution_dict['F'][k] = F_k

    # Extract u_p_k (k=1..N-1), normalized in the solver, rescaled to physical here.
    for k in range(1, N):
        u_p_k_normalized = np.zeros(3)
        for i in range(3):
            prefix = "up" + str(i+1)
            var_id = id_func(prefix, k)
            u_p_k_normalized[i] = v_opt[var_id - 1]
        solution_dict['u_p_normalized'][k] = u_p_k_normalized
        solution_dict['u_p'][k] = float(u_max) * u_p_k_normalized

    return solution_dict


def compute_SO3_errors(sol, N):

    errors = {
        'R': {'orthogonality': [], 'determinant': [], 'norms': [], 'orthog': []},
        'F': {'orthogonality': [], 'determinant': [], 'norms': [], 'orthog': []}
    }
    
    # Check R_k (k=0..N)
    for k in range(N + 1):
        R = sol['R'][k]
        
        # Orthogonality error: ||R^T R - I||_F
        orthog_err = np.linalg.norm(R.T @ R - np.eye(3), 'fro')
        errors['R']['orthogonality'].append(orthog_err)
        
        # Determinant error: |det(R) - 1|
        det_err = np.abs(np.linalg.det(R) - 1.0)
        errors['R']['determinant'].append(det_err)
        
        # Column norm errors: sum(||col_i||^2 - 1)
        norm_err = 0.0
        for i in range(3):
            col_norm_sq = np.sum(R[:, i]**2)
            norm_err += (col_norm_sq - 1.0)**2
        errors['R']['norms'].append(np.sqrt(norm_err))
        
        # Column orthogonality errors: sum(col_i · col_j)^2 for i≠j
        orthog_err = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                dot_prod = np.dot(R[:, i], R[:, j])
                orthog_err += dot_prod**2
        errors['R']['orthog'].append(np.sqrt(orthog_err))
    
    # Check F_k (k=0..N-1)
    for k in range(N):
        F = sol['F'][k]
        
        # Orthogonality error: ||F^T F - I||_F
        orthog_err = np.linalg.norm(F.T @ F - np.eye(3), 'fro')
        errors['F']['orthogonality'].append(orthog_err)
        
        # Determinant error: |det(F) - 1|
        det_err = np.abs(np.linalg.det(F) - 1.0)
        errors['F']['determinant'].append(det_err)
        
        # Column norm errors: sum(||col_i||^2 - 1)
        norm_err = 0.0
        for i in range(3):
            col_norm_sq = np.sum(F[:, i]**2)
            norm_err += (col_norm_sq - 1.0)**2
        errors['F']['norms'].append(np.sqrt(norm_err))
        
        # Column orthogonality errors: sum(col_i · col_j)^2 for i≠j
        orthog_err = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                dot_prod = np.dot(F[:, i], F[:, j])
                orthog_err += dot_prod**2
        errors['F']['orthog'].append(np.sqrt(orthog_err))
    
    return errors


def print_SO3_errors(errors, N):

    print("\n" + "="*70)
    print("SO(3) CONSTRAINT VIOLATION ANALYSIS")
    print("="*70)
    
    print("\n--- ROTATION MATRICES R_k (k=0..N) ---")
    print(f"{'k':<4} {'Orthog||R^TR-I||':<18} {'|det(R)-1|':<15} {'Norm Error':<15} {'Orthog Error':<15}")
    print("-" * 70)
    for k in range(N + 1):
        print(f"{k:<4} {errors['R']['orthogonality'][k]:<18.2e} "
              f"{errors['R']['determinant'][k]:<15.2e} "
              f"{errors['R']['norms'][k]:<15.2e} "
              f"{errors['R']['orthog'][k]:<15.2e}")
    
    print(f"\nMax Orthogonality Error:  {max(errors['R']['orthogonality']):.2e}")
    print(f"Max Determinant Error:    {max(errors['R']['determinant']):.2e}")
    print(f"Max Norm Error:           {max(errors['R']['norms']):.2e}")
    print(f"Max Orthog Error:         {max(errors['R']['orthog']):.2e}")
    
    print("\n--- STEP ROTATION MATRICES F_k (k=0..N-1) ---")
    print(f"{'k':<4} {'Orthog||F^TF-I||':<18} {'|det(F)-1|':<15} {'Norm Error':<15} {'Orthog Error':<15}")
    print("-" * 70)
    for k in range(N):
        print(f"{k:<4} {errors['F']['orthogonality'][k]:<18.2e} "
              f"{errors['F']['determinant'][k]:<15.2e} "
              f"{errors['F']['norms'][k]:<15.2e} "
              f"{errors['F']['orthog'][k]:<15.2e}")
    
    print(f"\nMax Orthogonality Error:  {max(errors['F']['orthogonality']):.2e}")
    print(f"Max Determinant Error:    {max(errors['F']['determinant']):.2e}")
    print(f"Max Norm Error:           {max(errors['F']['norms']):.2e}")
    print(f"Max Orthog Error:         {max(errors['F']['orthog']):.2e}")


def main():
    
    total_start = time.time()

    # --- CSTSS parameters ---
    params = {}
    kappa = 2; params['kappa'] = kappa
    relax_mode = "SOS"; params['relax_mode'] = relax_mode
    cs_mode = "MD"; params['cs_mode'] = cs_mode
    ts_mode = "NON"; params['ts_mode'] = ts_mode
    ts_mom_mode = "NON"; params['ts_mom_mode'] = ts_mom_mode
    ts_eq_mode = "NON"; params['ts_eq_mode'] = ts_eq_mode
    if_solve = True; params['if_solve'] = if_solve
    if_mex = True; params['if_mex'] = if_mex

    # --- System parameters ---
    N = 20; params['N'] = N  
    dt = 0.1; params['dt'] = dt  

    # Variable mapping and indexing
    var_mapping, var_start_dict, total_var_num = get_var_mapping_and_dict(N)
    params['total_var_num'] = total_var_num
    params['var_mapping'] = var_mapping
    params['var_start_dict'] = var_start_dict
    
    # Create prefix_k0: maps each prefix to its starting timestep
    prefix_k0 = {
        # Rotation matrices R_k (k=0..N)
        "r11": 0, "r12": 0, "r13": 0,
        "r21": 0, "r22": 0, "r23": 0,
        "r31": 0, "r32": 0, "r33": 0,
        "f11": 0, "f12": 0, "f13": 0,
        "f21": 0, "f22": 0, "f23": 0,
        "f31": 0, "f32": 0, "f33": 0,
        "up1": 1, "up2": 1, "up3": 1,
    }
    params['prefix_k0'] = prefix_k0
    params['id'] = lambda prefix, k: get_id(prefix, k, var_start_dict, prefix_k0)
    
    # --- Physical parameters for 3D pendulum (from Evaluation.tex) ---
    J_11 = 0.33396; params['J_11'] = J_11
    J_22 = 0.33396; params['J_22'] = J_22
    J_33 = 0.00125; params['J_33'] = J_33
    
    J = np.diag([J_11, J_22, J_33])
    trace_J = J_11 + J_22 + J_33
    J_d = 0.5 * trace_J * np.eye(3) - J
    
    params['J'] = J
    params['J_d'] = J_d
    
    delta_1 = J_d[0, 0]; params['delta_1'] = delta_1
    delta_2 = J_d[1, 1]; params['delta_2'] = delta_2
    delta_3 = J_d[2, 2]; params['delta_3'] = delta_3
    
    # Gravitational and geometric parameters
    m = 1.0; params['m'] = m  
    g = 9.81; params['g'] = g  
    rho_c = np.array([0.0, 0.0, 0.5]); params['rho_c'] = rho_c  
    e_3 = np.array([0.0, 0.0, 1.0]); params['e_3'] = e_3  

    # --- Initial states ---
    init_angle_deg = 5.0; params['init_angle_deg'] = init_angle_deg
    init_angle = np.deg2rad(init_angle_deg)
    R_0_init = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(init_angle), -np.sin(init_angle)],
        [0.0, np.sin(init_angle), np.cos(init_angle)]
    ]); params['R_0_init'] = R_0_init
    
    F_0_init = np.eye(3); params['F_0_init'] = F_0_init
    
    # --- Desired terminal states ---
    
    # Rotation around the x-axis. Store the degree value explicitly so
    target_angle_deg = 180.0; params['target_angle_deg'] = target_angle_deg
    angle = np.deg2rad(target_angle_deg)
    R_des = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]
    ]); params['R_des'] = R_des
    
    # --- Loss coefficients ---
    rho_R = 20.0; params['rho_R'] = rho_R  
    rho_F = 8.0; params['rho_F'] = rho_F  
    alpha_R = 15.0; params['alpha_R'] = alpha_R  
    alpha_F = 2.00; params['alpha_F'] = alpha_F  
    gamma = 0.09; params['gamma'] = gamma  
    
    # Desired final F state
    F_des = np.eye(3); params['F_des'] = F_des  

    # Constraint bounds
    f_c_min = 0.5; params['f_c_min'] = f_c_min  
    u_max = 5.0; params['u_max'] = u_max  

    # --- File management ---
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "Pendulum_SO3_Optimized/" + current_time + "/"
    for directory in ["./data/" + prefix_str, "./markdown/" + prefix_str,
                      "./figs/" + prefix_str, "./logs/" + prefix_str]:
        os.makedirs(directory, exist_ok=True)

    log_path = "./logs/" + prefix_str + "log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("3D Pendulum SO(3) Optimization Problem\n")
        log_file.write("=" * 50 + "\n")
        log_file.write("params: \n")
        log_file.write(str(params) + "\n")

    # --- Get remapping information ---
    ids_remap = get_remapped_ids(params); params['ids_remap'] = ids_remap

    # --- Create NumPolySystem ---
    ps = NumPolySystem(n_vars=total_var_num)
    id_func = params['id']

    def v(prefix, k):
        """Helper function to access variables"""
        return ps.var(id_func(prefix, k) - 1)

    eq_mask_sys = []

    # --- Initial constraints ---
    # Constraint R_0 to initial condition and SO(3) orthogonality
    eqs, ineqs, eq_mask = get_init_constraints(
        v("r11", 0), v("r12", 0), v("r13", 0), 
        v("r21", 0), v("r22", 0), v("r23", 0), 
        v("r31", 0), v("r32", 0), v("r33", 0), 
        params
    )
    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)
    
    # Constraint F_0 to prescribed initial step rotation
    eqs, ineqs, eq_mask = get_init_F_constraints(
        v("f11", 0), v("f12", 0), v("f13", 0),
        v("f21", 0), v("f22", 0), v("f23", 0),
        v("f31", 0), v("f32", 0), v("f33", 0),
        params
    )
    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)

    # --- Main constraint loop ---
    for k in range(1, N + 1):
        # Discrete Euler-Lagrange constraints (k=0..N-2)
        if k < N:
            eqs, ineqs, eq_mask = get_discrete_euler_lagrange(
                v("r31", k), v("r32", k), v("r33", k),
                v("f12", k - 1), v("f13", k - 1), v("f21", k - 1), v("f23", k - 1), 
                v("f32", k - 1), v("f31", k - 1),
                v("f12", k), v("f13", k), v("f21", k), v("f23", k), 
                v("f32", k), v("f31", k),
                v("up1", k), v("up2", k), v("up3", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

        # Kinematic constraints: R_k = R_{k-1} * F_{k-1}
        eqs, ineqs, eq_mask = get_kinematic_constraints(
            v("r11", k - 1), v("r12", k - 1), v("r13", k - 1),
            v("r21", k - 1), v("r22", k - 1), v("r23", k - 1),
            v("r31", k - 1), v("r32", k - 1), v("r33", k - 1),
            v("r11", k), v("r12", k), v("r13", k),
            v("r21", k), v("r22", k), v("r23", k),
            v("r31", k), v("r32", k), v("r33", k),
            v("f11", k - 1), v("f12", k - 1), v("f13", k - 1),
            v("f21", k - 1), v("f22", k - 1), v("f23", k - 1),
            v("f31", k - 1), v("f32", k - 1), v("f33", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # SO(3) orthogonality constraints for R_k
        eqs, ineqs, eq_mask = get_SO3_orthogonality_constraints_R(
            v("r11", k), v("r12", k), v("r13", k),
            v("r21", k), v("r22", k), v("r23", k),
            v("r31", k), v("r32", k), v("r33", k),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # SO(3) orthogonality constraints for F_{k-1}
        eqs, ineqs, eq_mask = get_SO3_orthogonality_constraints_F(
            v("f11", k - 1), v("f12", k - 1), v("f13", k - 1),
            v("f21", k - 1), v("f22", k - 1), v("f23", k - 1),
            v("f31", k - 1), v("f32", k - 1), v("f33", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Step angle bound constraint for F_{k-1}
        eqs, ineqs, eq_mask = get_step_angle_bound_constraint(
            v("f11", k - 1), v("f22", k - 1), v("f33", k - 1),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Control bounds for u_p_k (only for k=1..N-1, no control at k=0)
        if k <= N - 1:  # k=1 to N-1
            eqs, ineqs, eq_mask = get_control_bounds(
                v("up1", k), v("up2", k), v("up3", k),
                params
            )
            for eq in eqs:
                ps.add_eq(eq)
            for ineq in ineqs:
                ps.add_ineq(ineq)
            eq_mask_sys.extend(eq_mask)

    # --- Objective ---
    obj_expr = build_objective(v, params, N)

    ps.set_obj(obj_expr)

    # --- Clean polynomials ---
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    # --- Get supp_rpt data ---
    poly_data = ps.get_supp_rpt_data(kappa)

    print("Construction Finish!")

    # --- Get clique decomposition ---
    cliques = []

    params["cliques"] = cliques

    # --- Run CSTSS ---
    start_time = time.time()
    result, res, coeff_info, aux_info = CSTSS_pybind(
        poly_data, kappa, total_var_num, params
    )
    elapsed_time = time.time() - start_time

    params["aux_info"] = aux_info

    with open(log_path, "a") as log_file:
        result_str = str(result) if isinstance(result, list) else f"{result:.20f}"
        log_file.write(
            f"\nPendulum_SO3 N={N}, Relax={relax_mode}, TS={ts_mode}, "
            f"CS={str(cs_mode)}, result={result_str}, "
            f"operation time={elapsed_time:.5f}, "
            f"mosek time={aux_info.get('mosek_time', 0):.5f}\n"
        )

    # --- Remap clique ids ---
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_remapped = []
        aver_remapped = []
        for clique in aux_info["cliques"]:
            remapped = [params["ids_remap"][i - 1] for i in clique]
            cliques_remapped.append(remapped)
            aver_remapped.append(np.mean(remapped))
        cliques_rank = np.argsort(aver_remapped)
        params["cliques_rank"] = cliques_rank
    else:
        params["cliques_rank"] = []

    # --- Markdown debug (for visualization of constraints/objective) ---
    clique_supp_list = []
    clique_coeff_list = []
    kappa_width = 2 * kappa
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_rank = params["cliques_rank"]
        for i in range(len(cliques_rank)):
            ii = cliques_rank[i]
            sorted_vars = sorted(aux_info["cliques"][ii])
            supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
            for idx_v, j in enumerate(sorted_vars):
                supp[idx_v, -1] = j
            clique_supp_list.append(supp)
            clique_coeff_list.append(np.ones(len(sorted_vars)))

    md_path = "./markdown/" + prefix_str + "opt_problem.md"
    with open(md_path, "w") as md_file:
        md_file.write("equality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_h'], aux_info['coeff_h'], var_mapping, md_file)
        md_file.write("inequality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_g'], aux_info['coeff_g'], var_mapping, md_file)
        md_file.write("objective: \n")
        numpoly_visualize([aux_info['supp_rpt_f']], [aux_info['coeff_f']], var_mapping, md_file)
        md_file.write("cliques: \n")
        numpoly_visualize(clique_supp_list, clique_coeff_list, var_mapping, md_file)

    # --- Early exit if not solving (for debugging problem formulation only) ---
    if not params['if_solve']:
        supp_rpt_f = aux_info.get("supp_rpt_f", None)
        supp_rpt_g = aux_info.get("supp_rpt_g", None)
        supp_rpt_h = aux_info.get("supp_rpt_h", None)
        coeff_f = aux_info.get("coeff_f", None)
        coeff_g = aux_info.get("coeff_g", None)
        coeff_h = aux_info.get("coeff_h", None)
        with open("./data/" + prefix_str + "polys.pkl", "wb") as f:
            pickle.dump({'supp_rpt_f': supp_rpt_f, 'supp_rpt_g': supp_rpt_g,
                         'supp_rpt_h': supp_rpt_h, 'coeff_f': coeff_f,
                         'coeff_g': coeff_g, 'coeff_h': coeff_h}, f)
        params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
        with open("./data/" + prefix_str + "params.pkl", "wb") as f:
            pickle.dump(params_to_save, f)
        print("Debugging mode: Problem formulation saved. No solving performed.")
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.5f} s")
        with open(log_path, "a") as log_file:
            log_file.write(f"Debugging mode (if_solve=False)\ntotal time={total_time:.5f}\n")
        return

    # --- Extract solution (only if solving) ---
    Xopt = res['Xopt']
    yopt = res['yopt']
    Sopt = res['Sopt']

    if relax_mode == 'MOMENT':
        Xs = Xopt
    elif relax_mode == 'SOS':
        Xs = [-S for S in Sopt]

    # --- Build moment matrix reports for extraction ---
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

    if ts_mode == "NON":
        v_opt_robust, output_info_robust = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2)
        with open("./data/" + prefix_str + "v_opt_robust.pkl", "wb") as f:
            pickle.dump(v_opt_robust, f)
        v_opt_ordered, output_info_ordered = ordered_extract_CS(
            Xs, mom_mat_rpt, total_var_num, 1e-2, params.get("cliques_rank", []))
        with open("./data/" + prefix_str + "v_opt_ordered.pkl", "wb") as f:
            pickle.dump(v_opt_ordered, f)
        print("Ordered solution (v_opt_ordered):")
        print(v_opt_ordered)

    v_opt_naive, output_info_naive = naive_extract(Xs, aux_info['mon_rpt'], aux_info['ts_info'], total_var_num)
    print(v_opt_naive)
    with open("./data/" + prefix_str + "v_opt_naive.pkl", "wb") as f:
        pickle.dump(v_opt_naive, f)

    # --- Extract and display solution variables (all three methods) ---
    sol_naive = extract_solution_variables(v_opt_naive, var_start_dict, N, id_func, params['u_max'])

    # Extract with ordered and robust methods if available (ts_mode == "NON")
    if ts_mode == "NON":
        sol_ordered = extract_solution_variables(v_opt_ordered, var_start_dict, N, id_func, params['u_max'])
        sol_robust = extract_solution_variables(v_opt_robust, var_start_dict, N, id_func, params['u_max'])
    else:
        sol_ordered = None
        sol_robust = None
    
    print("\n" + "="*60)
    print("SOLUTION VARIABLES (NAIVE EXTRACTION)")
    print("="*60)
    for k in range(N + 1):
        print(f"\nk={k}:")
        print(f"  R_{k} =\n{sol_naive['R'][k]}")
        if k < N:
            print(f"  F_{k} =\n{sol_naive['F'][k]}")
        if 1 <= k < N:
            up = sol_naive['u_p'][k]
            print(f"  u_p_{k} = [{up[0]:.6f}, {up[1]:.6f}, {up[2]:.6f}]")
    
    # --- Compute and display SO(3) constraint violations (all three methods) ---
    errors_naive = compute_SO3_errors(sol_naive, N)
    print_SO3_errors(errors_naive, N)
    
    if sol_ordered is not None:
        print("\n" + "="*60)
        print("SOLUTION VARIABLES (ORDERED EXTRACTION)")
        print("="*60)
        for k in range(N + 1):
            print(f"\nk={k}:")
            print(f"  R_{k} =\n{sol_ordered['R'][k]}")
            if k < N:
                print(f"  F_{k} =\n{sol_ordered['F'][k]}")
            if 1 <= k < N:
                up = sol_ordered['u_p'][k]
                print(f"  u_p_{k} = [{up[0]:.6f}, {up[1]:.6f}, {up[2]:.6f}]")
        
        errors_ordered = compute_SO3_errors(sol_ordered, N)
        print_SO3_errors(errors_ordered, N)
    
    if sol_robust is not None:
        print("\n" + "="*60)
        print("SOLUTION VARIABLES (ROBUST EXTRACTION)")
        print("="*60)
        for k in range(N + 1):
            print(f"\nk={k}:")
            print(f"  R_{k} =\n{sol_robust['R'][k]}")
            if k < N:
                print(f"  F_{k} =\n{sol_robust['F'][k]}")
            if 1 <= k < N:
                up = sol_robust['u_p'][k]
                print(f"  u_p_{k} = [{up[0]:.6f}, {up[1]:.6f}, {up[2]:.6f}]")
        
        errors_robust = compute_SO3_errors(sol_robust, N)
        print_SO3_errors(errors_robust, N)

    with open("./data/" + prefix_str + "data.pkl", "wb") as f:
        pickle.dump({'aux_info': aux_info, 'mom_mat_rpt': mom_mat_rpt,
                     'mom_mat_num': mom_mat_num, 'total_var_num': total_var_num}, f)

    supp_rpt_f = aux_info.get("supp_rpt_f", None)
    supp_rpt_g = aux_info.get("supp_rpt_g", None)
    supp_rpt_h = aux_info.get("supp_rpt_h", None)
    coeff_f = aux_info.get("coeff_f", None)
    coeff_g = aux_info.get("coeff_g", None)
    coeff_h = aux_info.get("coeff_h", None)
    with open("./data/" + prefix_str + "polys.pkl", "wb") as f:
        pickle.dump({'supp_rpt_f': supp_rpt_f, 'supp_rpt_g': supp_rpt_g,
                     'supp_rpt_h': supp_rpt_h, 'coeff_f': coeff_f,
                     'coeff_g': coeff_g, 'coeff_h': coeff_h}, f)
    params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
    with open("./data/" + prefix_str + "params.pkl", "wb") as f:
        pickle.dump(params_to_save, f)

    print(f"Optimization completed in {elapsed_time:.5f}s")
    print(f"Objective value: {result:.20f}")
    
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.5f} s")
    with open(log_path, "a") as log_file:
        log_file.write(f"total time={total_time:.5f}\n")


if __name__ == "__main__":
    main()
