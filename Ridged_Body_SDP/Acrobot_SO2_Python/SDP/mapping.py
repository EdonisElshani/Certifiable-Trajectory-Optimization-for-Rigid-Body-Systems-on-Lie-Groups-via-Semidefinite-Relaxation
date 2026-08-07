from __future__ import annotations

import numpy as np


# k = 0,...,N
LONG_PREFIXES = ["c1", "s1", "c2", "s2"]

# k = 0,...,N-1
MID_PREFIXES = ["a1", "b1", "a2", "b2"]

# k = 1,...,N-1
SHORT_PREFIXES = ["lam0x", "lam0y", "lam12x", "lam12y", "u"]


PREFIX_K0 = {
    "c1": 0,
    "s1": 0,
    "c2": 0,
    "s2": 0,
    "a1": 0,
    "b1": 0,
    "a2": 0,
    "b2": 0,
    "lam0x": 1,
    "lam0y": 1,
    "lam12x": 1,
    "lam12y": 1,
    "u": 1,
}


def get_var_mapping_and_dict(N: int):
    """
    Create scalar variables for the reduced x-free/v-free SO(2) Acrobot POP.

    Decision variables:

        R_i,k:
            c_i,k, s_i,k, k = 0,...,N

        F_i,k:
            a_i,k, b_i,k, k = 0,...,N-1

        lambda_k, u_k:
            k = 1,...,N-1

    This matches the thesis formulation and the old monolithic reduced SDP code.
    """
    if N < 2:
        raise ValueError("N must be at least 2, because u_k and lambda_k use k=1,...,N-1.")

    var_start_dict = {}
    var_mapping = {}
    cnt = 1

    long_list = list(range(N + 1))
    mid_list = list(range(N))
    short_list = list(range(1, N))

    fmt = {
        "c1": "c_{{1,{k}}}",
        "s1": "s_{{1,{k}}}",
        "c2": "c_{{2,{k}}}",
        "s2": "s_{{2,{k}}}",
        "a1": "a_{{1,{k}}}",
        "b1": "b_{{1,{k}}}",
        "a2": "a_{{2,{k}}}",
        "b2": "b_{{2,{k}}}",
        "lam0x": "\\lambda_{{0,x,{k}}}",
        "lam0y": "\\lambda_{{0,y,{k}}}",
        "lam12x": "\\lambda_{{12,x,{k}}}",
        "lam12y": "\\lambda_{{12,y,{k}}}",
        "u": "u_{{{k}}}",
    }

    ordered_prefixes = [
        ("c1", long_list),
        ("s1", long_list),
        ("c2", long_list),
        ("s2", long_list),
        ("a1", mid_list),
        ("b1", mid_list),
        ("a2", mid_list),
        ("b2", mid_list),
        ("lam0x", short_list),
        ("lam0y", short_list),
        ("lam12x", short_list),
        ("lam12y", short_list),
        ("u", short_list),
    ]

    for prefix, klist in ordered_prefixes:
        var_start_dict[prefix] = cnt
        for k in klist:
            var_mapping[cnt] = fmt[prefix].format(k=k)
            cnt += 1

    total_var_num = cnt - 1
    var_start_dict["N"] = int(N)

    return var_mapping, var_start_dict, total_var_num


def get_id(prefix: str, k: int, var_start_dict: dict, prefix_k0: dict = PREFIX_K0) -> int:
    """
    Return 1-based SPOT variable index.
    """
    if prefix not in prefix_k0:
        raise KeyError(f"Unknown prefix: {prefix}")

    return var_start_dict[prefix] + (int(k) - prefix_k0[prefix])


def attach_mapping_to_params(params: dict) -> dict:
    """
    Add variable mapping data to params.
    """
    N = int(params["N"])

    var_mapping, var_start_dict, total_var_num = get_var_mapping_and_dict(N)

    params["var_mapping"] = var_mapping
    params["var_start_dict"] = var_start_dict
    params["total_var_num"] = total_var_num
    params["prefix_k0"] = PREFIX_K0

    params["id"] = lambda prefix, k: get_id(
        prefix=prefix,
        k=k,
        var_start_dict=var_start_dict,
        prefix_k0=PREFIX_K0,
    )

    return params


def get_remapped_ids(params: dict) -> np.ndarray:
    """
    Remap variables into timestep-grouped order for sparse/chordal handling.
    """
    N = int(params["N"])
    total_var_num = int(params["total_var_num"])
    idf = params["id"]

    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1

    for k in range(N + 1):
        ids_remap[idf("c1", k) - 1] = idx
        idx += 1
        ids_remap[idf("s1", k) - 1] = idx
        idx += 1
        ids_remap[idf("c2", k) - 1] = idx
        idx += 1
        ids_remap[idf("s2", k) - 1] = idx
        idx += 1

        if k < N:
            ids_remap[idf("a1", k) - 1] = idx
            idx += 1
            ids_remap[idf("b1", k) - 1] = idx
            idx += 1
            ids_remap[idf("a2", k) - 1] = idx
            idx += 1
            ids_remap[idf("b2", k) - 1] = idx
            idx += 1

        if 1 <= k <= N - 1:
            ids_remap[idf("lam0x", k) - 1] = idx
            idx += 1
            ids_remap[idf("lam0y", k) - 1] = idx
            idx += 1
            ids_remap[idf("lam12x", k) - 1] = idx
            idx += 1
            ids_remap[idf("lam12y", k) - 1] = idx
            idx += 1
            ids_remap[idf("u", k) - 1] = idx
            idx += 1

    return ids_remap