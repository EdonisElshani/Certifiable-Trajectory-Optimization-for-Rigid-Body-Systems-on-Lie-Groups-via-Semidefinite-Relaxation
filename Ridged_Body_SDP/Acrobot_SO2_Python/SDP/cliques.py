from __future__ import annotations

import numpy as np


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _variable_blocks(idf):

    # Scalar variable-block helpers.

    def R1(k):
        return [idf("c1", k), idf("s1", k)]

    def R2(k):
        return [idf("c2", k), idf("s2", k)]

    def F1(k):
        return [idf("a1", k), idf("b1", k)]

    def F2(k):
        return [idf("a2", k), idf("b2", k)]

    def L0(k):
        return [idf("lam0x", k), idf("lam0y", k)]

    def L12(k):
        return [idf("lam12x", k), idf("lam12y", k)]

    def U(k):
        return [idf("u", k)]

    return R1, R2, F1, F2, L0, L12, U


def _full_I1(R1, R2, F1, F2, L0, L12, U):
    # First baseline clique, sec. 3.1. |I1| = 25
    return (
        R1(0) + R2(0)
        + R1(1) + R2(1)
        + R1(2) + R2(2)
        + F1(0) + F2(0)
        + F1(1) + F2(1)
        + L0(1) + L12(1) + U(1)
    )


def _full_Ik(k, R1, R2, F1, F2, L0, L12, U):
    # Full interior clique I_k, valid for k >= 2, sec. 3.2. |I2| = 21
    return (
        R1(k) + R2(k)
        + R1(k + 1) + R2(k + 1)
        + F1(k - 1) + F2(k - 1)
        + F1(k) + F2(k)
        + L0(k) + L12(k) + U(k)
    )


def _D_all(k, R1, R2, F1, F2, L0, L12, U):
    # Joint dynamics clique, sec. 4.1. |D_all_k| = 17
    return (
        R1(k)
        + F1(k - 1) + F1(k)
        + R2(k)
        + F2(k - 1) + F2(k)
        + L0(k) + L12(k) + U(k)
    )


def _K_12(k, R1, R2, F1, F2):
    # Combined kinematic clique, sec. 4.2. |K12_k| = 12
    return (
        R1(k) + R2(k)
        + R1(k + 1) + R2(k + 1)
        + F1(k) + F2(k)
    )


def _generate_full_Ik_cliques(N, idf):

    # Baseline clique scheme (reduced_maximal_acrobot.pdf Model 2): the full
    # interior clique I_k at every stage k = 1, ..., N-1.

    if N < 2:
        raise ValueError("N must be at least 2 for the full_Ik clique scheme.")

    R1, R2, F1, F2, L0, L12, U = _variable_blocks(idf)
    cliques = []
    for k in range(1, N):
        if k == 1:
            clique = _full_I1(R1, R2, F1, F2, L0, L12, U)
        else:
            clique = _full_Ik(k, R1, R2, F1, F2, L0, L12, U)
        cliques.append(_dedupe_keep_order(clique))
    return cliques


def _generate_protected_hybrid_cliques(N, idf):

    # Protected hybrid clique scheme

    if N < 3:
        raise ValueError(
            "The protected hybrid scheme assumes at least "
            "the two baseline stages I1 and I2 (N >= 3)."
        )

    R1, R2, F1, F2, L0, L12, U = _variable_blocks(idf)

    cliques = [
        _dedupe_keep_order(_full_I1(R1, R2, F1, F2, L0, L12, U)),
        _dedupe_keep_order(_full_Ik(2, R1, R2, F1, F2, L0, L12, U)),
    ]
    for k in range(3, N):
        cliques.append(_dedupe_keep_order(_D_all(k, R1, R2, F1, F2, L0, L12, U)))
        cliques.append(_dedupe_keep_order(_K_12(k, R1, R2, F1, F2)))
    return cliques


def _check_no_duplicate_ids(cliques, scheme):
    # |C| = |unique(C)| for every clique C
    for i, clique in enumerate(cliques):
        if len(clique) != len(set(clique)):
            raise ValueError(
                f"[{scheme}] clique #{i} contains duplicate scalar IDs: {clique}"
            )


def _check_protected_hybrid_structure(cliques, N, idf):

    expected_sizes = [25, 21] + [17, 12] * (N - 3)
    sizes = [len(c) for c in cliques]
    assert sizes == expected_sizes, (
        f"protected_hybrid clique sizes {sizes} != expected {expected_sizes}"
    )
    assert len(cliques) == 2 * N - 4, (
        f"protected_hybrid clique count {len(cliques)} != expected {2 * N - 4}"
    )

    R1, R2, F1, F2, L0, L12, U = _variable_blocks(idf)

    # I1/I2 must equal the full_Ik baseline exactly
    baseline_I1 = _dedupe_keep_order(_full_I1(R1, R2, F1, F2, L0, L12, U))
    baseline_I2 = _dedupe_keep_order(_full_Ik(2, R1, R2, F1, F2, L0, L12, U))
    assert cliques[0] == baseline_I1, "protected_hybrid I1 != full_Ik baseline I1"
    assert cliques[1] == baseline_I2, "protected_hybrid I2 != full_Ik baseline I2"

    def overlap(a, b):
        return set(a) & set(b)

    if len(cliques) > 2:
        expected_I2_D3 = set(R1(3) + R2(3) + F1(2) + F2(2))
        assert overlap(cliques[1], cliques[2]) == expected_I2_D3
        assert len(expected_I2_D3) == 8

    idx = 2
    for k in range(3, N):
        D_k = cliques[idx]
        K_k = cliques[idx + 1]

        expected_DK = set(R1(k) + R2(k) + F1(k) + F2(k))
        assert overlap(D_k, K_k) == expected_DK, f"D_all_{k} inter K12_{k} mismatch"
        assert len(expected_DK) == 8

        if idx + 2 < len(cliques):
            D_next = cliques[idx + 2]
            expected_KD = set(R1(k + 1) + R2(k + 1) + F1(k) + F2(k))
            assert overlap(K_k, D_next) == expected_KD, (
                f"K12_{k} inter D_all_{k + 1} mismatch"
            )
            assert len(expected_KD) == 8

        idx += 2


def _T_12(k, R1, R2, F1, F2, L12):
    # Cross-link translational clique (experimental) |T12_k| = 14
    return (
        R1(k) + F1(k - 1) + F1(k)
        + R2(k) + F2(k - 1) + F2(k)
        + L12(k)
    )


def _T_1x(k, R1, F1, idf):
    # Link-1 horizontal translational clique (experimental) |T1x_k| = 8
    return (
        R1(k) + F1(k - 1) + F1(k)
        + [idf("lam0x", k), idf("lam12x", k)]
    )


def _T_1y(k, R1, F1, idf):
    # Link-1 vertical translational clique (experimental) |T1y_k| = 8
    return (
        R1(k) + F1(k - 1) + F1(k)
        + [idf("lam0y", k), idf("lam12y", k)]
    )


def _Q_1(k, R1, L0, L12, U, idf):
    # Link-1 rotational clique (experimental) |Q1_k| = 9
    return (
        R1(k) + [idf("b1", k - 1), idf("b1", k)]
        + L0(k) + L12(k) + U(k)
    )


def _Q_2(k, R2, L12, U, idf):
    # Link-2 rotational clique (experimental) |Q2_k| = 7
    return (
        R2(k) + [idf("b2", k - 1), idf("b2", k)]
        + L12(k) + U(k)
    )


def _K_1(k, R1, F1):
    # Link-1 kinematic clique (experimental) |K1_k| = 6
    return R1(k) + R1(k + 1) + F1(k)


def _K_2(k, R2, F2):
    # Link-2 kinematic clique (experimental) |K2_k| = 6
    return R2(k) + R2(k + 1) + F2(k)


def _generate_equation_family_extreme_cliques(N, idf):

    if N < 3:
        raise ValueError(
            "The equation_family_extreme scheme assumes at least "
            "the two baseline stages I1 and I2 (N >= 3)."
        )

    R1, R2, F1, F2, L0, L12, U = _variable_blocks(idf)

    cliques = [
        _full_I1(R1, R2, F1, F2, L0, L12, U),
        _full_Ik(2, R1, R2, F1, F2, L0, L12, U),
    ]
    for k in range(3, N):
        cliques.append(_T_12(k, R1, R2, F1, F2, L12))
        cliques.append(_T_1x(k, R1, F1, idf))
        cliques.append(_T_1y(k, R1, F1, idf))
        cliques.append(_Q_1(k, R1, L0, L12, U, idf))
        cliques.append(_Q_2(k, R2, L12, U, idf))
        cliques.append(_K_1(k, R1, F1))
        cliques.append(_K_2(k, R2, F2))
    return cliques


def _check_equation_family_extreme_structure(cliques, N, idf):

    expected_count = 2 + 7 * (N - 3)
    if len(cliques) != expected_count:
        raise ValueError(
            f"equation_family_extreme clique count {len(cliques)} != "
            f"expected {expected_count}"
        )

    expected_sizes = [25, 21] + [14, 8, 8, 9, 7, 6, 6] * (N - 3)
    sizes = [len(c) for c in cliques]
    if sizes != expected_sizes:
        raise ValueError(
            f"equation_family_extreme clique sizes {sizes} != "
            f"expected {expected_sizes}"
        )

    R1, R2, F1, F2, L0, L12, U = _variable_blocks(idf)

    baseline_I1 = _dedupe_keep_order(_full_I1(R1, R2, F1, F2, L0, L12, U))
    baseline_I2 = _dedupe_keep_order(_full_Ik(2, R1, R2, F1, F2, L0, L12, U))
    if cliques[0] != baseline_I1:
        raise ValueError("equation_family_extreme I1 != full_Ik baseline I1")
    if cliques[1] != baseline_I2:
        raise ValueError("equation_family_extreme I2 != full_Ik baseline I2")

    # No scalar ID duplicated inside any raw clique.
    _check_no_duplicate_ids(cliques, "equation_family_extreme")

    # No duplicate complete clique may be silently discarded for this scheme.
    seen_cliques = set()
    for i, clique in enumerate(cliques):
        key = tuple(clique)
        if key in seen_cliques:
            raise ValueError(
                f"equation_family_extreme clique #{i} duplicates an earlier "
                "complete clique; duplicate complete cliques must raise, not "
                "be silently discarded, for this scheme"
            )
        seen_cliques.add(key)

    # Each seven-clique stage group must be in the exact order
    # T12_k, T1x_k, T1y_k, Q1_k, Q2_k, K1_k, K2_k, and match the helper-
    # generated definitions exactly, including scalar variable order.
    idx = 2
    for k in range(3, N):
        expected_group = [
            _T_12(k, R1, R2, F1, F2, L12),
            _T_1x(k, R1, F1, idf),
            _T_1y(k, R1, F1, idf),
            _Q_1(k, R1, L0, L12, U, idf),
            _Q_2(k, R2, L12, U, idf),
            _K_1(k, R1, F1),
            _K_2(k, R2, F2),
        ]
        actual_group = cliques[idx: idx + 7]
        if actual_group != expected_group:
            raise ValueError(
                f"equation_family_extreme stage k={k} group order/content "
                f"mismatch: got {actual_group}, expected {expected_group}"
            )
        idx += 7


def get_cliques_for_cstss(N: int, params: dict):

    # Build the SELF clique list for CSTSS.

    idf = params["id"]
    scheme = params.get("clique_scheme", "protected_hybrid")

    if scheme == "full_Ik":
        cliques = _generate_full_Ik_cliques(N, idf)
    elif scheme == "protected_hybrid":
        cliques = _generate_protected_hybrid_cliques(N, idf)
        _check_protected_hybrid_structure(cliques, N, idf)
    elif scheme == "equation_family_extreme":
        cliques = _generate_equation_family_extreme_cliques(N, idf)
        _check_equation_family_extreme_structure(cliques, N, idf)
    else:
        raise ValueError(f"Unknown clique_scheme: {scheme!r}")

    _check_no_duplicate_ids(cliques, scheme)

    unique_cliques = []
    seen = set()
    for clique in cliques:
        key = tuple(clique)
        if key not in seen:
            unique_cliques.append(clique)
            seen.add(key)

    sizes = [len(c) for c in unique_cliques]

    print("\n" + "=" * 80)
    print(f"SELF CLIQUE DEBUG: clique_scheme = {scheme!r}")
    print("=" * 80)
    print(f"number of SELF cliques: {len(unique_cliques)}")
    if sizes:
        print(f"min clique size:      {min(sizes)}")
        print(f"max clique size:      {max(sizes)}")
        print(f"mean clique size:     {np.mean(sizes):.2f}")
        print(f"clique sizes:         {sizes}")
    else:
        print("No cliques created.")
    print("=" * 80 + "\n")

    return unique_cliques
