"""
Inside (forward) algorithm: compute the probability that cfg generates terminal string x.

The inside table alpha[i, j, A] = Pr[A =>* x[i..j]] is filled bottom-up.
The string probability is alpha[0, n-1, root].

Same batched-split-offset structure as cyk.py but accumulates float64 probabilities
instead of boolean OR. Rule probabilities are uniform: P(A -> r) = 1/|R(A)| for
original NTs, and 1.0 for auxiliary / preterminal NTs from binarization.

Numerical range: for cfg3 strings of typical length ~250-300, the root probability
is ~10^{-50} to 10^{-120}, comfortably within float64 range (~2e-308). No scaling needed.
"""

from collections import defaultdict

import numpy as np

from dp.binarize import binarize


def string_prob(x, cfg) -> float:
    """
    Return Pr[x | cfg] = alpha[0, n-1, root] under the uniform-rule CFG distribution.
    Returns 0.0 for strings not in L(cfg).
    """
    bcfg = binarize(cfg)
    n = len(x)

    nt_list = sorted(bcfg.all_nts)
    nt_idx = {nt: i for i, nt in enumerate(nt_list)}
    N = len(nt_list)
    root_id = nt_idx[bcfg.root]

    A_ids = np.array([nt_idx[A] for A, B, C, _ in bcfg.binary_rules])
    B_ids = np.array([nt_idx[B] for A, B, C, _ in bcfg.binary_rules])
    C_ids = np.array([nt_idx[C] for A, B, C, _ in bcfg.binary_rules])
    probs = np.array([p for _, _, _, p in bcfg.binary_rules], dtype=np.float64)  # (R,)

    # Group rule positions by LHS NT
    rules_by_A: dict[int, np.ndarray] = {}
    tmp: dict = defaultdict(list)
    for r, a in enumerate(A_ids):
        tmp[int(a)].append(r)
    for a, rs in tmp.items():
        rules_by_A[a] = np.array(rs)

    alpha = np.zeros((n, n, N), dtype=np.float64)
    x_arr = np.asarray(x)

    # Base case: preterminals
    for t, pt in bcfg.preterminals.items():
        pos = np.where(x_arr == t)[0]
        if pos.size:
            alpha[pos, pos, nt_idx[pt]] = 1.0

    i_base = np.arange(n)

    for l in range(2, n + 1):
        n_spans = n - l + 1
        i_arr = i_base[:n_spans]

        I, D = np.meshgrid(i_arr, np.arange(l - 1), indexing="ij")  # (n_spans, l-1)
        K = I + D
        J = I + (l - 1)

        left = alpha[I, K]       # (n_spans, l-1, N)
        right = alpha[K + 1, J]  # (n_spans, l-1, N)

        # prob-weighted product for each (span, split, rule)
        left_B = left[:, :, B_ids]     # (n_spans, l-1, R)
        right_C = right[:, :, C_ids]   # (n_spans, l-1, R)
        # weighted[s, d, r] = probs[r] * alpha[i,k,B_r] * alpha[k+1,j,C_r]
        weighted = left_B * right_C * probs  # (n_spans, l-1, R)  — probs broadcasts over axes 0,1

        # Sum over splits: (n_spans, R)
        sum_splits = weighted.sum(axis=1)

        # Scatter into alpha: for each LHS NT A, sum the contributions of its rules
        for A_id, rule_idxs in rules_by_A.items():
            alpha[i_arr, i_arr + (l - 1), A_id] += sum_splits[:, rule_idxs].sum(axis=1)

    return float(alpha[0, n - 1, root_id])
