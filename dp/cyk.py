"""
CYK membership check: decide whether a terminal string x belongs to L(cfg).

Algorithm: standard CYK on the binarized grammar (see dp/binarize.py).
Complexity: O(n^3 * R) where n = len(x), R = number of binarized rules.

The dp table dp[i, j, A] = True iff nonterminal A can generate x[i..j].
Answer: dp[0, n-1, root].

Performance note: all split offsets for a given span length are batched into a
single (n_spans × (l-1) × N) array fetch, reducing the number of Python-level
numpy calls from O(n^2 * R) to O(n * |NT|). Typical runtime: ~0.3s for n=280.
"""

from collections import defaultdict

import numpy as np

from dp.binarize import binarize


def is_valid(x, cfg) -> bool:
    """Return True iff terminal string x ∈ L(cfg)."""
    bcfg = binarize(cfg)
    n = len(x)

    nt_list = sorted(bcfg.all_nts)
    nt_idx = {nt: i for i, nt in enumerate(nt_list)}
    N = len(nt_list)
    root_id = nt_idx[bcfg.root]

    # Precompute rule index arrays (computed once, reused in every span-length iteration)
    A_ids = np.array([nt_idx[A] for A, B, C, _ in bcfg.binary_rules])
    B_ids = np.array([nt_idx[B] for A, B, C, _ in bcfg.binary_rules])
    C_ids = np.array([nt_idx[C] for A, B, C, _ in bcfg.binary_rules])

    # Group rule positions by their LHS NT for fast scatter
    rules_by_A: dict[int, np.ndarray] = {}
    tmp: dict = defaultdict(list)
    for r, a in enumerate(A_ids):
        tmp[int(a)].append(r)
    for a, rs in tmp.items():
        rules_by_A[a] = np.array(rs)

    dp = np.zeros((n, n, N), dtype=bool)
    x_arr = np.asarray(x)

    # Base case: span length 1 — each position covered by its preterminal
    for t, pt in bcfg.preterminals.items():
        pos = np.where(x_arr == t)[0]
        if pos.size:
            dp[pos, pos, nt_idx[pt]] = True

    i_base = np.arange(n)

    for l in range(2, n + 1):
        n_spans = n - l + 1
        i_arr = i_base[:n_spans]  # start positions: 0 .. n_spans-1

        # Batch all (span, split) pairs for this span length into a 2D grid:
        #   I[s, d] = i_arr[s]  (start position)
        #   K[s, d] = i_arr[s] + d  (split position)
        #   J[s, d] = i_arr[s] + l-1  (end position, constant along d-axis)
        I, D = np.meshgrid(i_arr, np.arange(l - 1), indexing="ij")  # (n_spans, l-1)
        K = I + D
        J = I + (l - 1)

        left = dp[I, K]        # (n_spans, l-1, N): dp[i, k, :]
        right = dp[K + 1, J]   # (n_spans, l-1, N): dp[k+1, j, :]

        # For every rule r, check if (B_r, C_r) fires at any split
        left_B = left[:, :, B_ids]    # (n_spans, l-1, R)
        right_C = right[:, :, C_ids]  # (n_spans, l-1, R)
        both = left_B & right_C       # True where rule r fires at split d for span s

        # OR over all splits → (n_spans, R): does rule r fire at any split for span s?
        any_split = both.any(axis=1)  # (n_spans, R)

        # Scatter into dp: for each LHS NT A, OR the contributions of its rules
        for A_id, rule_idxs in rules_by_A.items():
            contrib = any_split[:, rule_idxs].any(axis=1)  # (n_spans,)
            if contrib.any():
                dp[i_arr, i_arr + (l - 1), A_id] |= contrib

    return bool(dp[0, n - 1, root_id])
