"""
Result 3: per-token KL divergence between true CFG distribution and GPT's predictions.

At each position c in a test string x[0..n-1]:

  P_true(t | x[0..c-1])  — computed here via a prefix inside algorithm
  P_model(t | x[0..c-1]) — softmax of GPT logits at position c, renormalised over {1,2,3}

KL(P_true || P_model) is averaged over all positions and all test strings.

Paper targets (Figure 4):
  cfg3b, GPT_rot: ~0.00008 nats/token
  cfg3f, GPT_rot: ~0.00455 nats/token

──────────────────────────────────────────────────
HOW THE PREFIX ALGORITHM WORKS
──────────────────────────────────────────────────

We want P(next = t | prefix x[0..c-1]).  This equals:

    unnorm(t)   total probability mass in L(G) of strings that start with x[0..c-1]+t
    ─────────── = ────────────────────────────────────────────────────────────────────
    Σ_t unnorm(t)   (normalise)

unnorm(t) = right(root, 0)  where:

    right(A, start) = P(A generates any string that starts with y[start..c])
                      y = x[0..c-1] + [t]

Recursion on binary rules A → B C:

    right(A, start) = Σ_{A→BC} p ×
        [   col[start, B]                                          ← B covers y[start..c] exactly;
          + Σ_{k=start}^{c-1} alpha[start,k,B] × right(C,k+1)   ←  B covers y[start..k],
        ]                                                             C handles the rest

    right(A, c+1) = 1.0  for all A  (nothing left to match; A generates any continuation)

    col[start, A] = P(A generates exactly y[start..c])
                  = precomputed bottom-up using alpha[i,j,A] for j < c (from the full inside table)
                    plus the base col[c, PT_t] = 1.0

alpha[i,j,A] is the standard inside table for the FULL test string (computed once per string).
For spans entirely within x[0..c-1] (j < c), alpha is identical to the prefix inside table.
"""

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg
from dp.binarize import binarize, BinarizedCFG
from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0
EOS_TOKEN = 4
TERMINALS = (1, 2, 3)


# ── Rule preprocessing (done once per grammar) ────────────────────────────────

def _prep_rules(bcfg: BinarizedCFG, nt_list: list, nt_idx: dict):
    """Build numpy arrays for vectorised inside / right computations."""
    A_ids = np.array([nt_idx[A] for A, B, C, _ in bcfg.binary_rules], dtype=np.intp)
    B_ids = np.array([nt_idx[B] for A, B, C, _ in bcfg.binary_rules], dtype=np.intp)
    C_ids = np.array([nt_idx[C] for A, B, C, _ in bcfg.binary_rules], dtype=np.intp)
    probs = np.array([p           for _, _, _, p in bcfg.binary_rules], dtype=np.float64)

    rules_by_A: Dict[int, np.ndarray] = {}
    tmp: Dict = defaultdict(list)
    for r, a in enumerate(A_ids):
        tmp[int(a)].append(r)
    for a, rs in tmp.items():
        rules_by_A[a] = np.array(rs, dtype=np.intp)

    return A_ids, B_ids, C_ids, probs, rules_by_A


# ── Full inside table (one pass per test string) ──────────────────────────────

def build_inside_table(
    x: list,
    bcfg: BinarizedCFG,
    nt_idx: dict,
    nt_list: list,
    A_ids, B_ids, C_ids, probs, rules_by_A,
) -> np.ndarray:
    """
    alpha[i, j, A] = P(A =>* x[i..j])   shape (n, n, N)
    Identical vectorisation to inside.py.
    """
    n = len(x)
    N = len(nt_list)
    alpha = np.zeros((n, n, N), dtype=np.float64)
    x_arr = np.asarray(x)

    for t, pt in bcfg.preterminals.items():
        pos = np.where(x_arr == t)[0]
        if pos.size:
            alpha[pos, pos, nt_idx[pt]] = 1.0

    i_base = np.arange(n)
    for l in range(2, n + 1):
        n_spans = n - l + 1
        i_arr = i_base[:n_spans]

        I, D = np.meshgrid(i_arr, np.arange(l - 1), indexing="ij")
        K, J = I + D, I + (l - 1)

        left  = alpha[I, K]       # (n_spans, l-1, N)
        right = alpha[K + 1, J]   # (n_spans, l-1, N)

        left_B  = left[:,  :, B_ids]   # (n_spans, l-1, R)
        right_C = right[:, :, C_ids]   # (n_spans, l-1, R)
        weighted    = left_B * right_C * probs          # broadcast probs over axes 0,1
        sum_splits  = weighted.sum(axis=1)              # (n_spans, R)

        for A_id, rule_idxs in rules_by_A.items():
            alpha[i_arr, i_arr + (l - 1), A_id] += sum_splits[:, rule_idxs].sum(axis=1)

    return alpha


# ── Precompute right-function matrix (once per grammar) ──────────────────────

def _precompute_right_matrix(bcfg: BinarizedCFG, nt_idx: dict, nt_list: list):
    """
    The right() recursion for a fixed start is:

        right(A, start) = IS(A, start) + Σ_{A→BC} p × right(B, start)

    where IS is the inner k-sum (already known from previously computed starts),
    and the same-start dependency A → B (left child) forms a DAG.

    In matrix form: right = (I − W)^{−1} × IS,  where W[A,B] = Σ_{A→BC} p.

    Since the grammar is acyclic (hierarchical levels), (I−W) is non-singular
    (unit lower-triangular in topological order, det = 1).
    """
    
    N = len(nt_list)
    W = np.zeros((N, N), dtype=np.float64)
    for A, B, C, p in bcfg.binary_rules:
        W[nt_idx[A], nt_idx[B]] += p          # left-child weight

    M = np.linalg.inv(np.eye(N) - W)          # (I − W)^{−1}
    return M


# ── Last-column table — batch over all terminal candidates ────────────────────

def _build_last_columns(
    c: int,
    alpha: np.ndarray,
    bcfg: BinarizedCFG,
    nt_idx: dict,
    A_ids, B_ids, C_ids, probs,
    terminals=TERMINALS,
) -> np.ndarray:
    """
    col[v, start, A] = P(A generates exactly y_v[start..c])
    where y_v = x[0..c-1] + [terminals[v]].

    shape: (|V|, c+1, N)

    For spans with end < c the tokens are identical across v, so alpha[start,k,B]
    is shared.  Only the base at position c differs (one preterminal per terminal).
    """
    V = len(terminals)
    N = alpha.shape[2]
    col = np.zeros((V, c + 1, N), dtype=np.float64)

    # Base: col[v, c, PT_{terminals[v]}] = 1.0
    for v, t in enumerate(terminals):
        pt = bcfg.preterminals.get(t)
        if pt is not None:
            col[v, c, nt_idx[pt]] = 1.0

    # Fill spans [start, c] bottom-up (start goes from c-1 down to 0)
    for start in range(c - 1, -1, -1):
        # k ranges from start to c-1
        k_arr = np.arange(start, c)                  # (S,) where S = c-start
        left   = alpha[start, k_arr, :]               # (S, N)  — shared across v
        right_c = col[:, k_arr + 1, :]                # (V, S, N)

        left_B  = left[:, B_ids]                      # (S, R)
        right_C = right_c[:, :, C_ids]                # (V, S, R)

        # weighted[v, k, r] = left_B[k, r] * right_C[v, k, r] * probs[r]
        weighted = left_B[np.newaxis] * right_C * probs   # (V, S, R)
        sum_k    = weighted.sum(axis=1)                    # (V, R)  — sum over k

        # Scatter sum_k[v, r] into col[v, start, A_ids[r]]
        delta = np.zeros((V, N), dtype=np.float64)
        for r, a_id in enumerate(A_ids):
            delta[:, a_id] += sum_k[:, r]
        col[:, start] += delta

    return col   # (V, c+1, N)


# ── Right function — batch over all terminal candidates ───────────────────────

def _compute_right_all(
    c: int,
    alpha: np.ndarray,
    bcfg: BinarizedCFG,
    nt_idx: dict,
    A_ids, B_ids, C_ids, probs,
    M: np.ndarray,
    terminals=TERMINALS,
) -> np.ndarray:
    """
    R[v, start, A] = P(A generates a string starting with y_v[start..c])
    where y_v = x[0..c-1] + [terminals[v]].

    CORRECT formula (handles NTs that generate beyond the prefix boundary):

        right(A, start) = IS(A, start) + Σ_{A→BC} p × right(B, start)

    where IS(A, start) = Σ_{A→BC} p × Σ_{k=start}^{c-1} alpha[start,k,B] × right(C,k+1)

    In matrix form:  right[start] = M @ (IS[start] + base[start])
    where M = (I − W)^{−1}  (precomputed once per grammar).

    Base cases (preterminals):
        right(PT_t, c)     = 1.0 if t == terminals[v], else 0
        right(PT_t, start) = 0   for start < c  (PT generates 1 token; can't start
                                                  a 2+-token prefix)
        right(any A, c+1)  = 1.0 (no remaining prefix; any continuation is valid)

    R shape: (V, c+2, N)
    """
    V = len(terminals)
    N = alpha.shape[2]
    R = np.zeros((V, c + 2, N), dtype=np.float64)
    R[:, c + 1, :] = 1.0    # base: nothing left to constrain

    # Preterminal base cases at start = c
    for t, pt in bcfg.preterminals.items():
        pt_id = nt_idx[pt]
        for v, tv in enumerate(terminals):
            if tv == t:
                R[:, c, pt_id] = 0.0          # reset (was 0)
                R[v, c, pt_id] = 1.0          # only this v matches

    # Non-preterminals at start = c: IS = 0 (no k < c), so right = M @ base
    # Apply M to obtain right[c] from the preterminal base
    base_c = R[:, c, :].copy()                # (V, N)
    R[:, c, :] = base_c @ M.T                 # (V, N) @ (N, N) → (V, N)

    # Fill start from c−1 down to 0
    for start in range(c - 1, -1, -1):
        # IS[start, v, A] = Σ_{A→BC} p × Σ_{k=start}^{c-1} alpha[start,k,B] × R[v,k+1,C]
        k_arr   = np.arange(start, c)            # (S,)
        left    = alpha[start, k_arr, :]          # (S, N)  — shared across v
        right_R = R[:, k_arr + 1, :]             # (V, S, N)

        left_B  = left[:, B_ids]                 # (S, R)
        right_C = right_R[:, :, C_ids]           # (V, S, R)

        weighted = left_B[np.newaxis] * right_C * probs   # (V, S, R)
        sum_k    = weighted.sum(axis=1)                    # (V, R)

        IS = np.zeros((V, N), dtype=np.float64)
        for r, a_id in enumerate(A_ids):
            IS[:, a_id] += sum_k[:, r]

        # right[start] = M @ IS[start]   (preterminals stay 0 for start < c)
        R[:, start, :] = IS @ M.T         # (V, N) @ (N, N) → (V, N)

    return R   # (V, c+2, N)


# ── True next-token distribution for every position in one string ─────────────

def true_dists_for_string(
    x: list,
    alpha: np.ndarray,
    bcfg: BinarizedCFG,
    nt_idx: dict,
    A_ids, B_ids, C_ids, probs,
    M: np.ndarray,
    terminals=TERMINALS,
) -> np.ndarray:
    """
    For every position c in x, compute P_true(next = t | x[0..c-1]) for each terminal t.
    Returns p_true of shape (n, |V|), normalised.

    Per position c: _compute_right_all gives R[v, start, A] via the correct
    formula right = M @ IS, where M = (I−W)^{−1} handles NTs that overshoot
    the prefix boundary.
    """
    n = len(x)
    V = len(terminals)
    root_id = nt_idx[bcfg.root]
    p_true = np.zeros((n, V), dtype=np.float64)

    for c in range(n):
        R = _compute_right_all(
            c, alpha, bcfg, nt_idx, A_ids, B_ids, C_ids, probs, M, terminals
        )
        unnorm = R[:, 0, root_id]    # (V,)
        Z = unnorm.sum()
        p_true[c] = unnorm / Z if Z > 0 else np.ones(V) / V

    return p_true   # (n, V)


# ── GPT predicted distribution ────────────────────────────────────────────────

def gpt_dists_for_string(
    model: GPT2Rotary,
    x: list,
    device: str,
    terminals=TERMINALS,
) -> np.ndarray:
    """
    One GPT forward pass on [BOS, x[0], ..., x[n-1]].
    The logit at output position c predicts x[c] (the token at string position c).
    Returns p_model of shape (n, |V|), renormalised over terminal indices.
    """
    n = len(x)
    token_ids = torch.tensor([[BOS_TOKEN] + list(x)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(token_ids)   # (1, n+1, vocab_size)

    # Position c of the logit output predicts x[c]
    logits_seq = logits[0, :n, :].float().cpu().numpy()    # (n, vocab_size)
    logits_seq -= logits_seq.max(axis=1, keepdims=True)    # numerical stability
    exp_logits = np.exp(logits_seq)

    terminal_ids = list(terminals)
    probs_terminals = exp_logits[:, terminal_ids]          # (n, V)
    Z = probs_terminals.sum(axis=1, keepdims=True)
    p_model = np.where(Z > 0, probs_terminals / Z, 1.0 / len(terminals))

    return p_model   # (n, V), renormalised over terminals


# ── KL for one string ─────────────────────────────────────────────────────────

def kl_per_token(p_true: np.ndarray, p_model: np.ndarray, eps: float = 1e-40) -> np.ndarray:
    """
    KL(p_true[c] || p_model[c]) for each position c.
    Shape: (n,).   Convention: 0 × log(0/q) = 0.
    """
    log_ratio = np.log(np.maximum(p_true, eps)) - np.log(np.maximum(p_model, eps))
    return (p_true * log_ratio).sum(axis=1)   # (n,)


def compute_kl_for_string(
    x: list,
    model: GPT2Rotary,
    bcfg: BinarizedCFG,
    nt_idx: dict,
    nt_list: list,
    A_ids, B_ids, C_ids, probs, rules_by_A,
    M: np.ndarray,
    device: str,
    terminals=TERMINALS,
) -> Tuple[float, int]:
    """
    Returns (sum_of_kl_over_positions, n_positions) for one test string.
    Divide the first by the second to get average per-token KL.
    """
    alpha   = build_inside_table(x, bcfg, nt_idx, nt_list, A_ids, B_ids, C_ids, probs, rules_by_A)
    p_true  = true_dists_for_string(x, alpha, bcfg, nt_idx, A_ids, B_ids, C_ids, probs, M, terminals)
    p_model = gpt_dists_for_string(model, x, device, terminals)

    kl_vals = kl_per_token(p_true, p_model)
    return float(kl_vals.sum()), len(x)


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_kl_evaluation(
    gpt_checkpoint_path: str,
    cfg_path: str,
    n_strings: int = 200,
    max_string_len: int = 400,
    device: str = "cuda",
    random_gpt: bool = False,
) -> float:
    """
    Evaluate Result 3 on n_strings fresh test strings.

    Complexity note: O(n³ × |V|) per string due to the prefix inside recursion.
    For n≈300, |V|=3 this takes ~5–15 s/string in Python with numpy.
    Use n_strings=50–100 for a quick check; 200+ for publication-quality numbers.

    Paper targets (Figure 4):
      cfg3b, GPT_rot: ~0.00008 nats/token
      cfg3f, GPT_rot: ~0.00455 nats/token
    """
    from tqdm import tqdm

    cfg  = load_cfg(cfg_path)
    bcfg = binarize(cfg)

    nt_list = sorted(bcfg.all_nts)
    nt_idx  = {nt: i for i, nt in enumerate(nt_list)}
    A_ids, B_ids, C_ids, probs, rules_by_A = _prep_rules(bcfg, nt_list, nt_idx)
    M = _precompute_right_matrix(bcfg, nt_idx, nt_list)
    print(f"Grammar: {len(nt_list)} NTs, {len(bcfg.binary_rules)} rules after binarization")

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    if not random_gpt:
        state = torch.load(gpt_checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded GPT weights from {gpt_checkpoint_path}")
    else:
        print("GPT_rand: using random weights (expected KL >> 0)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    total_kl        = 0.0
    total_positions = 0
    n_done          = 0

    print(f"\nEvaluating KL on {n_strings} strings  (max_len={max_string_len})...")
    pbar = tqdm(total=n_strings)

    while n_done < n_strings:
        sample = cfg.sample_string()
        x = sample.string
        if len(x) == 0 or len(x) > max_string_len:
            continue

        kl_sum, n_pos = compute_kl_for_string(
            x, model, bcfg, nt_idx, nt_list,
            A_ids, B_ids, C_ids, probs, rules_by_A, M, device,
        )

        total_kl        += kl_sum
        total_positions += n_pos
        n_done          += 1
        pbar.update(1)

        if n_done % 5 == 0:
            pbar.set_postfix({"KL nats/tok": f"{total_kl / total_positions:.5f}"})

    pbar.close()

    avg_kl = total_kl / total_positions
    print(f"\n{'='*50}")
    print(f"  Result 3 — KL(P_true || P_model)")
    print(f"  {avg_kl:.6f} nats/token   ({n_done} strings, {total_positions} positions)")
    print(f"{'='*50}")
    print(f"  Paper target (GPT_rot, cfg3b): ~0.00008")
    print(f"  Paper target (GPT_rot, cfg3f): ~0.00455")

    return avg_kl


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KL divergence evaluation (Result 3)")
    parser.add_argument("--checkpoint", required=True, help="Path to GPT_rot .pt checkpoint")
    parser.add_argument("--cfg",        default="cfg/grammars/cfg3b.txt")
    parser.add_argument("--n_strings",  type=int, default=200)
    parser.add_argument("--max_len",    type=int, default=400)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_gpt", action="store_true")
    args = parser.parse_args()

    run_kl_evaluation(
        gpt_checkpoint_path=args.checkpoint,
        cfg_path=os.path.join(project_root, args.cfg),
        n_strings=args.n_strings,
        max_string_len=args.max_len,
        device=args.device,
        random_gpt=args.random_gpt,
    )
