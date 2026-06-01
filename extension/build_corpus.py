"""
Step A of the activation-patching extension: build a reusable corpus of valid
CFG strings *once*, so the experiment never has to sample/search in its hot loop.

What this produces
------------------
A file `extension/cache/corpus_<grammar>.pt` holding M valid strings, each with
its full parse annotations (boundaries + ancestor NT symbols for levels 2..6).
`build_trials` (Step B) then matches clean/donor pairs purely by indexing into
this corpus — no CFG generation, no model — and the experiment just loads the
frozen pairs.

Why annotations for *all* levels 2..6 (not just NT5)
----------------------------------------------------
The corpus is reusable. The NT5 headline, plus the optional NT6 / NT3 *contrast*
panels (EXTENSION.md §5), all read from the same file. Storing every level costs
almost nothing and saves re-sampling later.

Why this also measures the acceptance rate
------------------------------------------
The open question for M is: for a given `x_clean`, does at least one donor in the
corpus pass the joint filter (≥ MIN_SHARED level-ℓ boundaries shared at the same
absolute prefix index, each with a *different* NT-ℓ)? Matching is pairwise, so M
strings give ~M² candidate pairs — but the per-pair probability `p` is grammar-
specific and worth knowing before committing. `measure_acceptance` reports it
empirically so the M=2000 / N=500 choice rests on a real number, not a guess.

Note: strings from `cfg.sample_string()` are valid by construction, so no CYK
check is needed when *building* the corpus (only later, on generated completions).
"""

import os
import sys
import random
import argparse

import numpy as np
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.append(project_root)

from cfg.grammar import load_cfg

# Design constants (match EXTENSION.md).
PREFIX_LEN = 50      # clean prefix = [BOS] + x_clean[:50]; matching is within this window
TARGET_LEVEL = 5     # NT5 headline
MIN_SHARED = 3       # accept a donor with >= 3 qualifying boundary positions
STORE_LEVELS = (2, 3, 4, 5, 6)  # keep all NT levels so NT6/NT3 contrasts are free

CACHE_DIR = os.path.join(this_dir, "cache")  # extension/cache (next to these scripts)
MODEL_MAX_SEQ_LEN = 512  # [BOS] + string must fit the context window


# ── Build the corpus ────────────────────────────────────────────────────────────

def build_corpus(cfg_path: str, M: int, out_path: str, seed: int = 0):
    """
    Sample M valid strings and store the annotations the matcher/experiment need.

    Each stored sample is a plain dict (cheap to torch.save/load):
        string      : List[int]              terminal tokens (no BOS)
        length      : int
        boundaries  : {level: List[int]}      b_ℓ(i) over terminal positions
        symbols     : {level: List[int]}      ancestor NT symbol s_ℓ(i)
    """
    random.seed(seed)
    cfg = load_cfg(cfg_path)

    samples = []
    attempts = 0
    while len(samples) < M:
        attempts += 1
        s = cfg.sample_string()
        # Must fit the context window and contain the target level.
        if s.length > MODEL_MAX_SEQ_LEN - 2:
            continue
        if TARGET_LEVEL not in s.ancestor_symbols:
            continue
        samples.append({
            "string": s.string,
            "length": s.length,
            "boundaries": {lv: s.boundaries[lv] for lv in STORE_LEVELS if lv in s.boundaries},
            "symbols": {lv: s.ancestor_symbols[lv] for lv in STORE_LEVELS if lv in s.ancestor_symbols},
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "meta": {
            "cfg_path": cfg_path,
            "M": M,
            "seed": seed,
            "prefix_len": PREFIX_LEN,
            "store_levels": STORE_LEVELS,
            "attempts": attempts,
        },
        "samples": samples,
    }
    torch.save(payload, out_path)
    print(f"Saved {len(samples)} samples to {out_path} "
          f"({attempts} draws, {len(samples)/attempts:.1%} kept).")
    return payload


# ── Matching primitives (shared with Step B later) ──────────────────────────────

def prefix_boundary_mask(sample: dict, level: int, prefix_len: int) -> np.ndarray:
    """Bool array (prefix_len,): True where position i is a level-ℓ boundary."""
    b = sample["boundaries"].get(level, [])
    out = np.zeros(prefix_len, dtype=bool)
    n = min(prefix_len, len(b))
    if n:
        out[:n] = np.asarray(b[:n], dtype=bool)
    return out


def prefix_symbols(sample: dict, level: int, prefix_len: int) -> np.ndarray:
    """Int array (prefix_len,): NT-ℓ symbol per position, -1 past the string end."""
    sy = sample["symbols"].get(level, [])
    out = np.full(prefix_len, -1, dtype=np.int64)
    n = min(prefix_len, len(sy))
    if n:
        out[:n] = np.asarray(sy[:n], dtype=np.int64)
    return out


def qualifying_positions(clean: dict, donor: dict, level: int, prefix_len: int) -> list:
    """
    The joint filter, evaluated across all prefix positions at once:
    position i qualifies iff it is a level-ℓ boundary in BOTH strings AND the
    NT-ℓ identity differs. Returns the list of qualifying terminal indices.
    """
    cb = prefix_boundary_mask(clean, level, prefix_len)
    db = prefix_boundary_mask(donor, level, prefix_len)
    cs = prefix_symbols(clean, level, prefix_len)
    ds = prefix_symbols(donor, level, prefix_len)
    mask = cb & db & (cs != ds)
    return list(np.nonzero(mask)[0])


# ── Acceptance measurement (answers the "is M enough" question) ─────────────────

def measure_acceptance(payload: dict, level: int = TARGET_LEVEL,
                       prefix_len: int = PREFIX_LEN, min_shared: int = MIN_SHARED,
                       n_clean: int = 500, seed: int = 0):
    """
    Empirically estimate, on the built corpus:
      - mean # of level-ℓ boundaries per prefix,
      - pairwise acceptance p = P(a random (clean, donor) pair has >= min_shared
        qualifying positions),
      - per-clean feasibility = fraction of cleans for which >= 1 donor qualifies
        (this is what actually decides whether we can fill N trials).
    """
    samples = payload["samples"]
    M = len(samples)

    # Pre-stack boundary masks and symbol arrays for fully vectorized matching.
    B = np.stack([prefix_boundary_mask(s, level, prefix_len) for s in samples])  # (M, W)
    S = np.stack([prefix_symbols(s, level, prefix_len) for s in samples])        # (M, W)

    mean_boundaries = B.sum(axis=1).mean()

    rng = np.random.default_rng(seed)
    clean_idx = rng.choice(M, size=min(n_clean, M), replace=False)

    pair_accepts = 0
    pair_total = 0
    feasible = 0
    qual_counts_all = []

    for i in clean_idx:
        # Count qualifying positions of clean i against every donor at once.
        # qualifying = clean_boundary & donor_boundary & (clean_sym != donor_sym)
        joint = B[i] & B & (S[i] != S)          # (M, W) bool
        counts = joint.sum(axis=1)              # (M,) qualifying positions per donor
        counts[i] = -1                          # exclude self
        accepts = counts >= min_shared          # (M,) bool

        pair_accepts += int(accepts.sum())
        pair_total += (M - 1)
        feasible += int(accepts.any())
        qual_counts_all.append(counts[counts >= 0])

    p = pair_accepts / pair_total if pair_total else 0.0
    feasibility = feasible / len(clean_idx) if len(clean_idx) else 0.0
    mean_qual = np.concatenate(qual_counts_all).mean() if qual_counts_all else 0.0

    print("\n--- Acceptance measurement -----------------------------")
    print(f"  level                 : NT{level}   (prefix window = {prefix_len})")
    print(f"  corpus size M         : {M}")
    print(f"  mean level-{level} boundaries / prefix : {mean_boundaries:.2f}")
    print(f"  mean qualifying positions / pair    : {mean_qual:.2f}")
    print(f"  pairwise acceptance p (>= {min_shared} shared) : {p:.4f}")
    print(f"  per-clean feasibility (>=1 donor)   : {feasibility:.1%} "
          f"of {len(clean_idx)} sampled cleans")
    # Rough projection: with ~M donor candidates, P(a clean finds >=1 donor).
    if 0 < p < 1:
        proj = 1.0 - (1.0 - p) ** (M - 1)
        print(f"  projected P(find donor | M={M}) ~ {proj:.3f}  "
              f"-> expect ~{int(proj * M)} usable cleans (need 500)")
    print("--------------------------------------------------------")


# ── Quick inspection of what got cached ─────────────────────────────────────────

def show_examples(payload: dict, level: int = TARGET_LEVEL,
                  prefix_len: int = PREFIX_LEN, n: int = 2):
    samples = payload["samples"]
    print(f"\n--- {n} example prefixes (level NT{level}) ---")
    for s in samples[:n]:
        toks = s["string"][:prefix_len]
        bmask = prefix_boundary_mask(s, level, prefix_len)
        syms = prefix_symbols(s, level, prefix_len)
        bpos = list(np.nonzero(bmask)[0])
        print(f"  len={s['length']}  prefix tokens: {toks}")
        print(f"    NT{level} boundary positions: {bpos}")
        print(f"    NT{level} symbol at each boundary: "
              f"{[int(syms[i]) for i in bpos]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build + measure the patching corpus (Step A)")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3b.txt")
    parser.add_argument("--M", type=int, default=2000, help="corpus size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output .pt path")
    args = parser.parse_args()

    cfg_path = os.path.join(project_root, args.cfg)
    grammar_name = os.path.splitext(os.path.basename(args.cfg))[0]
    out_path = args.out or os.path.join(CACHE_DIR, f"corpus_{grammar_name}.pt")

    payload = build_corpus(cfg_path, args.M, out_path, seed=args.seed)
    show_examples(payload)
    measure_acceptance(payload)
