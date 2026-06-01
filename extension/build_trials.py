"""
Step B of the activation-patching extension: freeze N clean/donor trials.

Reads the corpus from Step A and, by pure indexing (no CFG generation, no model),
produces N trials. Each trial is one `x_clean` paired with one boundary-matched
donor and the exact positions to patch. The experiment (later piece) loads this
file and runs all three conditions — clean / donor-patch / noise-patch — off the
*same* frozen trials, so the conditions differ only in what gets injected, never
in which string or which positions.

Strategy A (single coherent donor): for each clean we find donors sharing
>= MIN_SHARED level-ℓ boundary positions at the same absolute prefix index, each
with a different NT-ℓ, and pick one at random. The corpus measurement showed
100% per-clean feasibility, so a single coherent donor per clean is easily found.

Self-contained output: each trial stores the actual prefix token lists (with BOS)
and the patch positions already converted to *sequence* indices, so the
experiment needs nothing but this file.

Position convention (the one offset that matters)
-------------------------------------------------
`boundaries[ℓ][i]` indexes terminal position i. The token stream fed to the model
is [BOS] + string, so terminal i lives at sequence index i+1. We compute
`positions_seq = positions_terminal + 1` here, once, and store that — the patch
hook consumes sequence indices directly.
"""

import os
import sys
import argparse

import numpy as np
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.append(project_root)
sys.path.append(this_dir)  # so `build_corpus` (sibling module) is importable

from build_corpus import (
    PREFIX_LEN, TARGET_LEVEL, MIN_SHARED, CACHE_DIR,
    prefix_boundary_mask, prefix_symbols, qualifying_positions,
)

BOS_TOKEN = 0


def build_trials(corpus_path: str, N: int, out_path: str,
                 level: int = TARGET_LEVEL, prefix_len: int = PREFIX_LEN,
                 min_shared: int = MIN_SHARED, seed: int = 0):
    payload = torch.load(corpus_path, weights_only=False)
    samples = payload["samples"]
    M = len(samples)

    # Stack masks/symbols once for vectorized donor search (same as Step A).
    B = np.stack([prefix_boundary_mask(s, level, prefix_len) for s in samples])  # (M, W)
    S = np.stack([prefix_symbols(s, level, prefix_len) for s in samples])        # (M, W)

    rng = np.random.default_rng(seed)
    clean_order = rng.permutation(M)  # distinct clean per trial, random order

    trials = []
    for ci in clean_order:
        if len(trials) >= N:
            break

        # Qualifying-position count of this clean against every donor at once.
        joint = B[ci] & B & (S[ci] != S)   # (M, W)
        counts = joint.sum(axis=1)         # (M,)
        counts[ci] = -1                    # exclude self
        eligible = np.nonzero(counts >= min_shared)[0]
        if eligible.size == 0:
            continue                       # no donor for this clean; skip it

        di = int(rng.choice(eligible))

        clean, donor = samples[ci], samples[di]
        pos_term = [int(p) for p in qualifying_positions(clean, donor, level, prefix_len)]
        pos_seq = [p + 1 for p in pos_term]  # BOS offset, computed once here

        clean_prefix = [BOS_TOKEN] + clean["string"][:prefix_len]
        donor_prefix = [BOS_TOKEN] + donor["string"][:prefix_len]

        nt_clean = [int(clean["symbols"][level][p]) for p in pos_term]
        nt_donor = [int(donor["symbols"][level][p]) for p in pos_term]

        trials.append({
            "clean_idx": int(ci),
            "donor_idx": di,
            "clean_prefix": clean_prefix,     # [BOS] + x_clean[:prefix_len]
            "donor_prefix": donor_prefix,     # [BOS] + x_donor[:prefix_len]
            "positions_terminal": pos_term,   # 0-based terminal indices
            "positions_seq": pos_seq,         # sequence indices (BOS-offset applied)
            "nt_clean": nt_clean,             # NT-ℓ identity at each patched position (clean)
            "nt_donor": nt_donor,             # NT-ℓ identity at each patched position (donor)
            "n_patched": len(pos_term),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = {
        "meta": {
            "corpus_path": corpus_path,
            "level": level,
            "prefix_len": prefix_len,
            "min_shared": min_shared,
            "seed": seed,
            "N_requested": N,
            "N_built": len(trials),
        },
        "trials": trials,
    }
    torch.save(out, out_path)

    # ── Summary so the trials are auditable without reading the code ──
    n_patched = np.array([t["n_patched"] for t in trials])
    n_distinct_donors = len({t["donor_idx"] for t in trials})
    print(f"Built {len(trials)}/{N} trials -> {out_path}")
    print(f"  patched positions / trial : mean {n_patched.mean():.2f}, "
          f"min {n_patched.min()}, max {n_patched.max()}")
    print(f"  distinct donors used      : {n_distinct_donors}")
    print(f"  distinct cleans used      : {len({t['clean_idx'] for t in trials})}")

    # Show one full trial so the clean/donor/positions/NT-swap are inspectable.
    t = trials[0]
    print("\n--- example trial[0] ---")
    print(f"  clean prefix : {t['clean_prefix']}")
    print(f"  donor prefix : {t['donor_prefix']}")
    print(f"  patch terminal positions : {t['positions_terminal']}")
    print(f"  patch sequence positions : {t['positions_seq']}  (BOS-offset)")
    print(f"  NT{level} clean -> donor   : {t['nt_clean']} -> {t['nt_donor']}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze clean/donor trials (Step B)")
    parser.add_argument("--corpus", default=os.path.join(CACHE_DIR, "corpus_cfg3b.pt"))
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--level", type=int, default=TARGET_LEVEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(CACHE_DIR, f"trials_cfg3b_nt{args.level}.pt")
    build_trials(args.corpus, args.N, out_path, level=args.level, seed=args.seed)
