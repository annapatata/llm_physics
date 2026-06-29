"""
Synonymic invariance S_{k,l} -- BLOCK DELTA variant (Extension A).

Identical to synonymic_invariance.py but measures S on the block DELTA
(what each transformer block adds to the residual stream) rather than
the full accumulated residual stream:

    delta[k] = stream[k] - stream[k-1]   for k >= 1
    delta[0] = stream[0]                  (embedding, no predecessor)

This isolates each block's independent contribution, analogous to how
the original RHM paper measures stability on an FCN (no residuals),
where each layer's output IS the delta.

Run:
    python evaluation/synonymic_invariance_delta.py \
        --checkpoint gpt_checkpoint_step_6500.pt \
        --cfg cfg/grammars/cfg3b.txt \
        --levels 2 3 4 5 6 --n_pairs 1000 --device cpu
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cfg.grammar import load_cfg, CFG
from cfg.synonym_swap import synonym_swap, SwapResult
from models.gpt_rot import GPT2Rotary

sys.path.insert(0, os.path.join(project_root, "evaluation"))
from synonymic_invariance import (
    build_pairs,
    extract_all_layers_at,
    _seq,
    _extract_batched,
)

BOS_TOKEN = 0


def compute_S_delta_for_level(
    model: GPT2Rotary,
    pairs: List[SwapResult],
    device: str,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Same as compute_S_for_level but operates on block deltas instead of
    the residual stream. delta[k] = stream[k] - stream[k-1].
    """
    base_seqs = [_seq(r.base.string) for r in pairs]
    swap_seqs = [_seq(r.swapped.string) for r in pairs]
    base_pos  = [min(r.base_boundary_index + 1, 511) for r in pairs]
    swap_pos  = [min(r.swapped_boundary_index + 1, 511) for r in pairs]

    base_at, base_pool = _extract_batched(model, base_seqs, base_pos, device, batch_size)
    swap_at, swap_pool = _extract_batched(model, swap_seqs, swap_pos, device, batch_size)

    # Block delta: stream[k] - stream[k-1]; embedding kept as-is for k=0.
    base_db = torch.cat([base_at[:1],   base_at[1:]   - base_at[:-1]],   dim=0)
    swap_db = torch.cat([swap_at[:1],   swap_at[1:]   - swap_at[:-1]],   dim=0)
    base_dp = torch.cat([base_pool[:1], base_pool[1:] - base_pool[:-1]], dim=0)
    swap_dp = torch.cat([swap_pool[:1], swap_pool[1:] - swap_pool[:-1]], dim=0)

    num_b = ((base_db - swap_db) ** 2).sum(dim=-1).mean(dim=1).numpy()
    num_p = ((base_dp - swap_dp) ** 2).sum(dim=-1).mean(dim=1).numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    for i in range(len(perm)):
        if perm[i] == i:
            perm[i] = perm[(i + 1) % len(perm)]

    den_b = ((base_db - base_db[:, perm]) ** 2).sum(dim=-1).mean(dim=1).numpy()
    den_p = ((base_dp - base_dp[:, perm]) ** 2).sum(dim=-1).mean(dim=1).numpy()

    return {
        "boundary": num_b / den_b,
        "pooled":   num_p / den_p,
    }


def run(
    checkpoint_path: str,
    cfg_path: str,
    levels: Tuple[int, ...] = (2, 3, 4, 5, 6),
    n_pairs: int = 1000,
    device: str = "cuda",
    batch_size: int = 32,
    random_gpt: bool = False,
    out_dir: str = "results",
    seed: int = 0,
) -> Dict[int, Dict[str, np.ndarray]]:
    cfg = load_cfg(cfg_path)

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    if not random_gpt:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Loaded GPT weights from {checkpoint_path}")
    else:
        print("GPT_rand: RANDOM weights (control).")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.n_layer
    results: Dict[int, Dict[str, np.ndarray]] = {}

    for level in levels:
        print(f"\n{'='*60}\n  Swap level l = {level} (NT{level})\n{'='*60}")
        pairs = build_pairs(cfg, level, n_pairs)
        if not pairs:
            print(f"  no eligible pairs at level {level}; skipping.")
            continue
        results[level] = compute_S_delta_for_level(
            model, pairs, device, batch_size=batch_size, seed=seed
        )
        s = results[level]["boundary"]
        print("  S_delta_{k,l} (boundary) by layer k = 0..%d:" % n_layers)
        print("   " + "  ".join(f"{v:.2f}" for v in s))

    _report_and_save(results, levels, n_layers, out_dir, random_gpt)
    return results


def _report_and_save(results, levels, n_layers, out_dir, random_gpt):
    os.makedirs(out_dir, exist_ok=True)
    used = [l for l in levels if l in results]
    if not used:
        print("No results to save.")
        return

    K = n_layers + 1
    Sb = np.full((K, len(used)), np.nan)
    Sp = np.full((K, len(used)), np.nan)
    for j, l in enumerate(used):
        Sb[:, j] = results[l]["boundary"]
        Sp[:, j] = results[l]["pooled"]

    tag = "random" if random_gpt else "trained"
    npz_path = os.path.join(out_dir, f"synonymic_invariance_delta_{tag}.npz")
    np.savez(npz_path, S_boundary=Sb, S_pooled=Sp,
             layers=np.arange(K), levels=np.array(used))
    print(f"\nSaved raw arrays -> {npz_path}")

    txt_path = os.path.join(out_dir, f"synonymic_invariance_delta_{tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Synonymic sensitivity S_{{k,l}} -- block delta ({tag} GPT) -- boundary\n")
        f.write("delta[k] = stream[k] - stream[k-1]  (isolates each block's contribution)\n")
        f.write("rows = layer k (0 = embeddings), cols = swap level l\n\n")
        header = "layer " + "".join(f"  NT{l:>2}" for l in used)
        f.write(header + "\n")
        for k in range(K):
            f.write(f"{k:>5} " + "".join(f"  {Sb[k, j]:.2f}" for j in range(len(used))) + "\n")
        f.write("\n(lower = more invariant.)\n")
    print(f"Saved table    -> {txt_path}")
    print(f"\n{open(txt_path, encoding='utf-8').read()}")

    _maybe_plot(Sb, Sp, used, K, out_dir, tag)


def _maybe_plot(Sb, Sp, used, K, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping figures.")
        return

    w = 1.2 * len(used) + 2
    h = 0.4 * K + 1

    # Heatmap
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(Sb, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(used)))
    ax.set_xticklabels([f"NT{l}" for l in used])
    ax.set_yticks(range(K))
    ax.set_xlabel("swap level l")
    ax.set_ylabel("transformer layer k  (0 = embeddings)")
    ax.set_title(f"S_{{k,l}} block delta -- boundary ({tag})")
    fig.colorbar(im, ax=ax, label="S (0 = invariant, 1 = sensitive)")
    fig.tight_layout()
    hm = os.path.join(out_dir, f"synonymic_invariance_delta_heatmap_{tag}.png")
    fig.savefig(hm, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap  -> {hm}")

    # Curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, l in enumerate(used):
        ax.plot(range(K), Sb[:, j], marker="o", label=f"NT{l}")
    ax.set_xlabel("transformer layer k")
    ax.set_ylabel("$S_{k,l}$ delta (boundary)")
    ax.set_title(f"Synonymic sensitivity -- block delta ({tag})")
    ax.legend(title="swap level")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    cv = os.path.join(out_dir, f"synonymic_invariance_delta_curves_{tag}.png")
    fig.savefig(cv, dpi=150)
    plt.close(fig)
    print(f"Saved curves   -> {cv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synonymic invariance -- block delta variant")
    parser.add_argument("--checkpoint", default="gpt_checkpoint_step_6500.pt")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3b.txt")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n_pairs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_gpt", action="store_true")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        cfg_path=os.path.join(project_root, args.cfg),
        levels=tuple(args.levels),
        n_pairs=args.n_pairs,
        device=args.device,
        batch_size=args.batch_size,
        random_gpt=args.random_gpt,
        out_dir=args.out_dir,
        seed=args.seed,
    )
