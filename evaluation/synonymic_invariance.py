"""
Synonymic invariance  S_{k,ℓ}  (Extension A).

Imports the Random Hierarchy Model's central measurable
(documentation/extension_A_synonymic_invariance.md) into our language-model
transformer. For network layer k and swap level ℓ:

        ⟨ ‖ f_k(x) − f_k(P_ℓ x) ‖²  ⟩_{x, P_ℓ}
  S_{k,ℓ} = ────────────────────────────────────────
        ⟨ ‖ f_k(x) − f_k(y)     ‖²  ⟩_{x, y}

  numerator   — how much layer k moves under a swap that *shouldn't* matter
                (a synonym swap: same meaning, different wording).
  denominator — the natural scale: how much it moves between two unrelated inputs.

  S ≈ 1  → layer still "hears the wording" (sensitive).
  S ≈ 0  → layer "hears only the meaning"  (invariant).

Two readouts of f_k (see §5 of the doc):
  - boundary : f_k(x) = h_k at the swapped subtree's right-boundary position j
               (where R5 showed the subtree summary lives). Primary.
  - pooled   : f_k(x) = mean over all token positions. Secondary / sanity.

The model is frozen; NO retraining and NO probe training are needed — this is
just forward passes with hooks on every block. Because numerator and denominator
share the same layer k, any per-layer activation-scale factor cancels in the
ratio, so no explicit normalisation is required.

Run:
    python evaluation/synonymic_invariance.py \
        --checkpoint gpt_checkpoint_step_6500.pt \
        --cfg cfg/grammars/cfg3b.txt \
        --levels 2 3 4 5 6 --n_pairs 1000 --device cuda
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg, CFG
from cfg.synonym_swap import synonym_swap, SwapResult
from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0


# ── All-layer hidden-state extraction (extends probing.py's single-block hook) ────

def _make_store_hook(store: dict, key: int):
    def hook_fn(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        store[key] = out.detach()
    return hook_fn


@torch.no_grad()
def extract_all_layers_at(
    model: GPT2Rotary,
    batch_token_ids: List[List[int]],
    positions: List[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One batched forward pass with a hook on every block. For each sequence b,
    return its hidden state at `positions[b]` and its mean-pooled hidden state,
    for every layer.

    `positions[b]` is an index into the *token* sequence (0 = BOS), so a terminal
    boundary index j in the string maps to position j+1.

    Returns
    -------
    at_pos : (n_layers+1, B, n_embd)   hidden at positions[b]  (layer 0 = embeddings)
    pooled : (n_layers+1, B, n_embd)   mean over real (non-pad) token positions
    """
    lengths = [len(ids) for ids in batch_token_ids]
    max_len = max(lengths)
    B = len(batch_token_ids)

    padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_token_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    store: dict = {}
    handles = []
    # Layer 0 = token embeddings (hook the embedding module), layers 1..L = blocks.
    handles.append(model.wte.register_forward_hook(_make_store_hook(store, 0)))
    for k, blk in enumerate(model.blocks, start=1):
        handles.append(blk.register_forward_hook(_make_store_hook(store, k)))

    model(padded)
    for h in handles:
        h.remove()

    n_keys = len(store)                      # n_layers + 1
    n_embd = store[0].shape[-1]
    at_pos = torch.empty(n_keys, B, n_embd)
    pooled = torch.empty(n_keys, B, n_embd)

    pos_t = torch.tensor(positions, dtype=torch.long, device=device)
    for k in range(n_keys):
        hk = store[k]                        # (B, max_len, n_embd)
        at_pos[k] = hk[torch.arange(B, device=device), pos_t].cpu()
        for b in range(B):
            pooled[k, b] = hk[b, : lengths[b]].mean(dim=0).cpu()
    return at_pos, pooled


# ── Build (x, P_ℓ x) pairs for one level ─────────────────────────────────────────

def build_pairs(
    cfg: CFG,
    level: int,
    n_pairs: int,
    max_attempts: int = 200,
    allow_unmatched: bool = True,
) -> List[SwapResult]:
    """
    Sample `n_pairs` synonym swaps at `level`.

    Leads with length-matched swaps. If `allow_unmatched` and length-matched
    swaps are impossible at this level (cfg3b ℓ=6: every NT6 symbol's two rules
    differ in length), falls back to best-effort swaps that read each string's own
    boundary index. Reports how many of each kind were used.
    """
    out: List[SwapResult] = []
    tries = matched = unmatched = 0
    budget = 50 * n_pairs
    while len(out) < n_pairs and tries < budget:
        tries += 1
        # require_length_match=False still tries up to max_attempts for a match,
        # only falling back to a best-effort swap if none is found.
        res = synonym_swap(cfg, level, max_attempts=max_attempts,
                           require_length_match=not allow_unmatched)
        if res is None:
            continue
        if res.length_matched:
            matched += 1
        else:
            unmatched += 1
        out.append(res)
    note = "" if unmatched == 0 else f"  [{unmatched} best-effort, length-mismatched]"
    print(f"  level {level}: {len(out)} pairs from {tries} attempts "
          f"({matched} length-matched){note}")
    return out


# ── The S_{k,ℓ} computation for one level ────────────────────────────────────────

def _seq(string: List[int]) -> List[int]:
    """[BOS] + terminals, capped to the model context (512 incl. BOS)."""
    return [BOS_TOKEN] + string[:511]


def compute_S_for_level(
    model: GPT2Rotary,
    pairs: List[SwapResult],
    device: str,
    batch_size: int = 32,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Returns {'boundary': S[k], 'pooled': S[k]} — synonymic sensitivity per layer
    (length n_layers+1, index 0 = embeddings).
    """
    # Boundary token position is j+1 (BOS offset); guard against the cap at 511.
    # Each string uses its OWN boundary index (identical when length-matched).
    base_seqs = [_seq(r.base.string) for r in pairs]
    swap_seqs = [_seq(r.swapped.string) for r in pairs]
    base_pos = [min(r.base_boundary_index + 1, 511) for r in pairs]
    swap_pos = [min(r.swapped_boundary_index + 1, 511) for r in pairs]

    base_at, base_pool = _extract_batched(model, base_seqs, base_pos, device, batch_size)
    swap_at, swap_pool = _extract_batched(model, swap_seqs, swap_pos, device, batch_size)

    # Numerator: paired ‖f_k(x) − f_k(P_ℓ x)‖²  averaged over pairs, per layer.
    num_b = ((base_at - swap_at) ** 2).sum(dim=-1).mean(dim=1).numpy()
    num_p = ((base_pool - swap_pool) ** 2).sum(dim=-1).mean(dim=1).numpy()

    # Denominator: ‖f_k(x) − f_k(y)‖² over UNRELATED base strings (x, y) — the
    # natural scale. Pair each x with a shuffled partner y (its own boundary pos).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    # avoid x paired with itself
    for i in range(len(perm)):
        if perm[i] == i:
            perm[i] = perm[(i + 1) % len(perm)]
    den_b = ((base_at - base_at[:, perm]) ** 2).sum(dim=-1).mean(dim=1).numpy()
    den_p = ((base_pool - base_pool[:, perm]) ** 2).sum(dim=-1).mean(dim=1).numpy()

    return {
        "boundary": num_b / den_b,
        "pooled": num_p / den_p,
        "num_boundary": num_b, "den_boundary": den_b,
        "num_pooled": num_p, "den_pooled": den_p,
    }


def _extract_batched(model, seqs, positions, device, batch_size):
    """Chunk extract_all_layers_at over the dataset, concatenating along B."""
    at_chunks, pool_chunks = [], []
    for i in range(0, len(seqs), batch_size):
        s = seqs[i : i + batch_size]
        p = positions[i : i + batch_size]
        at, pool = extract_all_layers_at(model, s, p, device)
        at_chunks.append(at)
        pool_chunks.append(pool)
    return torch.cat(at_chunks, dim=1), torch.cat(pool_chunks, dim=1)


# ── Experiment driver ────────────────────────────────────────────────────────────

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
        print("GPT_rand: RANDOM weights (control). Expect a flat S ≈ 1 with no staircase.")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.n_layer
    results: Dict[int, Dict[str, np.ndarray]] = {}

    for level in levels:
        print(f"\n{'='*60}\n  Swap level ℓ = {level} (NT{level})\n{'='*60}")
        pairs = build_pairs(cfg, level, n_pairs)
        if not pairs:
            print(f"  no eligible pairs at level {level}; skipping.")
            continue
        results[level] = compute_S_for_level(
            model, pairs, device, batch_size=batch_size, seed=seed
        )
        s = results[level]["boundary"]
        print("  S_{k,ℓ} (boundary) by layer k = 0..%d:" % n_layers)
        print("   " + "  ".join(f"{v:.2f}" for v in s))

    _report_and_save(results, levels, n_layers, out_dir, random_gpt)
    return results


def _report_and_save(results, levels, n_layers, out_dir, random_gpt):
    os.makedirs(out_dir, exist_ok=True)
    used = [l for l in levels if l in results]
    if not used:
        print("No results to save.")
        return

    # (n_layers+1) × n_levels matrices for the headline heatmap.
    K = n_layers + 1
    Sb = np.full((K, len(used)), np.nan)
    Sp = np.full((K, len(used)), np.nan)
    for j, l in enumerate(used):
        Sb[:, j] = results[l]["boundary"]
        Sp[:, j] = results[l]["pooled"]

    tag = "random" if random_gpt else "trained"
    npz_path = os.path.join(out_dir, f"synonymic_invariance_{tag}.npz")
    np.savez(npz_path, S_boundary=Sb, S_pooled=Sp,
             layers=np.arange(K), levels=np.array(used))
    print(f"\nSaved raw arrays → {npz_path}")

    # Plain-text table (always works, no matplotlib needed).
    txt_path = os.path.join(out_dir, f"synonymic_invariance_{tag}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Synonymic sensitivity S_{{k,ℓ}}  ({tag} GPT)  — boundary readout\n")
        f.write("rows = layer k (0 = embeddings), cols = swap level ℓ\n\n")
        header = "layer " + "".join(f"  NT{l:>2}" for l in used)
        f.write(header + "\n")
        for k in range(K):
            f.write(f"{k:>5} " + "".join(f"  {Sb[k, j]:.2f}" for j in range(len(used))) + "\n")
        f.write("\n(lower = more invariant. Expect S to drop with depth, later for "
                "smaller ℓ — the k≈ℓ+1 staircase.)\n")
    print(f"Saved table       → {txt_path}")
    print(f"\n{open(txt_path).read()}")

    _maybe_plot(Sb, Sp, used, K, out_dir, tag)


def _maybe_plot(Sb, Sp, used, K, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures (raw .npz/.txt saved).")
        return

    # Heatmap: layer × level.
    fig, ax = plt.subplots(figsize=(1.2 * len(used) + 2, 0.4 * K + 1))
    im = ax.imshow(Sb, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(used)))
    ax.set_xticklabels([f"NT{l}" for l in used])
    ax.set_yticks(range(K))
    ax.set_xlabel("swap level ℓ")
    ax.set_ylabel("transformer layer k  (0 = embeddings)")
    ax.set_title(f"Synonymic sensitivity $S_{{k,\\ell}}$ — boundary ({tag})")
    fig.colorbar(im, ax=ax, label="S (0 = invariant, 1 = sensitive)")
    fig.tight_layout()
    hm = os.path.join(out_dir, f"synonymic_invariance_heatmap_{tag}.png")
    fig.savefig(hm, dpi=150)
    print(f"Saved heatmap     → {hm}")

    # Per-level curves: S vs depth.
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for j, l in enumerate(used):
        ax2.plot(range(K), Sb[:, j], marker="o", label=f"NT{l}")
    ax2.set_xlabel("transformer layer k")
    ax2.set_ylabel("$S_{k,\\ell}$ (boundary)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Synonymic sensitivity vs depth ({tag})")
    ax2.legend(title="swap level")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    cv = os.path.join(out_dir, f"synonymic_invariance_curves_{tag}.png")
    fig2.savefig(cv, dpi=150)
    print(f"Saved curves      → {cv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synonymic invariance S_{k,ℓ} (Extension A)")
    parser.add_argument("--checkpoint", default="gpt_checkpoint_step_6500.pt")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3b.txt")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n_pairs", type=int, default=1000,
                        help="length-matched (x, P_ℓ x) pairs per level")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_gpt", action="store_true",
                        help="random-weight control (expect flat S≈1, no staircase)")
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
