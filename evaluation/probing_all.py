"""
Per-layer × per-level diagonal probe scan.

The existing Result 5 probe in `probing.py` only reads `model.blocks[-1]` (the
final layer). This module extends that to scan all 12 blocks for each level
ℓ ∈ {2,3,4,5,6}, producing a 12×5 grid of diagonal-probe accuracies.

Why we need this for the activation-patching extension (see EXTENSION.md):
  (a) Pick ℓ* — the layer at which NT-boundary representations peak — as the
      target layer for the patch.
  (b) Check whether any level besides NT6 has a clear representation at some
      mid-layer that the last-layer-only probe missed. If yes, the patching
      experiment can test the dissociation at multiple levels; if no, NT6-only
      focus is well justified.

Design notes:
  - One shared GPT forward pass per training step caches the output of every
    block via 12 forward hooks. All 60 probes then train against that cached
    activation, so total cost is ~n_iters GPT forward passes (not n_iters × 60).
  - Probes use reduced training (~5K iters vs. probing.py's 30K). We need the
    *shape* of the per-layer curve to choose ℓ*, not publication-quality probes.
  - Reuses MultiHeadLinearProbe and build_label_mapping from probing.py to keep
    the protocol identical wherever the two scripts overlap.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.append(project_root)
sys.path.append(this_dir)  # so `probing` (sibling module) is importable

from cfg.grammar import load_cfg, CFG
from models.gpt_rot import GPT2Rotary
from probing import (
    MultiHeadLinearProbe,
    build_label_mapping,
    BOS_TOKEN,
)


# ── Multi-layer hidden state extraction ────────────────────────────────────────

def _make_layer_hook(store: dict, layer_idx: int):
    """Hook factory — caches that block's output into store[layer_idx]."""
    def hook_fn(module, input, output):
        # GPTBlock returns plain tensor (no attention) or (tensor, weights).
        out = output[0] if isinstance(output, tuple) else output
        store[layer_idx] = out.detach()
    return hook_fn


def extract_all_layer_states(
    model: GPT2Rotary,
    batch_token_ids: List[List[int]],
    device: str,
) -> Dict[int, torch.Tensor]:
    """
    Single forward pass; cache the output of every block. Returns
        {layer_idx: (B, max_len, n_embd) tensor on `device`}

    Hidden states stay on the GPU to avoid a CPU↔GPU round-trip every step.
    """
    lengths = [len(ids) for ids in batch_token_ids]
    max_len = max(lengths)

    padded = torch.zeros(len(batch_token_ids), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_token_ids):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    store: Dict[int, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(_make_layer_hook(store, idx))
        for idx, block in enumerate(model.blocks)
    ]

    try:
        with torch.no_grad():
            model(padded)
    finally:
        for h in handles:
            h.remove()

    return store


# ── Batch collection for all (layer, level) probes ─────────────────────────────

def _collect_multilayer_batch(
    cfg: CFG,
    model: GPT2Rotary,
    levels: List[int],
    label_maps: Dict[int, Dict[int, int]],
    batch_size: int,
    device: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Sample fresh strings, run one shared forward pass through the frozen GPT,
    return:
      hidden_per_layer : {layer_idx: (B, T, n_embd) sliced to symbol positions}
      labels_per_level : {level:    (B, T) padded with -100 for cross_entropy}

    Requires that every sample has ALL requested levels present in its parse
    tree (rare to skip for these CFGs, but the check matches probing.py).
    """
    samples, token_id_lists = [], []
    while len(samples) < batch_size:
        s = cfg.sample_string()
        if not all(L in s.ancestor_symbols for L in levels):
            continue
        toks = [BOS_TOKEN] + s.string[:511]
        samples.append(s)
        token_id_lists.append(toks)

    raw_hidden = extract_all_layer_states(model, token_id_lists, device)

    max_sym_len = max(min(s.length, 511) for s in samples)
    B = batch_size
    n_embd = raw_hidden[0].shape[-1]

    # Slice each layer's tensor to symbol positions (skip BOS at pos 0)
    hidden_per_layer: Dict[int, torch.Tensor] = {}
    for layer_idx, tensor in raw_hidden.items():
        sliced = torch.zeros(B, max_sym_len, n_embd, device=device)
        for i, s in enumerate(samples):
            sym_len = min(s.length, 511)
            sliced[i, :sym_len] = tensor[i, 1:sym_len + 1]
        hidden_per_layer[layer_idx] = sliced

    labels_per_level: Dict[int, torch.Tensor] = {}
    for L in levels:
        label_tensor = torch.full((B, max_sym_len), -100, dtype=torch.long, device=device)
        for i, s in enumerate(samples):
            sym_len = min(s.length, 511)
            labels = [label_maps[L][sym] for sym in s.ancestor_symbols[L][:sym_len]]
            label_tensor[i, :sym_len] = torch.tensor(labels, dtype=torch.long, device=device)
        labels_per_level[L] = label_tensor

    return hidden_per_layer, labels_per_level


# ── Train all 60 diagonal probes against a shared forward pass ─────────────────

def train_all_probes(
    probes: Dict[Tuple[int, int], MultiHeadLinearProbe],
    optimizers: Dict[Tuple[int, int], torch.optim.Optimizer],
    gpt_model: GPT2Rotary,
    cfg: CFG,
    levels: List[int],
    label_maps: Dict[int, Dict[int, int]],
    n_iters: int,
    batch_size: int,
    device: str,
) -> Dict[Tuple[int, int], float]:
    """
    Each step:
      1. Sample a batch of fresh strings.
      2. One GPT forward pass with 12 hooks → cached hidden states per layer.
      3. For every (layer, level) probe, take one diagonal-mode training step
         on the cached states at that layer + labels at that level.

    Returns final EMA loss per (layer, level).
    """
    gpt_model.eval()
    for p in gpt_model.parameters():
        p.requires_grad_(False)

    ema_loss: Dict[Tuple[int, int], Optional[float]] = {key: None for key in probes}

    progress = tqdm(range(1, n_iters + 1), desc=f"All-probes ({n_iters} iters)")
    for step in progress:
        hidden_per_layer, labels_per_level = _collect_multilayer_batch(
            cfg, gpt_model, levels, label_maps, batch_size, device
        )

        T = labels_per_level[levels[0]].shape[1]
        diag_mask = MultiHeadLinearProbe.make_diagonal_mask(T, delta=0, device=device)

        for (layer_idx, level), probe in probes.items():
            opt = optimizers[(layer_idx, level)]
            opt.zero_grad()

            logits = probe(hidden_per_layer[layer_idx], attn_mask=diag_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, probe.n_classes),
                labels_per_level[level].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            opt.step()

            v = loss.item()
            prev = ema_loss[(layer_idx, level)]
            ema_loss[(layer_idx, level)] = v if prev is None else 0.95 * prev + 0.05 * v

        if step % 500 == 0:
            # Show one representative loss to keep the progress bar readable.
            last_layer = max(layer_idx for layer_idx, _ in probes)
            top_level = levels[-1]
            ref = ema_loss[(last_layer, top_level)]
            progress.set_postfix({f"L{last_layer}_NT{top_level}": f"{ref:.3f}"})

    return ema_loss


# ── Evaluation ─────────────────────────────────────────────────────────────────

@dataclass
class LayerLevelResult:
    layer: int
    level: int
    boundary_accuracy: float
    n_boundary_positions: int


def evaluate_all_probes(
    probes: Dict[Tuple[int, int], MultiHeadLinearProbe],
    gpt_model: GPT2Rotary,
    cfg: CFG,
    levels: List[int],
    label_maps: Dict[int, Dict[int, int]],
    n_eval: int,
    device: str,
) -> Dict[Tuple[int, int], LayerLevelResult]:
    """One forward pass per eval sample, then loop probes over the cached layers."""
    for probe in probes.values():
        probe.eval()
    gpt_model.eval()

    n_layers = max(layer_idx for layer_idx, _ in probes) + 1
    correct = {key: 0 for key in probes}
    total = {key: 0 for key in probes}

    with torch.no_grad():
        for _ in tqdm(range(n_eval), desc="Eval", leave=False):
            sample = cfg.sample_string()
            if not all(L in sample.ancestor_symbols for L in levels):
                continue

            sym_len = min(sample.length, 511)
            token_ids = [BOS_TOKEN] + sample.string[:511]
            raw_hidden = extract_all_layer_states(gpt_model, [token_ids], device)

            # Slice each layer to symbol positions, batched as (1, T, n_embd).
            T = sym_len
            sliced = {idx: raw_hidden[idx][:, 1:T + 1] for idx in raw_hidden}

            diag_mask = MultiHeadLinearProbe.make_diagonal_mask(T, delta=0, device=device)

            for L in levels:
                labels = torch.tensor(
                    [label_maps[L][sym] for sym in sample.ancestor_symbols[L][:T]],
                    dtype=torch.long, device=device,
                )
                is_boundary = torch.tensor(
                    sample.boundaries[L][:T], dtype=torch.bool, device=device
                )
                if not is_boundary.any():
                    continue

                for layer_idx in range(n_layers):
                    probe = probes[(layer_idx, L)]
                    logits = probe(sliced[layer_idx], attn_mask=diag_mask).squeeze(0)
                    preds = logits.argmax(dim=-1)
                    correct[(layer_idx, L)] += (preds[is_boundary] == labels[is_boundary]).sum().item()
                    total[(layer_idx, L)] += is_boundary.sum().item()

    return {
        key: LayerLevelResult(
            layer=key[0],
            level=key[1],
            boundary_accuracy=correct[key] / total[key] if total[key] > 0 else 0.0,
            n_boundary_positions=total[key],
        )
        for key in probes
    }


# ── Main experiment ────────────────────────────────────────────────────────────

def run_per_layer_scan(
    gpt_checkpoint_path: str,
    cfg_path: str,
    levels: Tuple[int, ...] = (2, 3, 4, 5, 6),
    n_probe_iters: int = 5_000,
    n_eval_samples: int = 500,
    batch_size: int = 60,
    d_pos: int = 1024,
    device: str = 'cuda',
    output_json: Optional[str] = None,
) -> Dict[Tuple[int, int], LayerLevelResult]:
    cfg = load_cfg(cfg_path)

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    state = torch.load(gpt_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"Loaded GPT weights from {gpt_checkpoint_path}")

    label_maps = {L: build_label_mapping(cfg, L) for L in levels}
    for L in levels:
        print(f"  Level {L} label map: {label_maps[L]}")

    n_layers = len(model.blocks)
    probes: Dict[Tuple[int, int], MultiHeadLinearProbe] = {}
    optimizers: Dict[Tuple[int, int], torch.optim.Optimizer] = {}
    for layer_idx in range(n_layers):
        for L in levels:
            probe = MultiHeadLinearProbe(
                n_embd=768, n_heads=16, d_pos=d_pos, n_classes=len(label_maps[L])
            ).to(device)
            probes[(layer_idx, L)] = probe
            optimizers[(layer_idx, L)] = torch.optim.AdamW(
                probe.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=1e-3
            )

    print(
        f"\nTraining {len(probes)} probes "
        f"({n_layers} layers × {len(levels)} levels) for {n_probe_iters} iters."
    )
    train_all_probes(
        probes, optimizers, model, cfg, list(levels), label_maps,
        n_iters=n_probe_iters, batch_size=batch_size, device=device,
    )

    print(f"\nEvaluating on {n_eval_samples} fresh strings...")
    results = evaluate_all_probes(
        probes, model, cfg, list(levels), label_maps,
        n_eval=n_eval_samples, device=device,
    )

    # ── Per-layer × per-level accuracy table ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  Diagonal probe accuracy @ boundary positions (per layer × per level)")
    print("=" * 72)
    header = "  Layer  " + "  ".join(f"  NT{L}" for L in levels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for layer_idx in range(n_layers):
        row = f"  {layer_idx:>5}  "
        for L in levels:
            acc = results[(layer_idx, L)].boundary_accuracy
            row += f"  {acc:>5.1%}"
        print(row)
    print("=" * 72)

    print("\n  Peak layer per level:")
    for L in levels:
        layer_accs = [
            (layer_idx, results[(layer_idx, L)].boundary_accuracy)
            for layer_idx in range(n_layers)
        ]
        peak_layer, peak_acc = max(layer_accs, key=lambda x: x[1])
        print(f"    NT{L}: layer {peak_layer:>2}  ({peak_acc:.1%})")

    if output_json:
        serialised = {
            f"L{layer_idx}_NT{L}": {
                "layer": layer_idx,
                "level": L,
                "boundary_accuracy": results[(layer_idx, L)].boundary_accuracy,
                "n_boundary_positions": results[(layer_idx, L)].n_boundary_positions,
            }
            for layer_idx in range(n_layers) for L in levels
        }
        with open(output_json, "w") as f:
            json.dump(serialised, f, indent=2)
        print(f"\nSaved per-(layer, level) results to {output_json}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-layer × per-level diagonal probe scan (Result 5 over depth)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GPT_rot .pt checkpoint")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3b.txt",
                        help="Path to grammar file (relative to project root)")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n_iters", type=int, default=5_000,
                        help="Probe training iterations (5K = enough for shape, not full accuracy)")
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of fresh strings for evaluation")
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--d_pos", type=int, default=1024,
                        help="Probe position-embedding dim. d_pos=1024 matches probing.py "
                             "(~9GB total GPU); d_pos=512 fits comfortably in 12GB.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default=None,
                        help="Save per-(layer, level) accuracies to this JSON file")
    args = parser.parse_args()

    run_per_layer_scan(
        gpt_checkpoint_path=args.checkpoint,
        cfg_path=os.path.join(project_root, args.cfg),
        levels=tuple(args.levels),
        n_probe_iters=args.n_iters,
        n_eval_samples=args.n_eval,
        batch_size=args.batch_size,
        d_pos=args.d_pos,
        device=args.device,
        output_json=args.output_json,
    )
