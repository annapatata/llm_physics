"""
Attention analysis for Results 6-9 of
    Allen-Zhu & Li 2023, "Physics of Language Models: Part 1" (arXiv:2305.13673v4).

Result 6 — Position-based attention
    The position bias  Ā_{l,h,p}  =  E_{x, (i,j): j-i=p} [ A_{l,h,j→i}(x) ]
    is the average attention weight at relative distance p = j - i, computed
    over many sampled strings. Raw attention is strongly determined by p alone,
    and different heads operate at different scales (Figure 8 in the paper).

Results 7-9 use the position-bias-removed residual

        B_{l,h,j→i}(x)  =  A_{l,h,j→i}(x)  −  Ā_{l,h,j-i}.

    B > 0 means j attends to i more than distance alone predicts; it is the
    content / structure-driven component of attention.

Result 7 — B concentrates at NT-end positions
    Average B over pairs (i, j) where the *attended-to* position i+δ is a
    boundary at level ℓ. Plot vs δ ∈ {−2, −1, 0, 1, 2}: peaks sharply at δ=0.

Result 8 — NT-end → NT-end attention
    Average B over pairs where BOTH i and j are NT boundaries at level ℓ.
    Boundary positions preferentially talk to other boundary positions.

Result 9 — Adjacent NT-end attention by ancestor distance r
    With  b♯(i) = b♯(j) = ℓ  and  r = p_ℓ(j) − p_ℓ(i),  average B as a function
    of r. Highest at r = 0 or r = 1 — exactly the DP recurrence pattern in
    which j reads back to its nearest left boundary k of the same level.

Conventions used throughout this file:
    Model positions:  0 = BOS,  1..L = terminal tokens (so model_pos k corresponds
                      to terminal index k - 1 = i_term).
    sample.boundaries[ℓ][i_term]       — b_ℓ at terminal index i_term
    sample.deepest_boundary[i_term]    — b♯ at terminal index i_term
    sample.ancestor_indices[ℓ][i_term] — p_ℓ at terminal index i_term
    All averaging skips BOS and considers only causal pairs (j > i).
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg, CFG, CFGSample
from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0
EOS_TOKEN = 4

NT_LEVELS: Tuple[int, ...] = (2, 3, 4, 5, 6)
DEFAULT_DELTAS: Tuple[int, ...] = (-2, -1, 0, 1, 2)
DEFAULT_MAX_R: int = 8


# ── Attention extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_attentions_batch(
    model: GPT2Rotary,
    batch_token_ids: List[List[int]],
    device: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Run a batch through the model with return_all_attentions=True.

    Returns
    -------
    attentions : list of n_layer tensors, each (B, n_heads, T_max, T_max).
        Rows for end-padded positions are NOT meaningful — the caller must crop
        to real positions using `lengths`.
    lengths : real (un-padded) token count of each sequence (BOS included).
    """
    lengths = [len(ids) for ids in batch_token_ids]
    T_max = max(lengths)

    padded = torch.zeros(len(batch_token_ids), T_max, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_token_ids):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    _, attentions = model(padded, return_all_attentions=True)
    return attentions, lengths


def _sample_and_tokenize(cfg: CFG, max_seq_len: int) -> Tuple[CFGSample, List[int]]:
    """Sample a string, prepend BOS, and truncate to max_seq_len tokens."""
    sample = cfg.sample_string()
    tokens = [BOS_TOKEN] + sample.string[:max_seq_len - 1]
    return sample, tokens


# ── Result 6: position bias Ā ──────────────────────────────────────────────────

@torch.no_grad()
def compute_position_bias(
    model: GPT2Rotary,
    cfg: CFG,
    n_strings: int = 200,
    batch_size: int = 2,
    device: str = 'cuda',
    max_seq_len: int = 512,
) -> torch.Tensor:
    """
    Compute Ā_{l,h,p} averaged across n_strings sampled strings.

    Ā_{l,h,p} is the average of A_{l,h,j→i}(x) over all sampled x and over all
    causal (i, j) pairs with j - i = p, restricted to real terminal positions
    (i, j ≥ 1; BOS at position 0 is excluded).

    Returns
    -------
    A_bar : (n_layers, n_heads, max_seq_len) CPU tensor.
        A_bar[..., 0]  = avg self-attention A_{l,h,j→j}.
        A_bar[..., p]  = avg A_{l,h,j→j-p} for j-p ≥ 1.
    """
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_head

    sum_bias = torch.zeros(n_layers, n_heads, max_seq_len, device=device)
    count_bias = torch.zeros(max_seq_len, device=device)

    n_done = 0
    pbar = tqdm(total=n_strings, desc="R6 / position bias")
    while n_done < n_strings:
        b = min(batch_size, n_strings - n_done)
        batch_tokens = [_sample_and_tokenize(cfg, max_seq_len)[1] for _ in range(b)]
        attentions, lengths = extract_attentions_batch(model, batch_tokens, device)

        # (n_layers, B, H, T_max, T_max).  ~ n_layers * B * H * T^2 * 4 bytes
        attn_stack = torch.stack(attentions, dim=0)

        for s in range(b):
            L = lengths[s]
            if L < 3:           # need BOS + ≥2 terminals
                continue
            # Crop to real terminal positions: rows j ∈ [1, L-1], cols i ∈ [1, L-1]
            A = attn_stack[:, s, :, 1:L, 1:L]   # (n_layers, H, T, T)
            T = A.shape[-1]
            for p in range(T):
                # Diagonal at offset -p collects (j, i=j-p) pairs.
                diag = torch.diagonal(A, offset=-p, dim1=-2, dim2=-1)   # (n_layers, H, T-p)
                sum_bias[:, :, p] += diag.sum(dim=-1)
                count_bias[p] += (T - p)

        del attn_stack, attentions
        n_done += b
        pbar.update(b)
    pbar.close()

    safe_count = count_bias.clamp(min=1).view(1, 1, -1)
    return (sum_bias / safe_count).cpu()


# ── Results 7-9: residual statistics on B = A - Ā(j-i) ─────────────────────────

@dataclass
class AttentionStats:
    """
    Aggregated attention statistics for Results 6-9.

    Shapes
    ------
    A_bar    : (n_layers, n_heads, max_seq_len)         ← Result 6
    B_R7     : (n_layers, n_heads, n_levels, n_deltas)  ← Result 7
    B_R8     : (n_layers, n_heads, n_levels)            ← Result 8
    B_R9     : (n_layers, n_heads, n_levels, max_r+1)   ← Result 9

    Companion count_R* tensors give the number of pairs averaged per cell so the
    caller can re-aggregate (e.g. weighted average across CFGs or strings).
    """
    A_bar: torch.Tensor

    deltas: Tuple[int, ...]
    levels: Tuple[int, ...]
    max_r: int

    B_R7: torch.Tensor
    B_R8: torch.Tensor
    B_R9: torch.Tensor

    count_R7: torch.Tensor
    count_R8: torch.Tensor
    count_R9: torch.Tensor

    n_strings: int = 0


@torch.no_grad()
def compute_attention_stats(
    model: GPT2Rotary,
    cfg: CFG,
    A_bar: torch.Tensor,
    n_strings: int = 200,
    batch_size: int = 2,
    device: str = 'cuda',
    max_seq_len: int = 512,
    deltas: Tuple[int, ...] = DEFAULT_DELTAS,
    levels: Tuple[int, ...] = NT_LEVELS,
    max_r: int = DEFAULT_MAX_R,
) -> AttentionStats:
    """
    Single-pass accumulator for Results 7, 8, 9.

    Given the pre-computed position-bias tensor `A_bar` (see compute_position_bias),
    sample n_strings strings, form the residual B = A - Ā(j-i) per string, and
    accumulate sums and counts under each conditioning rule.
    """
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_head
    n_levels = len(levels)
    n_deltas = len(deltas)
    A_bar = A_bar.to(device)

    sum_R7 = torch.zeros(n_layers, n_heads, n_levels, n_deltas, device=device)
    cnt_R7 = torch.zeros(n_levels, n_deltas, device=device)

    sum_R8 = torch.zeros(n_layers, n_heads, n_levels, device=device)
    cnt_R8 = torch.zeros(n_levels, device=device)

    sum_R9 = torch.zeros(n_layers, n_heads, n_levels, max_r + 1, device=device)
    cnt_R9 = torch.zeros(n_levels, max_r + 1, device=device)

    n_done = 0
    pbar = tqdm(total=n_strings, desc="R7-R9 / B residuals")
    while n_done < n_strings:
        b = min(batch_size, n_strings - n_done)
        batch_samples: List[CFGSample] = []
        batch_tokens: List[List[int]] = []
        for _ in range(b):
            s, t = _sample_and_tokenize(cfg, max_seq_len)
            batch_samples.append(s)
            batch_tokens.append(t)

        attentions, lengths = extract_attentions_batch(model, batch_tokens, device)
        attn_stack = torch.stack(attentions, dim=0)   # (n_layers, B, H, T_max, T_max)

        for s_idx in range(b):
            L = lengths[s_idx]
            T = L - 1                                  # real terminal count
            if T < 2:
                continue
            sample = batch_samples[s_idx]
            assert T == min(sample.length, max_seq_len - 1)

            # ── Build B = A - Ā(j-i) on the real-token block ───────────────────
            A = attn_stack[:, s_idx, :, 1:L, 1:L]     # (n_layers, H, T, T)
            jj = torch.arange(T, device=device).view(-1, 1)
            ii = torch.arange(T, device=device).view(1, -1)
            dist = (jj - ii).clamp(min=0)              # (T, T) — 0 for j<i (masked out below)
            A_bar_grid = A_bar[:, :, dist]             # (n_layers, H, T, T)
            B = A - A_bar_grid                         # (n_layers, H, T, T)

            causal = (jj > ii)                         # strict: exclude self-attention

            # ── Boundary labels for this sample (terminal-indexed, length T) ───
            b_level = torch.zeros(n_levels, T, dtype=torch.bool, device=device)
            anc_idx = torch.zeros(n_levels, T, dtype=torch.long, device=device)
            for lvi, lv in enumerate(levels):
                b_level[lvi] = torch.tensor(
                    sample.boundaries[lv][:T], dtype=torch.bool, device=device
                )
                anc_idx[lvi] = torch.tensor(
                    sample.ancestor_indices[lv][:T], dtype=torch.long, device=device
                )
            deepest = torch.tensor(
                sample.deepest_boundary[:T], dtype=torch.long, device=device
            )

            # ── Result 7 : column position i + δ is a level-ℓ boundary ─────────
            i_idx = torch.arange(T, device=device)
            for lvi in range(n_levels):
                for di, delta in enumerate(deltas):
                    shifted = i_idx + delta
                    in_range = (shifted >= 0) & (shifted < T)
                    safe_shift = shifted.clamp(0, T - 1)
                    is_boundary_at_shift = b_level[lvi][safe_shift] & in_range   # (T,)
                    col_mask = is_boundary_at_shift.view(1, -1)                  # (1, T)
                    M = causal & col_mask                                        # (T, T)
                    n_pairs = int(M.sum().item())
                    if n_pairs > 0:
                        sum_R7[:, :, lvi, di] += (B * M).sum(dim=(-1, -2))
                        cnt_R7[lvi, di] += n_pairs

            # ── Result 8 : both i and j are level-ℓ boundaries ─────────────────
            for lvi in range(n_levels):
                row_mask = b_level[lvi].view(-1, 1)     # j is boundary
                col_mask = b_level[lvi].view(1, -1)     # i is boundary
                M = causal & row_mask & col_mask
                n_pairs = int(M.sum().item())
                if n_pairs > 0:
                    sum_R8[:, :, lvi] += (B * M).sum(dim=(-1, -2))
                    cnt_R8[lvi] += n_pairs

            # ── Result 9 : b♯(i) = b♯(j) = ℓ  and  p_ℓ(j) − p_ℓ(i) = r ─────────
            for lvi, lv in enumerate(levels):
                same_deep = (deepest == lv)             # (T,)
                if not same_deep.any():
                    continue
                row_mask = same_deep.view(-1, 1)
                col_mask = same_deep.view(1, -1)
                pair_mask = causal & row_mask & col_mask
                if not pair_mask.any():
                    continue
                p_ell = anc_idx[lvi]                    # (T,)
                r_grid = p_ell.view(-1, 1) - p_ell.view(1, -1)   # (T, T) ≥ 0 on causal cells
                for r in range(max_r + 1):
                    M = pair_mask & (r_grid == r)
                    n_pairs = int(M.sum().item())
                    if n_pairs > 0:
                        sum_R9[:, :, lvi, r] += (B * M).sum(dim=(-1, -2))
                        cnt_R9[lvi, r] += n_pairs

        del attn_stack, attentions
        n_done += b
        pbar.update(b)
    pbar.close()

    def _safe(num: torch.Tensor, denom: torch.Tensor, leading_dims: int) -> torch.Tensor:
        d = denom.clamp(min=1)
        view = (1,) * leading_dims + tuple(d.shape)
        return (num / d.view(view)).cpu()

    return AttentionStats(
        A_bar=A_bar.cpu(),
        deltas=deltas, levels=levels, max_r=max_r,
        B_R7=_safe(sum_R7, cnt_R7, leading_dims=2),
        B_R8=_safe(sum_R8, cnt_R8, leading_dims=2),
        B_R9=_safe(sum_R9, cnt_R9, leading_dims=2),
        count_R7=cnt_R7.cpu(),
        count_R8=cnt_R8.cpu(),
        count_R9=cnt_R9.cpu(),
        n_strings=n_strings,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def save_attention_stats(stats: AttentionStats, path: str) -> None:
    torch.save({
        'A_bar': stats.A_bar,
        'deltas': stats.deltas,
        'levels': stats.levels,
        'max_r': stats.max_r,
        'B_R7': stats.B_R7, 'count_R7': stats.count_R7,
        'B_R8': stats.B_R8, 'count_R8': stats.count_R8,
        'B_R9': stats.B_R9, 'count_R9': stats.count_R9,
        'n_strings': stats.n_strings,
    }, path)


def load_attention_stats(path: str) -> AttentionStats:
    d = torch.load(path, map_location='cpu')
    return AttentionStats(
        A_bar=d['A_bar'],
        deltas=tuple(d['deltas']), levels=tuple(d['levels']), max_r=int(d['max_r']),
        B_R7=d['B_R7'], B_R8=d['B_R8'], B_R9=d['B_R9'],
        count_R7=d['count_R7'], count_R8=d['count_R8'], count_R9=d['count_R9'],
        n_strings=int(d['n_strings']),
    )


# ── Summary printing ───────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def print_summary(stats: AttentionStats, max_p_print: int = 16) -> None:
    """Print a paper-style textual summary. Tensors stay on CPU."""
    n_layers, n_heads, _ = stats.A_bar.shape

    print("\n" + "=" * 72)
    print("  Result 6 — Position bias Ā_{l,h,p}, averaged over layers and heads")
    print("=" * 72)
    A_bar_lh = stats.A_bar.mean(dim=(0, 1))   # (max_seq_len,)
    header = "  p   " + " ".join(f"{p:>6d}" for p in range(min(max_p_print, A_bar_lh.numel())))
    values = "  Ā   " + " ".join(f"{A_bar_lh[p].item():>6.4f}"
                                  for p in range(min(max_p_print, A_bar_lh.numel())))
    print(header)
    print(values)
    print("  (full curve in stats.A_bar — shape "
          f"{tuple(stats.A_bar.shape)}; head/layer breakdown available there.)")

    print("\n" + "=" * 72)
    print("  Result 7 — B at NT-end positions, averaged over layers/heads")
    print("  Rows = level ℓ, cols = δ (i + δ position relative to boundary)")
    print("=" * 72)
    B7_lh = stats.B_R7.mean(dim=(0, 1))       # (n_levels, n_deltas)
    print(f"  {'level':>6s}  " + "  ".join(f"δ={d:+d}".rjust(8) for d in stats.deltas))
    for lvi, lv in enumerate(stats.levels):
        row = "  ".join(_fmt(B7_lh[lvi, di].item()).rjust(8) for di in range(len(stats.deltas)))
        print(f"  NT{lv:<4d}  {row}")
    print("  (Expected: peak at δ=0 — NT boundaries attract excess attention.)")

    print("\n" + "=" * 72)
    print("  Result 8 — B for NT-end ↔ NT-end pairs at the same level")
    print("=" * 72)
    B8_lh = stats.B_R8.mean(dim=(0, 1))       # (n_levels,)
    for lvi, lv in enumerate(stats.levels):
        print(f"  NT{lv}  mean B = {_fmt(B8_lh[lvi].item())}  "
              f"(n_pairs = {int(stats.count_R8[lvi].item()):>10d})")
    print("  (Expected: strongly positive — boundaries preferentially attend to boundaries.)")

    print("\n" + "=" * 72)
    print("  Result 9 — B by ancestor distance r at b♯(i)=b♯(j)=ℓ")
    print("  Rows = level ℓ, cols = r = p_ℓ(j) − p_ℓ(i)")
    print("=" * 72)
    B9_lh = stats.B_R9.mean(dim=(0, 1))       # (n_levels, max_r+1)
    header = f"  {'level':>6s}  " + "  ".join(f"r={r}".rjust(8) for r in range(stats.max_r + 1))
    print(header)
    for lvi, lv in enumerate(stats.levels):
        row = "  ".join(_fmt(B9_lh[lvi, r].item()).rjust(8) for r in range(stats.max_r + 1))
        print(f"  NT{lv:<4d}  {row}")
    print("  (Expected: monotone decrease in r — nearest-left-boundary DP recurrence.)")

    print("\n" + "=" * 72)
    print(f"  Aggregated over n_strings = {stats.n_strings}.")
    print(f"  Raw tensors carry the full (layer, head) breakdown — "
          f"call stats.B_R7 / .B_R8 / .B_R9 for per-head plots.")
    print("=" * 72)


# ── Main experiment ────────────────────────────────────────────────────────────

def run_attention_experiment(
    gpt_checkpoint_path: Optional[str],
    cfg_path: str,
    n_strings_bias: int = 200,
    n_strings_residual: int = 200,
    batch_size: int = 2,
    device: str = 'cuda',
    random_gpt: bool = False,
    max_seq_len: int = 512,
    save_path: Optional[str] = None,
) -> AttentionStats:
    """
    Full attention analysis pipeline for Results 6-9.

    Pass random_gpt=True for the GPT_rand control (no checkpoint needed) —
    untrained attention should show ~no structured B residual.
    """
    cfg = load_cfg(cfg_path)

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    if not random_gpt:
        assert gpt_checkpoint_path is not None, "Provide --checkpoint or pass --random_gpt"
        state = torch.load(gpt_checkpoint_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"Loaded GPT_rot weights from {gpt_checkpoint_path}")
    else:
        print("GPT_rand control: using RANDOM untrained weights "
              "(B residuals should be near zero / unstructured).")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    print(f"\nPass 1/2 — computing position bias Ā over {n_strings_bias} strings ...")
    A_bar = compute_position_bias(
        model, cfg,
        n_strings=n_strings_bias,
        batch_size=batch_size,
        device=device,
        max_seq_len=max_seq_len,
    )

    print(f"\nPass 2/2 — computing residual B statistics over "
          f"{n_strings_residual} strings ...")
    stats = compute_attention_stats(
        model, cfg, A_bar,
        n_strings=n_strings_residual,
        batch_size=batch_size,
        device=device,
        max_seq_len=max_seq_len,
    )

    print_summary(stats)

    if save_path is not None:
        save_attention_stats(stats, save_path)
        print(f"\nSaved attention stats to {save_path}")

    return stats


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention analysis (Results 6-9)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to GPT_rot .pt checkpoint (omit with --random_gpt)")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3f.txt",
                        help="Grammar file path (relative to project root)")
    parser.add_argument("--n_strings_bias", type=int, default=200,
                        help="# strings for the position-bias pass")
    parser.add_argument("--n_strings_residual", type=int, default=200,
                        help="# strings for the B-residual pass")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_gpt", action="store_true",
                        help="GPT_rand control: skip checkpoint, use random weights")
    parser.add_argument("--save", default=None,
                        help="Optional path to save AttentionStats as a .pt file")
    args = parser.parse_args()

    cfg_path = (args.cfg if os.path.isabs(args.cfg)
                else os.path.join(project_root, args.cfg))

    run_attention_experiment(
        gpt_checkpoint_path=args.checkpoint,
        cfg_path=cfg_path,
        n_strings_bias=args.n_strings_bias,
        n_strings_residual=args.n_strings_residual,
        batch_size=args.batch_size,
        device=args.device,
        random_gpt=args.random_gpt,
        max_seq_len=args.max_seq_len,
        save_path=args.save,
    )
