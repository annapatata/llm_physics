"""
Plotting for Results 6-9 of arXiv:2305.13673v4.

Input  :  An AttentionStats .pt file written by evaluation/attention.py
          (via --save).
Output :  Four PNG figures in --out_dir (default: evaluation/figures/),
          one per result, matching the paper's Figure 8 / Figure 9 layout.

Figures
-------
fig_6_position_bias.png
    Result 6 — small-multiples grid of (layer × head). Each cell plots
    Ā_{l,h,p} versus distance p on a log-x axis. Different heads peak at
    different distances; the multi-scale story is visible at a glance.

fig_7_delta_peak.png
    Result 7 — average B over (i, j) pairs where i+δ lands on an NT
    boundary at level ℓ, plotted vs δ. One line per level ℓ ∈ {2..6}.
    Expect a sharp peak at δ = 0.

fig_8_end_to_end.png
    Result 8 — B for pairs where BOTH i and j are NT boundaries at the
    same level. Left subplot: mean B by level (bar chart). Right grid:
    per-(layer, head) heatmap, one panel per level.

fig_9_ancestor_distance.png
    Result 9 — B for boundary-to-boundary pairs grouped by ancestor
    distance r = p_ℓ(j) − p_ℓ(i). One curve per level ℓ. Highest at
    r = 0 or r = 1 — the DP recurrence pattern.

CLI
---
    python evaluation/plot_attention.py \\
        --stats evaluation/attention_stats_cfg3f.pt \\
        --out_dir evaluation/figures/cfg3f
"""

import os
import sys
import argparse
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from evaluation.attention import AttentionStats, load_attention_stats


# ── Figure 6: position bias Ā_{l,h,p} ──────────────────────────────────────────

def plot_position_bias(stats: AttentionStats, out_path: str,
                       max_p: Optional[int] = None) -> None:
    """
    Small-multiples grid: (layer, head) cells, each cell shows Ā vs p (log-x).
    """
    A_bar = stats.A_bar.numpy()                # (n_layers, n_heads, P)
    n_layers, n_heads, P = A_bar.shape

    if max_p is None:
        # cut off the noisy long tail where there are few pairs
        max_p = min(P, 256)
    A_bar = A_bar[:, :, :max_p]
    p = np.arange(max_p)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(1.0 * n_heads, 0.9 * n_layers),
        sharex=True, sharey=True,
    )
    if n_layers == 1:
        axes = np.array([axes])
    if n_heads == 1:
        axes = axes[:, None]

    y_max = float(np.nanpercentile(A_bar, 99.5))

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            ax.plot(p[1:], A_bar[l, h, 1:], linewidth=0.8)
            ax.set_xscale('log')
            ax.set_ylim(0, max(y_max, 1e-4))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if l == 0:
                ax.set_title(f"h{h}", fontsize=6, pad=1)
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=6, rotation=0,
                              labelpad=10, va='center')

    fig.suptitle(
        f"Result 6 — position bias  Ā_{{l,h,p}}  vs distance p (log-x)\n"
        f"({n_layers} layers × {n_heads} heads, max_p = {max_p}, "
        f"averaged over {stats.n_strings} strings)",
        fontsize=10,
    )
    fig.supxlabel("distance p = j − i (log)", fontsize=9)
    fig.supylabel("Ā", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Figure 7: B vs δ at NT boundaries ──────────────────────────────────────────

def plot_delta_peak(stats: AttentionStats, out_path: str) -> None:
    """
    Aggregate over layers and heads. One line per level ℓ, x-axis = δ.
    Peak at δ = 0 confirms boundaries pull excess attention.
    """
    B = stats.B_R7.mean(dim=(0, 1)).numpy()    # (n_levels, n_deltas)
    deltas = np.array(stats.deltas)
    levels = stats.levels

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(levels)))

    for lvi, lv in enumerate(levels):
        ax.plot(deltas, B[lvi], marker='o', color=colors[lvi],
                label=f"NT{lv}", linewidth=2)

    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', zorder=0)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_xlabel("δ  (i + δ is at NT boundary)")
    ax.set_ylabel("mean residual B  (over all layers & heads)")
    ax.set_title(
        f"Result 7 — B concentrates at NT-end positions\n"
        f"(averaged over {stats.n_strings} strings)"
    )
    ax.set_xticks(deltas)
    ax.legend(title="boundary level", loc='upper right', frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Figure 8: B for boundary ↔ boundary pairs ──────────────────────────────────

def plot_end_to_end(stats: AttentionStats, out_path: str) -> None:
    """
    Left  : bar chart — mean B over (layer, head) for each level ℓ.
    Right : grid of heatmaps — per-(layer, head) B at each level.
    """
    B_full = stats.B_R8.numpy()                # (n_layers, n_heads, n_levels)
    B_mean = B_full.mean(axis=(0, 1))          # (n_levels,)
    levels = stats.levels
    n_levels = len(levels)
    n_layers, n_heads, _ = B_full.shape

    fig = plt.figure(figsize=(3 + 1.2 * n_levels, 4 + 0.3 * n_layers))
    gs = fig.add_gridspec(2, n_levels + 1, width_ratios=[1.2] + [1.0] * n_levels,
                          height_ratios=[1.0, 0.04], hspace=0.4, wspace=0.3)

    # Left: bar chart of aggregate B per level
    ax_bar = fig.add_subplot(gs[0, 0])
    colors = cm.viridis(np.linspace(0.1, 0.9, n_levels))
    bars = ax_bar.bar(range(n_levels), B_mean, color=colors)
    ax_bar.set_xticks(range(n_levels))
    ax_bar.set_xticklabels([f"NT{lv}" for lv in levels])
    ax_bar.axhline(0, color='gray', linewidth=0.5)
    ax_bar.set_ylabel("mean B")
    ax_bar.set_title("aggregate", fontsize=10)
    for s in ('top', 'right'):
        ax_bar.spines[s].set_visible(False)
    for bar, v in zip(bars, B_mean):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:+.3f}", ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8)

    # Right: one heatmap per level (layer × head)
    vmax = float(np.nanpercentile(np.abs(B_full), 99))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = None
    for lvi, lv in enumerate(levels):
        ax = fig.add_subplot(gs[0, lvi + 1])
        im = ax.imshow(B_full[:, :, lvi], aspect='auto', cmap='RdBu_r', norm=norm)
        ax.set_title(f"NT{lv}", fontsize=10)
        ax.set_xlabel("head")
        if lvi == 0:
            ax.set_ylabel("layer")
        else:
            ax.set_yticks([])

    # Shared colorbar
    cax = fig.add_subplot(gs[1, 1:])
    fig.colorbar(im, cax=cax, orientation='horizontal', label="B (per layer, head)")

    fig.suptitle(
        f"Result 8 — NT-end ↔ NT-end attention residual B\n"
        f"(averaged over {stats.n_strings} strings)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Figure 9: B vs ancestor distance r ─────────────────────────────────────────

def plot_ancestor_distance(stats: AttentionStats, out_path: str) -> None:
    """
    One curve per level ℓ, x-axis = r ∈ {0..max_r}.
    Aggregated over (layer, head). Highest at small r — DP recurrence pattern.
    """
    B = stats.B_R9.mean(dim=(0, 1)).numpy()    # (n_levels, max_r+1)
    levels = stats.levels
    rs = np.arange(stats.max_r + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(levels)))

    for lvi, lv in enumerate(levels):
        ax.plot(rs, B[lvi], marker='o', color=colors[lvi],
                label=f"NT{lv}", linewidth=2)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_xlabel("ancestor distance  r = p_ℓ(j) − p_ℓ(i)")
    ax.set_ylabel("mean residual B  (over all layers & heads)")
    ax.set_title(
        f"Result 9 — adjacent NT-end attention\n"
        f"(averaged over {stats.n_strings} strings)"
    )
    ax.set_xticks(rs)
    ax.legend(title="b♯(i) = b♯(j) = ℓ", loc='upper right', frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Driver ─────────────────────────────────────────────────────────────────────

def plot_all(stats: AttentionStats, out_dir: str) -> None:
    """Write all four figures into out_dir, creating it if needed."""
    os.makedirs(out_dir, exist_ok=True)
    plot_position_bias(stats,     os.path.join(out_dir, "fig_6_position_bias.png"))
    plot_delta_peak(stats,        os.path.join(out_dir, "fig_7_delta_peak.png"))
    plot_end_to_end(stats,        os.path.join(out_dir, "fig_8_end_to_end.png"))
    plot_ancestor_distance(stats, os.path.join(out_dir, "fig_9_ancestor_distance.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot attention analysis figures (Results 6-9)"
    )
    parser.add_argument("--stats", required=True,
                        help="Path to .pt file written by attention.py --save")
    parser.add_argument("--out_dir", default=os.path.join("evaluation", "figures"),
                        help="Directory to write the four PNGs into")
    args = parser.parse_args()

    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(project_root, args.out_dir))

    stats = load_attention_stats(args.stats)
    plot_all(stats, out_dir)

    print(f"Wrote 4 figures to {out_dir}:")
    for name in ("fig_6_position_bias.png",
                 "fig_7_delta_peak.png",
                 "fig_8_end_to_end.png",
                 "fig_9_ancestor_distance.png"):
        print(f"  - {name}")
