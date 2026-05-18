"""
Probing experiments for Results 4 and 5.

Result 4 — Full multi-head linear probe: predict ancestor symbol s_ℓ(i) at every
position using all hidden states of the sequence (non-causal position attention).

Result 5 — Diagonal probe at NT boundaries: same architecture but attention is
restricted to only E_i itself (delta=0). Evaluated ONLY at boundary positions
where b_ℓ(i) = 1 (the last token of a level-ℓ subtree).

Key facts:
- GPT model is frozen throughout; only probe weights are trained.
- A separate probe is trained per level ℓ ∈ {2,3,4,5,6}.
- Paper hyperparameters: H=16 heads, d'=1024, lr=3e-3, 30K iters, batch 60.
- Control: run with random_gpt=True to verify learned (not architectural) encoding.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg, CFG, CFGSample
from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0
EOS_TOKEN = 4


# ── Hidden state extraction ────────────────────────────────────────────────────

def _make_hook(store: dict):
    def hook_fn(module, input, output):
        # GPTBlock returns plain tensor (no attention) or (tensor, weights)
        out = output[0] if isinstance(output, tuple) else output
        store['last'] = out.detach()
    return hook_fn


def extract_hidden_states_batch(
    model: GPT2Rotary,
    batch_token_ids: List[List[int]],
    device: str,
) -> List[torch.Tensor]:
    """
    Run a batch of token sequences through the frozen model in one forward pass.
    Returns one (seq_len, n_embd) CPU tensor per input sequence.

    End-padding is safe here: causal attention means position j only attends to
    positions ≤ j, so appending padding tokens cannot change the hidden states
    at real token positions.
    """
    lengths = [len(ids) for ids in batch_token_ids]
    max_len = max(lengths)

    padded = torch.zeros(len(batch_token_ids), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_token_ids):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    store = {}
    handle = model.blocks[-1].register_forward_hook(_make_hook(store))

    with torch.no_grad():
        model(padded)

    handle.remove()

    hidden = store['last']  # (B, max_len, n_embd)
    return [hidden[i, :lengths[i]].cpu() for i in range(len(batch_token_ids))]


# ── Label utilities ────────────────────────────────────────────────────────────

def build_label_mapping(cfg: CFG, level: int, n_samples: int = 300) -> Dict[int, int]:
    """
    Discover all NT symbols at `level` by sampling, return sorted {symbol: class_idx}.
    For these CFGs each level has exactly 3 NTs, so 300 samples is more than enough.
    """
    symbols: set = set()
    for _ in range(n_samples):
        sample = cfg.sample_string()
        if level in sample.ancestor_symbols:
            symbols.update(sample.ancestor_symbols[level])
    mapping = {sym: idx for idx, sym in enumerate(sorted(symbols))}
    return mapping


# ── Probe model (Equation 4.2) ─────────────────────────────────────────────────

class MultiHeadLinearProbe(nn.Module):
    """
    G_i(x) = Σ_{r,k}  w_{r,i→k} · f_r(E_k(x))

    w_{r,i→k} = softmax_k( <P_{i,r}, P_{k,r}> / sqrt(d') )   ← position-only weights
    f_r : R^{n_embd} → R^{n_classes}                          ← linear map per head

    The whole function is LINEAR in the hidden states E_k — the w weights depend
    only on positions i and k, not on content. That's what makes this a linear probe.

    Hyperparameters from paper Appendix A.4:
        H = 16 heads, d' = 1024, n_classes = 3 (NTs per level).
    """

    def __init__(
        self,
        n_embd: int = 768,
        n_heads: int = 16,
        d_pos: int = 1024,
        n_classes: int = 3,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.d_pos = d_pos

        # P_{i,r}: position embedding for position i in head r
        # Stored flat: (max_seq_len, n_heads * d_pos)
        self.pos_emb = nn.Embedding(max_seq_len, n_heads * d_pos)

        # f_r: linear map R^{n_embd} → R^{n_classes}, one per head
        # Stored flat as (n_embd, n_heads * n_classes)
        self.linear = nn.Linear(n_embd, n_heads * n_classes, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden    : (B, T, n_embd)  — frozen hidden states from last GPT layer
        attn_mask : (T, T) bool, True = allowed to attend. None means full attention.
        returns   : (B, T, n_classes)
        """
        B, T, _ = hidden.shape

        # Position embeddings: (T, n_heads, d_pos)
        pos = torch.arange(T, device=hidden.device)
        P = self.pos_emb(pos).view(T, self.n_heads, self.d_pos)
        P = P.permute(1, 0, 2)  # (H, T, d_pos)

        # Attention scores: (H, T, T) where [r, i, k] = <P_{i,r}, P_{k,r}> / sqrt(d')
        scores = torch.bmm(P, P.transpose(-2, -1)) / math.sqrt(self.d_pos)  # (H, T, T)

        if attn_mask is not None:
            # attn_mask: (T, T), broadcast over heads
            scores = scores.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (H, T, T)  — w_{r,i→k}

        # f_r(E_k): (B, T, H*n_classes) → (B, T, H, n_classes)
        transformed = self.linear(hidden).view(B, T, self.n_heads, self.n_classes)

        # G_i = Σ_{r,k} w[r,i,k] * f_r(E_k)
        # → einsum over head h, source position k
        output = torch.einsum('hik,bkhc->bihc', weights, transformed)  # (B, T, H, n_classes)

        return output.sum(dim=2)  # (B, T, n_classes)

    @staticmethod
    def make_diagonal_mask(T: int, delta: int, device) -> torch.Tensor:
        """
        Boolean mask where position i may attend to k only if |i - k| <= delta.
        delta=0 → diagonal only (E_i alone used to predict s_ℓ(i)).
        delta=1 → tridiagonal (E_{i-1}, E_i, E_{i+1}).
        """
        idx = torch.arange(T, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (T, T)
        return dist <= delta


# ── Training ───────────────────────────────────────────────────────────────────

def _collect_probe_batch(
    cfg: CFG,
    model: GPT2Rotary,
    level: int,
    label_map: Dict[int, int],
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `batch_size` fresh strings, extract hidden states in one batched GPT
    forward pass, return padded (B, T, n_embd) hidden tensor and (B, T) label tensor
    with -100 for padding positions (ignored by cross_entropy).
    """
    samples, token_id_lists = [], []

    while len(samples) < batch_size:
        s = cfg.sample_string()
        if level not in s.ancestor_symbols:
            continue
        # BOS + terminals (cap at 511 to stay within 512 with BOS)
        toks = [BOS_TOKEN] + s.string[:511]
        samples.append(s)
        token_id_lists.append(toks)

    # One batched GPT forward pass — much faster than batch_size separate passes
    hidden_list = extract_hidden_states_batch(model, token_id_lists, device)

    max_sym_len = max(min(s.length, 511) for s in samples)
    B = len(samples)
    n_embd = hidden_list[0].shape[-1]

    padded_hidden = torch.zeros(B, max_sym_len, n_embd)
    label_tensor = torch.full((B, max_sym_len), -100, dtype=torch.long)

    for i, (s, hidden) in enumerate(zip(samples, hidden_list)):
        sym_len = min(s.length, 511)
        # hidden[0] = BOS hidden state (no label); hidden[1..sym_len] = token hiddens
        padded_hidden[i, :sym_len] = hidden[1:sym_len + 1]
        labels = [label_map[sym] for sym in s.ancestor_symbols[level][:sym_len]]
        label_tensor[i, :sym_len] = torch.tensor(labels, dtype=torch.long)

    return padded_hidden, label_tensor


def train_probe(
    probe: MultiHeadLinearProbe,
    gpt_model: GPT2Rotary,
    cfg: CFG,
    level: int,
    label_map: Dict[int, int],
    n_iters: int = 30_000,
    batch_size: int = 60,
    device: str = 'cuda',
    diagonal_delta: Optional[int] = None,
) -> MultiHeadLinearProbe:
    """
    Train probe to predict ancestor symbol s_ℓ(i) at the given level.

    diagonal_delta: if set, restrict attention to |i-k| <= delta during training.
    Use None for the full probe (Result 4), 0 for the diagonal probe (Result 5).
    """
    probe.to(device)
    gpt_model.eval()
    for p in gpt_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=1e-3
    )

    mode = "diagonal" if diagonal_delta is not None else "full"
    progress = tqdm(range(1, n_iters + 1), desc=f"Level {level} [{mode}]")
    ema_loss = None

    for step in progress:
        hidden, labels = _collect_probe_batch(
            cfg, gpt_model, level, label_map, batch_size, device
        )
        hidden = hidden.to(device)
        labels = labels.to(device)

        T = hidden.shape[1]
        mask = None
        if diagonal_delta is not None:
            mask = MultiHeadLinearProbe.make_diagonal_mask(T, diagonal_delta, device)

        optimizer.zero_grad()
        logits = probe(hidden, attn_mask=mask)  # (B, T, n_classes)
        loss = F.cross_entropy(
            logits.reshape(-1, probe.n_classes),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()

        ema_loss = loss.item() if ema_loss is None else 0.95 * ema_loss + 0.05 * loss.item()
        if step % 500 == 0:
            progress.set_postfix({'ema_loss': f'{ema_loss:.4f}'})

    return probe


# ── Evaluation ─────────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    level: int
    full_accuracy: float      # Result 4: full attention, all positions
    boundary_accuracy: float  # Result 5: diagonal attention, boundary positions only
    n_full_positions: int
    n_boundary_positions: int


def evaluate_probe(
    full_probe: MultiHeadLinearProbe,
    diag_probe: MultiHeadLinearProbe,
    gpt_model: GPT2Rotary,
    cfg: CFG,
    level: int,
    label_map: Dict[int, int],
    n_eval: int = 500,
    device: str = 'cuda',
) -> ProbeResult:
    """
    Evaluate both probes on fresh strings.

    Result 4: full_probe with full attention → accuracy at every position.
    Result 5: diag_probe with diagonal mask → accuracy at boundary positions only.
    """
    full_probe.eval()
    diag_probe.eval()
    gpt_model.eval()

    full_correct = full_total = 0
    boundary_correct = boundary_total = 0

    with torch.no_grad():
        for _ in tqdm(range(n_eval), desc=f"Eval level {level}", leave=False):
            sample = cfg.sample_string()
            if level not in sample.ancestor_symbols:
                continue

            sym_len = min(sample.length, 511)
            token_ids = [BOS_TOKEN] + sample.string[:511]
            hidden = extract_hidden_states_batch(gpt_model, [token_ids], device)[0]
            hidden_tokens = hidden[1:sym_len + 1].unsqueeze(0).to(device)  # (1, T, n_embd)
            T = hidden_tokens.shape[1]

            labels = torch.tensor(
                [label_map[sym] for sym in sample.ancestor_symbols[level][:sym_len]],
                dtype=torch.long, device=device,
            )
            is_boundary = torch.tensor(
                sample.boundaries[level][:sym_len], dtype=torch.bool, device=device
            )

            # ── Result 4: full probe, all positions ──────────────────────────
            logits_full = full_probe(hidden_tokens).squeeze(0)   # (T, n_classes)
            preds_full = logits_full.argmax(dim=-1)
            full_correct += (preds_full == labels).sum().item()
            full_total += T

            # ── Result 5: diagonal probe, boundary positions only ────────────
            diag_mask = MultiHeadLinearProbe.make_diagonal_mask(T, delta=0, device=device)
            logits_diag = diag_probe(hidden_tokens, attn_mask=diag_mask).squeeze(0)
            preds_diag = logits_diag.argmax(dim=-1)

            if is_boundary.any():
                boundary_correct += (preds_diag[is_boundary] == labels[is_boundary]).sum().item()
                boundary_total += is_boundary.sum().item()

    return ProbeResult(
        level=level,
        full_accuracy=full_correct / full_total if full_total > 0 else 0.0,
        boundary_accuracy=boundary_correct / boundary_total if boundary_total > 0 else 0.0,
        n_full_positions=full_total,
        n_boundary_positions=boundary_total,
    )


# ── Main experiment ────────────────────────────────────────────────────────────

def run_probing_experiment(
    gpt_checkpoint_path: str,
    cfg_path: str,
    levels: Tuple[int, ...] = (2, 3, 4, 5, 6),
    n_probe_iters: int = 30_000,
    n_eval_samples: int = 500,
    device: str = 'cuda',
    random_gpt: bool = False,
) -> List[ProbeResult]:
    """
    Full probing experiment for Results 4 and 5.

    Trains two probes per level (full + diagonal), evaluates them, prints a
    summary table matching Figure 5 and Figure 7 from the paper.

    random_gpt=True: loads random weights instead of checkpoint — this is the
    GPT_rand control that should fail, proving the trained model's accuracy
    is learned, not an architectural artifact.
    """
    cfg = load_cfg(cfg_path)

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    if not random_gpt:
        state = torch.load(gpt_checkpoint_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"Loaded GPT weights from {gpt_checkpoint_path}")
    else:
        print("GPT_rand: using RANDOM weights (control condition — should give ~33% accuracy)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    results: List[ProbeResult] = []

    for level in levels:
        print(f"\n{'='*55}")
        print(f"  Level {level} (NT{level})")
        print(f"{'='*55}")

        label_map = build_label_mapping(cfg, level)
        print(f"  NT symbols → class indices: {label_map}")
        n_classes = len(label_map)

        # Train full probe (Result 4)
        full_probe = MultiHeadLinearProbe(
            n_embd=768, n_heads=16, d_pos=1024, n_classes=n_classes
        ).to(device)
        full_probe = train_probe(
            full_probe, model, cfg, level, label_map,
            n_iters=n_probe_iters, batch_size=60,
            device=device, diagonal_delta=None,
        )

        # Train diagonal probe (Result 5)
        diag_probe = MultiHeadLinearProbe(
            n_embd=768, n_heads=16, d_pos=1024, n_classes=n_classes
        ).to(device)
        diag_probe = train_probe(
            diag_probe, model, cfg, level, label_map,
            n_iters=n_probe_iters, batch_size=60,
            device=device, diagonal_delta=0,
        )

        result = evaluate_probe(
            full_probe, diag_probe, model, cfg, level, label_map,
            n_eval=n_eval_samples, device=device,
        )
        results.append(result)

        print(
            f"  Result 4 (full):     {result.full_accuracy:.1%}  "
            f"({result.n_full_positions} positions)"
        )
        print(
            f"  Result 5 (diagonal): {result.boundary_accuracy:.1%}  "
            f"({result.n_boundary_positions} boundary positions)"
        )

    # Summary table (matches paper Figure 5 / Figure 7 layout)
    print("\n" + "="*60)
    print(f"  {'Level':<8} {'Full (R4)':<18} {'Diagonal@boundary (R5)'}")
    print("  " + "-"*56)
    for r in results:
        label = f"NT{r.level}"
        print(f"  {label:<8} {r.full_accuracy:<18.1%} {r.boundary_accuracy:.1%}")
    print("="*60)

    if random_gpt:
        print("\nNOTE: these are random-weight (GPT_rand) control numbers.")
        print("Expected: ~33% (chance level for 3-class) for all levels.")
    else:
        print("\nPaper targets (GPT_rot, cfg3f):")
        print("  NT6: 100%  NT5: ~97%  NT4: ~95%  NT3: ~93%  NT2: ~94%")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probing experiment (Results 4-5)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GPT_rot .pt checkpoint")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3f.txt",
                        help="Path to grammar file (relative to project root)")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n_iters", type=int, default=30_000,
                        help="Probe training iterations per level")
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of fresh strings for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_gpt", action="store_true",
                        help="Use random GPT weights (GPT_rand control)")
    args = parser.parse_args()

    run_probing_experiment(
        gpt_checkpoint_path=args.checkpoint,
        cfg_path=os.path.join(project_root, args.cfg),
        levels=tuple(args.levels),
        n_probe_iters=args.n_iters,
        n_eval_samples=args.n_eval,
        device=args.device,
        random_gpt=args.random_gpt,
    )
