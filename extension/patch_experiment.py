"""
The activation-patching experiment (EXTENSION.md §2, §6).

Loads the frozen trials (Step B) and, for each trial, generates a completion
under several conditions, scoring each with:
  - CYK validity (structural — should break under the boundary donor patch)
  - bigram KL vs the grammar's true P(next|prev) (surface — should stay flat)

Conditions
----------
  clean              : generate from the clean prefix, no patch (baseline).
  donor              : inject the donor's hidden states at the NT-ℓ boundary
                       positions (the causal test).
  noise_boundary     : inject Gaussian noise (scaled to the residual magnitude)
                       at the SAME positions  -> specificity control: noise is
                       non-specific, so both metrics should drop together.
  noise_nonboundary  : inject the same-magnitude noise at the same NUMBER of
                       NON-boundary positions  -> sufficiency control: structure
                       should survive, showing boundaries carry the load.

All conditions run off the same frozen trial, so they differ only in what is
injected and where -- never in which string or which RNG the sampler starts from
(we reseed the sampler identically per trial for a paired comparison).

The patch is applied only in the prefix; continuation tokens are never directly
patched -- the effect reaches them through attention back onto the patched
prefix positions (the hook re-stamps every forward pass; see patching.py).
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.append(project_root)
sys.path.append(this_dir)

from cfg.grammar import load_cfg
from dp.cyk import is_valid
from models.gpt_rot import GPT2Rotary
from patching import ActivationPatcher, capture_residual, BOS_TOKEN, EOS_TOKEN
from build_corpus import prefix_boundary_mask, TARGET_LEVEL, PREFIX_LEN, CACHE_DIR

MODEL_MAX_SEQ_LEN = 512
TERMINALS = (1, 2, 3)
TERM_TO_IDX = {1: 0, 2: 1, 3: 2}

CONDITIONS = ("clean", "donor", "noise_boundary", "noise_nonboundary")


# ── Generation (mirrors evaluation.py, but a patcher hook may be active) ─────────

@torch.no_grad()
def generate(model, prefix_tokens, temperature=1.0, device="cuda"):
    """Autoregressive multinomial generation. If a patcher is enabled on the
    model, its hook fires on every model() call and re-applies the patch."""
    idx = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
    max_new = MODEL_MAX_SEQ_LEN - len(prefix_tokens)
    generated = []
    for _ in range(max_new):
        logits = model(idx)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        tok = nxt.item()
        generated.append(tok)
        if tok == EOS_TOKEN:
            break
        idx = torch.cat((idx, nxt), dim=1)
    return prefix_tokens + generated


# ── Surface metric: bigram KL vs the grammar's true transition distribution ──────

def true_bigram_distribution(corpus_payload, eps=1e-9):
    """Estimate P_true(next | prev) once from the corpus's valid strings."""
    counts = np.zeros((3, 3), dtype=np.float64)
    for s in corpus_payload["samples"]:
        add_bigrams(counts, s["string"])
    return normalize_rows(counts, eps)


def add_bigrams(counts, seq):
    for a, b in zip(seq[:-1], seq[1:]):
        if a in TERM_TO_IDX and b in TERM_TO_IDX:
            counts[TERM_TO_IDX[a], TERM_TO_IDX[b]] += 1.0


def normalize_rows(counts, eps=1e-9):
    c = counts + eps
    return c / c.sum(axis=1, keepdims=True)


def bigram_kl(cond_counts, p_true, eps=1e-9):
    """Prev-marginal-weighted KL( P_cond(.|prev) || P_true(.|prev) )."""
    total = cond_counts.sum()
    if total == 0:
        return float("nan")
    p_cond = normalize_rows(cond_counts, eps)
    prev_marg = cond_counts.sum(axis=1) / total
    kl_per_prev = (p_cond * (np.log(p_cond) - np.log(p_true))).sum(axis=1)
    return float((prev_marg * kl_per_prev).sum())


# ── One condition for one trial ─────────────────────────────────────────────────

def build_patch(model, trial, clean_sample, condition, layer, level, prefix_len,
                device, noise_gen):
    """
    Return (positions_seq, values) to patch for this condition, or (None, None)
    for the clean baseline. positions are SEQUENCE indices (BOS offset applied).
    """
    pos = trial["positions_seq"]
    n_embd = None

    if condition == "clean":
        return None, None

    if condition == "donor":
        donor_resid = capture_residual(model, trial["donor_prefix"], layer, device)
        values = donor_resid[pos]                              # (n_pos, n_embd)
        return pos, values

    # Both noise conditions need the clean residual to scale the noise.
    clean_resid = capture_residual(model, trial["clean_prefix"], layer, device)
    n_embd = clean_resid.shape[-1]

    if condition == "noise_boundary":
        sigma = clean_resid[pos].std()
        values = sigma * torch.randn(len(pos), n_embd, generator=noise_gen)
        return pos, values

    if condition == "noise_nonboundary":
        # Same NUMBER of positions, but at non-(level-ℓ-boundary) prefix slots.
        bmask = prefix_boundary_mask(clean_sample, level, prefix_len)  # terminal coords
        boundary_seq = {int(i) + 1 for i in np.nonzero(bmask)[0]}      # -> seq coords
        clean_len = clean_sample["length"]
        last_seq = min(prefix_len, clean_len)                          # valid seq range 1..last_seq
        candidates = [p for p in range(1, last_seq + 1) if p not in boundary_seq]
        k = min(len(pos), len(candidates))
        chosen = sorted(np.random.default_rng(trial["clean_idx"]).choice(
            candidates, size=k, replace=False).tolist())
        sigma = clean_resid[chosen].std()
        values = sigma * torch.randn(len(chosen), n_embd, generator=noise_gen)
        return chosen, values

    raise ValueError(f"unknown condition {condition}")


def run_trial(model, cfg, trial, clean_sample, conditions, layer, level,
              prefix_len, device, temperature, trial_seed, noise_gen):
    """Generate + score every condition for one trial. Returns dict per condition:
    {valid: bool, content: List[int]}."""
    out = {}
    for cond in conditions:
        pos, values = build_patch(model, trial, clean_sample, cond, layer, level,
                                  prefix_len, device, noise_gen)

        # Reseed the sampler identically for every condition (paired comparison).
        torch.manual_seed(trial_seed)

        if pos is None:
            completion = generate(model, trial["clean_prefix"], temperature, device)
        else:
            with ActivationPatcher(model, layer) as patcher:
                patcher.set_patch(pos, values.to(device))
                completion = generate(model, trial["clean_prefix"], temperature, device)

        content = [t for t in completion if t not in (BOS_TOKEN, EOS_TOKEN)]
        valid = len(content) > 0 and is_valid(content, cfg)
        out[cond] = {"valid": valid, "content": content}
    return out


# ── Experiment driver ───────────────────────────────────────────────────────────

def run_experiment(checkpoint_path, cfg_path, trials_path, corpus_path=None,
                   layer=-1, conditions=CONDITIONS, limit=None,
                   device="cuda", temperature=1.0, seed=0):
    trials_payload = torch.load(trials_path, weights_only=False)
    trials = trials_payload["trials"]
    level = trials_payload["meta"]["level"]
    prefix_len = trials_payload["meta"]["prefix_len"]
    if corpus_path is None:
        corpus_path = trials_payload["meta"]["corpus_path"]
        # Trials may have been frozen before the folder moved; if the stored
        # corpus path is gone, fall back to the same filename in the current
        # cache dir.
        if not os.path.exists(corpus_path):
            fallback = os.path.join(CACHE_DIR, os.path.basename(corpus_path))
            if os.path.exists(fallback):
                corpus_path = fallback
    corpus_payload = torch.load(corpus_path, weights_only=False)
    samples = corpus_payload["samples"]

    if limit is not None:
        trials = trials[:limit]

    cfg = load_cfg(cfg_path)
    p_true = true_bigram_distribution(corpus_payload)

    # Model.
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    noise_gen = torch.Generator().manual_seed(seed + 12345)

    # Accumulators.
    valid_counts = {c: 0 for c in conditions}
    bigram_counts = {c: np.zeros((3, 3), dtype=np.float64) for c in conditions}
    n = len(trials)

    for ti, trial in enumerate(tqdm(trials, desc=f"NT{level} patch @ layer {layer}")):
        clean_sample = samples[trial["clean_idx"]]
        res = run_trial(model, cfg, trial, clean_sample, conditions, layer, level,
                        prefix_len, device, temperature, seed + ti, noise_gen)
        for c in conditions:
            valid_counts[c] += int(res[c]["valid"])
            add_bigrams(bigram_counts[c], res[c]["content"])

    # ── Report ──
    cyk = {c: valid_counts[c] / n for c in conditions}
    kl = {c: bigram_kl(bigram_counts[c], p_true) for c in conditions}
    base = cyk.get("clean", float("nan"))
    base_kl = kl.get("clean", float("nan"))

    print(f"\n{'='*68}")
    print(f"  Activation patching @ NT{level}, layer {layer}   (N={n} trials)")
    print(f"{'='*68}")
    print(f"  {'condition':<18}{'CYK valid':>11}{'CYK drop':>11}"
          f"{'bigram KL':>12}{'KL drift':>11}")
    print("  " + "-" * 64)
    for c in conditions:
        drop = base - cyk[c]
        drift = kl[c] - base_kl
        print(f"  {c:<18}{cyk[c]:>10.1%}{drop:>11.1%}{kl[c]:>12.4f}{drift:>+11.4f}")
    print(f"{'='*68}")
    print("  Expected: donor -> large CYK drop, ~0 KL drift (structure breaks,")
    print("            surface holds). noise_boundary -> CYK drop AND KL rises.")
    print("            noise_nonboundary -> little CYK drop (boundaries suffice).")

    return {"cyk": cyk, "bigram_kl": kl, "n": n,
            "valid_counts": valid_counts, "bigram_counts": bigram_counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation-patching experiment")
    parser.add_argument("--checkpoint", default="gpt_checkpoint_step_6500.pt")
    parser.add_argument("--cfg", default="cfg/grammars/cfg3b.txt")
    parser.add_argument("--trials", default=os.path.join(CACHE_DIR, "trials_cfg3b_nt5.pt"))
    parser.add_argument("--corpus", default=None, help="defaults to trials' corpus_path")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first K trials (smoke test)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_experiment(
        checkpoint_path=args.checkpoint,
        cfg_path=os.path.join(project_root, args.cfg),
        trials_path=args.trials,
        corpus_path=args.corpus,
        layer=args.layer,
        limit=args.limit,
        device=args.device,
        temperature=args.temperature,
        seed=args.seed,
    )
