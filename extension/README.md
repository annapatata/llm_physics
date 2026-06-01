# Extension — Activation Patching of NT-Boundary Hidden States

A causal follow-up to the probing results (Result 5). Probing shows NT identity
is *decodable* from the hidden state at NT-boundary positions; this extension
tests whether that encoding is *used* — by overwriting (patching) those hidden
states during generation and checking whether structured generation breaks.

The full design rationale lives in `documentation/EXTENSION.md`. This README is
just a map of the files and how to run them.

## The core idea in one line

Swap a clean string's NT5-boundary hidden states for a donor string's (same
positions, *different* NT identity). If the model **uses** the encoding, the
completion stops being grammatical (CYK-invalid) while its local surface
statistics (bigram distribution) stay roughly unchanged — a structure-vs-surface
*dissociation* that a generic perturbation would not produce.

## Pipeline (run in this order)

| Step | File | What it does | Needs |
|---|---|---|---|
| 1 | `patching.py` | The patch hook + an identity check (no-op patch ⇒ identical logits). Run standalone to validate the mechanism. | nothing (CPU, random weights) |
| 2 | `build_corpus.py` | Sample M valid CFG strings once, store full parse annotations, and **measure the donor acceptance rate**. | grammar only |
| 3 | `build_trials.py` | Match N `(clean, donor)` pairs from the corpus by pure indexing; freeze them to disk. | corpus (step 2) |
| 4 | `patch_experiment.py` | For each trial, generate under 4 conditions and score CYK validity (structural) + bigram KL (surface). | trials (step 3) **and the GPT checkpoint** |

Outputs land in `extension/cache/`:
- `corpus_cfg3b.pt` — the M sampled strings + annotations.
- `trials_cfg3b_nt5.pt` — the N frozen clean/donor trials.

## Commands

```bash
# 1. validate the hook (CPU, no data, no checkpoint)
python extension/patching.py

# 2. build the corpus + see the acceptance rate
python extension/build_corpus.py --M 2000

# 3. freeze 500 trials
python extension/build_trials.py --N 500

# 4. smoke-test the experiment on 5 trials, then run full (GPU)
python extension/patch_experiment.py --limit 5 --device cpu
python extension/patch_experiment.py --checkpoint gpt_checkpoint_step_6500.pt --device cuda
```

## The four conditions (step 4)

| Condition | Injection | Expected | Role |
|---|---|---|---|
| `clean` | none | baseline CYK < 100% (undertrained model) | reference |
| `donor` | donor's hidden states @ NT5 boundaries | **CYK drops, bigram KL flat** | the causal test |
| `noise_boundary` | Gaussian noise @ same positions | CYK drops **and** bigram KL rises | specificity control |
| `noise_nonboundary` | noise @ same # of non-boundary positions | CYK ~unchanged | sufficiency control |

The headline result is the contrast between `donor` (structure-only collapse)
and `noise_boundary` (collapse of both) — that gap is the causal evidence.

## Key conventions (easy to trip on)

- **Target:** NT5, layer `-1` (the last block — the exact residual stream
  `evaluation/probing.py` reads, so the patch site == the probe site).
- **BOS offset:** annotations index terminals (no BOS); the model sequence is
  `[BOS] + string`, so terminal `i` is at sequence index `i+1`. Trials store both
  `positions_terminal` (for annotation lookups) and `positions_seq` (for the
  hook). The patch happens at `positions_seq`.
- **Donor filter (strategy A):** a donor qualifies if it shares ≥ 3 NT5 boundary
  positions with the clean string *at the same absolute prefix index*, each with
  a **different** NT5 identity. Only those shared positions are patched.
- **Prefix-only patch:** continuation tokens are never directly patched; the
  effect reaches them through attention. The hook re-stamps every forward pass.

## Paths after the move

These files were moved from `evaluation/` into `extension/`. All path resolution
is relative to the file location (`project_root = dirname(dirname(__file__))`),
so the move needed only one change: `CACHE_DIR` now points at `extension/cache`.
