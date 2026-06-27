# Extension A — Implementation Notes (Synonymic Invariance)

This document records **exactly what was built** to turn
[extension_A_synonymic_invariance.md](extension_A_synonymic_invariance.md) into
runnable code, how the experiment connects to our original reproduction, and how
long it takes to run. Nothing here needs retraining — Extension A is pure
forward-pass analysis on the **frozen** GPT checkpoint.

---

## 1. What the extension is (one paragraph)

The Random Hierarchy Model paper (`secondpaper.pdf`) and our Allen-Zhu
reproduction study the **same object**: a layered random CFG where every symbol
has several interchangeable rules ("synonyms"). The RHM's central measurement is
**synonymic sensitivity** `S_{k,ℓ}` — how much a network's layer-`k`
representation moves when you swap one synonym for another that *shouldn't matter*
(same parent symbol, different wording). Low `S` = the layer is **invariant** =
it has thrown away within-tree detail and kept only the constituent's identity.
Extension A measures `S_{k,ℓ}` on our already-trained GPT. It needs **no
retraining and no probe training** — only forward passes with hooks on every
transformer block.

---

## 2. How it connects to our original paper

We already have two results about the model's internals; this is the third corner:

| Result | Question | Method |
|---|---|---|
| Probing (R4/R5) | What is **present**? — can you read `s_ℓ` out of the boundary hidden state? | trained linear probes |
| Patching (extension) | What is **used**? — does overwriting the boundary state break generation? | activation patching |
| **Synonymic invariance (Extension A)** | What is **discarded**? — is synonym identity collapsed away in the representation? | `S_{k,ℓ}` ratio |

It reuses the same machinery as probing:
- the **same CFG sampler / parse annotations** (`CFGSample`: `ancestor_symbols`,
  `ancestor_indices`, `boundaries`),
- the **same "boundary position" insight** from R5 (the last token of a level-ℓ
  subtree stores that subtree's summary) — that is the position we read `f_k` at,
- the **same hidden-state hook pattern** from `evaluation/probing.py`, just
  registered on **all 12 blocks** instead of only the last.

So it slots directly into the existing pipeline and closes the
present / used / discarded triangle.

---

## 3. Files created

| File | Purpose |
|---|---|
| [cfg/synonym_swap.py](../cfg/synonym_swap.py) | Tree-based generator + the length-matched synonym-swap operator `P_ℓ`. Additive — does **not** touch `grammar.py`. |
| [evaluation/synonymic_invariance.py](../evaluation/synonymic_invariance.py) | All-layer hidden-state extraction + the `S_{k,ℓ}` metric (boundary & pooled readouts) + saving (`.npz`, `.txt`) and optional plots. |
| [tests/test_synonym_swap.py](../tests/test_synonym_swap.py) | Correctness tests for the swap operator. **All pass (exit 0).** |

### 3.1 `cfg/synonym_swap.py`

- `Node` / `build_tree(cfg, symbol, level)` — builds the parse as an explicit
  tree, **recording which rule (`rule_idx`) built each node**. The original
  `grammar.py` never exposes this, which is why a new generator was needed.
- `flatten(root)` — converts the tree back into a `CFGSample` with **exactly the
  same fields** `CFG.sample_string()` produces (string, `ancestor_symbols`,
  `ancestor_indices`, `boundaries`, `deepest_boundary`). The rest of the pipeline
  is therefore unchanged.
- `synonym_swap(cfg, level, require_length_match=True)` — samples a base string
  `x`, picks one level-ℓ node, re-expands it from a **different rule** of the same
  parent symbol (a synonym), and returns a `SwapResult` with both strings, the
  parent symbol, and the boundary index `j`.

### 3.2 The one real subtlety: length matching

Our rules have length **2 or 3**, so a swapped subtree can yield a different
number of terminals, which would shift every later position and move the boundary
index — contaminating the comparison with a pure positional effect (the RHM never
hits this because its branching is constant). The fix is **rejection sampling**:
keep re-expanding until the rebuilt subtree has the **same terminal length**.
Then the prefix up to the boundary is byte-identical and, because the transformer
is **causal**, any change in `h_k^{(j)}` is caused purely by the swap.

**Non-obvious grammar fact discovered while testing:** in **cfg3b, level 6**
(symbols 7/8/9) the two rules of *every* NT6 symbol differ in length
(e.g. `7→(3,1)` is length 2, `7→(1,2,3)` is length 3). So an exact length-matched
swap at ℓ=6 is **impossible**. The code handles this with
`require_length_match=False`: it still tries for a match first, and only if none
exists falls back to a **best-effort** swap where each string reads **its own**
boundary index. Levels 2–5 produce clean length-matched swaps; level 6 is
flagged as best-effort in the output. This is worth one sentence in the report.

### 3.3 `evaluation/synonymic_invariance.py`

- `extract_all_layers_at(...)` — one batched forward pass with a hook on the
  embedding (layer 0) and on **every block** (layers 1–12); returns the hidden
  state at the boundary position and the mean-pooled hidden state, per layer.
- `compute_S_for_level(...)` — builds the metric:
  - **numerator** = mean ‖f_k(x) − f_k(P_ℓ x)‖² over swap pairs,
  - **denominator** = mean ‖f_k(x) − f_k(y)‖² over **unrelated** strings (a
    shuffled pairing of the same pool) — the natural scale,
  - `S_{k,ℓ} = numerator / denominator`.
  No explicit per-layer normalisation is needed: numerator and denominator share
  layer `k`, so any activation-scale factor cancels in the ratio.
  Two readouts: **boundary** (primary, §5a of the design doc) and **pooled**
  (secondary sanity, §5b).
- `run(...)` — loops over levels, prints `S` per layer, and saves results.

### 3.4 Outputs (written to `results/`)

- `synonymic_invariance_trained.npz` — raw `S` arrays (`S_boundary`, `S_pooled`,
  shape `(13 layers) × (n levels)`).
- `synonymic_invariance_trained.txt` — human-readable `S_{k,ℓ}` table (always
  written, no dependencies).
- `synonymic_invariance_heatmap_trained.png` + `..._curves_trained.png` — only if
  `matplotlib` is installed (it is **not** installed on this machine; the `.npz`
  /`.txt` are always produced so figures can be regenerated later).

---

## 4. How to run it (for the teammate with the checkpoint + GPU)

```bash
# 0. (optional) validate the swap operator — CPU, no checkpoint needed
python tests/test_synonym_swap.py

# 1. quick smoke test (~1 min): few pairs, confirms the pipeline end-to-end
python evaluation/synonymic_invariance.py \
    --checkpoint gpt_checkpoint_step_6500.pt \
    --cfg cfg/grammars/cfg3b.txt \
    --levels 2 3 4 5 6 --n_pairs 50 --device cuda

# 2. full run — the headline S_{k,ℓ} heatmap
python evaluation/synonymic_invariance.py \
    --checkpoint gpt_checkpoint_step_6500.pt \
    --cfg cfg/grammars/cfg3b.txt \
    --levels 2 3 4 5 6 --n_pairs 1000 --device cuda

# 3. control — random weights should give a flat S ≈ 1 with no staircase
python evaluation/synonymic_invariance.py --random_gpt --device cuda
```

If `matplotlib` is wanted for the figures: `pip install matplotlib` (otherwise the
`.npz`/`.txt` still contain everything).

**Note:** the checkpoint `gpt_checkpoint_step_6500.pt` is gitignored and is **not**
present on this machine — it must be supplied by the teammate (same file the
patching extension uses).

---

## 5. How long it takes

Extension A does **no training** — that is the whole point, and the reason it is
far cheaper than the original probing (which trained 2 probes × 5 levels × 30k
iters = hours). The cost is two parts:

1. **Pair generation** (CPU sampler, rejection-sampled length matching): a few
   seconds to ~1–2 min per level for `n_pairs=1000`. Levels 2–5 length-match
   readily; level 6 is best-effort (fast).
2. **Forward passes** (GPU): for `n_pairs=1000` × 5 levels there are ~10k forward
   passes (base + swap), batched — on a GPU this is **well under a minute**.

**Estimated wall-clock on the teammate's GPU machine:**

| Config | Time |
|---|---|
| Smoke test (`--n_pairs 50`, 5 levels) | **< 1 min** |
| Full run (`--n_pairs 1000`, 5 levels) | **~5–10 min** |
| Same on CPU (no GPU) | ~30–60 min (forward passes dominate) |

Transient GPU memory is small (~0.4 GB: storing 13 layers × batch 32 × seq ~300 ×
768 during one forward pass).

---

## 6. What to expect in the results

- `S_{k,ℓ}` **decreases with depth `k`** — deeper layers more invariant.
- A **staircase**: the depth at which `S` drops moves *later* as ℓ goes 6 → 2
  (low levels invariant early, high levels invariant late) — our analogue of the
  RHM's `k ≈ ℓ+1` law.
- **Honest caveat:** our GPT is under-trained (6 500 steps). The RHM shows `S`
  collapses *as the task is learned*, so we likely see **partial** invariance — a
  softened staircase, not a sharp one. That is itself RHM-consistent (invariance
  tracks learning); we test whether the *trend* emerges, not whether it saturates.

---

## 7. Verification status

- `tests/test_synonym_swap.py` — **run, all passed (exit 0)** on this machine
  (CPU, no checkpoint needed): flatten reproduces `CFGSample`; levels 2–5 produce
  clean, local, grammatical length-matched swaps; level 6 confirmed best-effort.
- `evaluation/synonymic_invariance.py` — **not executed here** (no checkpoint / no
  GPU on this machine, per request). It reuses the validated swap operator and the
  same hook pattern already proven in `probing.py`/`patching.py`.
