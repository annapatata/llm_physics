# EXTENSION.md — Activation Patching of Boundary Hidden States

Extension to the Allen-Zhu & Li (2023) reproduction. Designed as a follow-up to Results 4-9: tests whether the boundary-hidden-state encoding of NT identity is **causally** load-bearing for structured generation, not just a correlate of it.

## 1. Hypothesis

Results 5-9 established that for trained GPT_rot:
- The NT identity of the level-ℓ subtree rooted at terminal i is linearly readable from the hidden state at i when `boundaries[ℓ][i] == 1` (Result 5).
- Attention preferentially routes from boundaries to nearest-previous same-level boundaries (Results 7-9), matching the data flow of CYK / inside DP.

**Causal claim being tested:** the model's structured generation depends specifically on the *content* of those boundary hidden states. If we corrupt them surgically, the model should produce strings that look locally normal (correct unigram / short n-gram statistics) but break at the structural level (CYK-invalid, broken nesting).

If the dissociation holds, we have causal — not merely correlational — evidence for the DP mechanism. If structural and surface metrics fall together, the boundary representation is incidental and the network achieves grammaticality some other way.

## 2. The intervention

Standard activation patching. Three full-sequence forward passes per trial:

1. **Clean run.** Feed valid CFG string `x_clean` (shape `(1, L+2)` with BOS/EOS) through the model in one pass. Cache the residual stream at every layer via forward hooks → `H_clean[ℓ]` of shape `(L+2, 768)`.
2. **Donor run.** Same with an independently sampled `x_donor`. Cache `H_donor[ℓ]`.
3. **Patched run.** Feed `x_clean` again. At target layer ℓ\*, install a hook that **overwrites** the layer's output: at positions i where `x_clean.boundaries[ℓ][i] == 1`, replace `H[ℓ\*][0, i, :]` with `H_donor[ℓ\*][0, i, :]`. Let the rest of the forward pass propagate. Read final-layer logits, or take the patched final-layer state at position L and continue with `generate_autoregressive` from `evaluation/evaluation.py`.

Same input-length and full-parallel forward-pass shape across all three runs — no token-by-token feeding. This is the same mechanism `probing.py:_make_hook` (lines 41-47) uses for reading, with a return value so PyTorch substitutes it back into the graph.

## 3. The critical dissociation: surface vs. structure

The whole experiment lives or dies on this metric pair. Without a *separation* between surface and structure, "accuracy dropped" is uninterpretable.

**Structural metrics (should break):**
- **CYK validity** of the patched-generation completion (`dp/cyk.py:is_valid`). Primary metric.
- **KL(P_model_patched ‖ P_CFG) at boundary positions and inside long subtrees**, computed with `dp/inside.py`.
- **Nesting integrity**: re-parse the generated string with CYK; count how many level-ℓ subtrees it contains vs. expected distribution.

**Surface metrics (should stay close to baseline):**
- **Unigram KL** between generated terminals and CFG's true terminal marginals (estimate marginals by sampling ~10K CFG strings once).
- **Bigram KL** under a CFG-marginal bigram model fit on sampled CFG strings.
- **Per-token entropy** averaged over generation — patching shouldn't make the model wildly uncertain about local choices.

The headline plot: for each intervention type, two bars — % drop in CYK validity vs. % drop in (1 − unigram-KL-normalized). Boundary-patch should show a large gap; controls should show similar drops on both.

## 4. Control suite

Without controls, this is not a real test.

| Control | Manipulation | Predicted outcome under hypothesis |
|---|---|---|
| **Random-position patch** | Same layer, same donor, same # positions, but patch *non-boundary* positions chosen at random | Smaller structural drop than boundary patch — ideally comparable to no-patch baseline |
| **Random-noise patch** | Replace boundary states with Gaussian noise of matched per-position norm | Isolates "missing info" vs "wrong info" — if noise hurts as much as donor, the model just needs *anything coherent*; if donor hurts more, it specifically follows the wrong DP state |
| **Early-layer patch** | Same boundary positions but at a layer *before* Result 5 accuracy ramps up (likely layer 0-3) | Smaller effect — boundary info isn't there yet |
| **Late-layer patch** | Same boundary positions at the layer where Result 5 accuracy peaks (TBD from our probing run, paper suggests mid-to-late) | Largest effect |
| **Level sweep** | Patch only level-2 boundaries vs. only level-6 boundaries (separate runs) | Given our checkpoint only encodes level-6 (see §7): NT2 patch ≈ no-op (no representation to corrupt), NT6 patch breaks structure. Sharper dissociation than the original "different breakage profiles" prediction |
| **GPT_rand control** | Same intervention on randomly initialized model | No specific effect of boundary positions — flat across position types |

## 5. What infrastructure already exists

- `cfg/grammar.py` → `CFGSample.boundaries[ℓ]`, `deepest_boundary`, `ancestor_indices` — all the position labels we need.
- `models/gpt_rot.py` → exposes `model.blocks[ℓ]`; supports `return_all_attentions`. Forward hooks attach cleanly per `probing.py:_make_hook`.
- `dp/cyk.py:is_valid` → structural oracle. ~0.3s per length-280 string; budget accordingly.
- `dp/inside.py` → for KL computation under the true CFG.
- `evaluation/evaluation.py:generate_autoregressive` → reusable for continuing generation from a patched state.
- `evaluation/probing.py` → already extracts hidden states in batches with hooks; copy that pattern.
- Trained checkpoint paths: `gpt_checkpoint_step_6500.pt` (also `/kaggle/input/datasets/periclesalexiou/model4k/gpt_weights_6500.pt` for Kaggle).

## 6. What needs to be written

New file: `evaluation/patching.py`.

Components:

1. **Multi-layer hook manager.** Register forward hooks on all 12 blocks. Two modes:
   - *Read mode*: cache outputs into a dict keyed by layer index.
   - *Write mode*: at a chosen layer, *return* a modified tensor that PyTorch will substitute back. Must handle that `GPTBlock.forward` returns either `tensor` or `(tensor, attn_weights)` (see `probing.py:_make_hook` for the unpacking trick).
2. **`run_with_patch(model, x_clean_tokens, donor_states, layer_star, positions_to_patch)`** → returns final-layer logits and the patched residual stream at the last position (for downstream generation).
3. **Metric helpers:**
   - `surface_metrics(generated_tokens, cfg)` → unigram-KL, bigram-KL, mean entropy.
   - `structural_metrics(generated_tokens, cfg)` → CYK-validity bool, optional inside-prob NLL.
4. **Experiment driver** that loops over N trials, applies each intervention condition, accumulates the metric pair, and prints the surface-vs-structure dissociation table.

Estimated size: ~300-400 LOC including controls.

## 7. Decisions (locked)

These were the open design choices flagged before coding; here are the chosen values with rationale.

- **Grammar: cfg3b.** Our checkpoint is trained on cfg3b (the paper's sharpest results are on cfg3f, but we work with what we have).

- **Target level: NT6 only.** Our Result 5 numbers (diagonal probe at boundaries) on the 6500-step checkpoint:

  | Level | Diag@boundary |
  |---|---|
  | NT2 | 39.2% |
  | NT3 | 47.6% |
  | NT4 | 33.7% (chance) |
  | NT5 | 42.5% |
  | NT6 | 81.9% |

  Chance is ~33% (3 classes). Only NT6 shows a strong boundary representation in our undertrained checkpoint (6500 steps vs. the paper's 100K). A causal test only makes sense for a representation that's actually formed — patching at NT4 would be patching noise. This also *sharpens* the level-sweep control (§4): under our hypothesis an NT2 patch should be a near-no-op, while an NT6 patch should break structural generation.

- **ℓ\*: chosen by per-layer diagonal-probe scan at NT6.** Scan all 12 blocks, ~5K iters per layer (vs. the full 30K). We only need the *shape* of the per-layer accuracy curve to find the peak — not publication-quality probes. The current `probing.py` only hooks the last block, so this is a small extension.

- **Donor: boundary-matched.** A plain same-length donor isn't guaranteed to have a level-6 boundary at the patched positions. If it doesn't, we're *erasing* the boundary representation rather than *swapping* it — that's closer in spirit to the noise control and muddies the causal claim. Boundary-matched donors (rejection-sample so the donor also has a level-6 boundary at every patched position) isolate the cleanest swap: one valid NT identity replaced by a different valid one.

- **Measurement: both per-position KL and CYK on continuation.** KL is fast, deterministic, and gives a per-position sensitivity signal. CYK on the autoregressive continuation is the headline structural-vs-surface dissociation. The two answer different questions; we report both.

- **N trials: 500 per condition.** Each trial is 2–3 forward passes + 1 CYK check (~0.3s); 500 is cheap and gives tight error bars.

## 8. Suggested execution order

1. Extend probing to scan layers at NT6 (the only level with a strong representation in our checkpoint — see §7). Pick ℓ\* from the per-layer accuracy peak.
2. Write the hook manager + `run_with_patch`. Test it does nothing when no patch is requested (identity check) and does *something* when a random patch is applied.
3. Implement metric helpers. Sanity-check on clean generations that surface KLs are near zero and CYK validity matches `evaluation.py`'s baseline.
4. Run main condition (boundary patch at ℓ\*) + random-position control. If no dissociation here, the rest is moot — stop and reconsider.
5. Run remaining controls (noise patch, early-layer, GPT_rand).
6. Run level sweep.
7. Plot: bar chart of structural drop vs. surface drop, per condition.

## 9. What success looks like

A figure showing: under boundary patching at the peak layer, CYK validity drops by ≥30 percentage points while unigram KL changes by <0.05 nats. Under random-position patching with matched count, both drops are within 5 pp / 0.05 nats of each other. Under GPT_rand, position type has no effect.

That's the entire result. One figure, one table of controls, one sentence: "corrupting boundary hidden states selectively breaks structural generation, providing causal evidence for the DP mechanism inferred from Results 5-9."

## 10. References to relevant code

- Hidden-state extraction pattern: `evaluation/probing.py:41-78`
- Generation loop: `evaluation/evaluation.py:23-53`
- Boundary labels: `cfg/grammar.py` (`CFGSample.boundaries`, `deepest_boundary`, `ancestor_indices`)
- Structural oracle: `dp/cyk.py:is_valid`
- True-distribution probability: `dp/inside.py`
- Model block list: `models/gpt_rot.py` (`model.blocks[ℓ]`)
- CONTEXT.md has the paper-section ↔ implementation map; consult before any design call.
