# Results — Activation Patching of NT5-Boundary Hidden States

**Question.** Probing (Result 5) shows NT identity is *decodable* from the hidden
state at NT-boundary positions. Decodable ≠ used. This extension asks the causal
question: if we **overwrite** those boundary states during generation, does
structured generation break? If yes, the encoding is causally read by the model,
not an inert correlate.

**Setup.** GPT2Rotary, cfg3b, checkpoint at step 6500. Patch target = **NT5**, at
**layer 5** (the earliest layer where NT5 is strongly decodable — see "Why layer
5"). N = 500 paired trials, temperature-1.0 multinomial generation. Each trial is
run under four conditions on the *same* seed so the comparison is paired.

---

## The four conditions

| Condition | What we inject at the NT5-boundary positions | Role |
|---|---|---|
| `clean` | nothing (baseline) | reference |
| `donor` | another valid string's hidden states — **same positions, different NT5 identity** | the causal test |
| `noise_boundary` | Gaussian noise of matched magnitude, **same positions** | specificity control |
| `noise_nonboundary` | the same noise at an equal number of **non-boundary** positions | sufficiency control |

The donor is a "plausible lie": an in-distribution state encoding a *different but
valid* parse, which the model trusts and propagates. Noise is "static":
out-of-distribution, partly filtered out by LayerNorm/softmax.

---

## Headline result (CYK)

CYK validity = is the generated completion a member of the grammar (yes/no per
string); the reported number is the rate over the 500 trials. "CYK drop" =
`clean_rate − condition_rate`.

| condition | CYK valid | CYK drop |
|---|---|---|
| clean | 98.8% | — |
| **donor** | **41.2%** | **57.6%** |
| **noise_boundary** | **69.8%** | **29.0%** |
| **noise_nonboundary** | **94.0%** | **4.8%** |

The full structural ordering holds with large margins:

**`donor` ≫ `noise_boundary` ≫ `noise_nonboundary`**

- **Specificity** (`donor` ≫ `noise_boundary`): 57.6% vs 29.0%, a **28.6-point
  gap**. At N = 500 each rate has SE ≈ 2.1%, so the gap is ≈ **9–10 SE** —
  decisive (it was only ~2.3 SE at the N = 50 pilot). A *targeted, legible* NT5
  swap breaks grammar about twice as much as equal-magnitude noise at the **same
  positions**. So the effect is driven by the *identity/legibility* of what we
  inject, not by the mere perturbation.
- **Sufficiency** (`noise_boundary` ≫ `noise_nonboundary`): 29.0% vs 4.8%. The
  same noise breaks structure ~6× more at the boundaries than away from them —
  the structural information is concentrated **at the boundaries**.

**Conclusion: the NT5-boundary encoding is causally *used* for structured
generation, not merely decodable from it.**

---

## Surface metric (n-grams) — a double dissociation

Second metric: KL divergence between each condition's terminal n-gram
distribution and the true grammar's, for orders 2, 3, 4 (bigram → 4-gram). This
captures *local* surface statistics, complementary to CYK's *global* check.

| condition | 2-gram | 3-gram | 4-gram | KL drift vs clean (3-gram / 4-gram) |
|---|---|---|---|---|
| clean | 0.0000 | 0.0000 | 0.0001 | — |
| donor | 0.0001 | 0.0001 | 0.0001 | +0.0000 / +0.0000 |
| noise_boundary | 0.0000 | 0.0000 | 0.0001 | +0.0000 / +0.0000 |
| noise_nonboundary | 0.0000 | 0.0013 | 0.0019 | +0.0013 / +0.0018 |

We originally predicted a **surface→structure gradient**: the donor's KL drift
should *grow* with n-gram order. That prediction was **wrong** — but what
replaced it is cleaner: a **double dissociation** between the structural metric
(CYK) and the surface metric (n-grams).

- **`donor`: structure breaks, surface intact.** It collapses CYK by 57.6% yet
  leaves *every* terminal statistic up to order 4 essentially perfect (drift
  ≈ 0). A donor encodes a different but *valid* parse, so the model keeps emitting
  locally well-formed terminals — but assembles them under an incoherent global
  structure. The break lives entirely **above** a 4-terminal window.
- **`noise_nonboundary`: surface wobbles, structure intact.** It is the *only*
  condition that moves the n-grams (+0.0013 / +0.0018) while barely touching CYK
  (4.8% drop). Noise mid-constituent nudges local emissions slightly off, but the
  boundaries — and therefore the global bracketing — stay intact, so CYK passes.

So the two manipulations land on **opposite axes**: donor hits global structure
and not local surface; non-boundary noise hits local surface and not global
structure. "Locally fine, globally broken" vs "locally wobbly, globally fine."

**Caveat.** All n-gram magnitudes are tiny (1e-3 to 1e-4, near the estimation
floor). The dissociation is best stated *qualitatively* — "the donor leaves all
measured surface statistics flat while devastating CYK" — and the quantitative
weight should rest on **CYK**. Do not over-read the 0.0018.

---

## Why these two controls answer the obvious objections

- *"Maybe any perturbation at those positions breaks grammar."* → `noise_boundary`
  perturbs the **same positions** with matched magnitude and breaks structure
  **half as much**. The effect needs a *legible, in-distribution* NT5 identity,
  not just a disturbance.
- *"Maybe perturbing anywhere in the prefix breaks grammar."* → `noise_nonboundary`
  perturbs equally many positions elsewhere and barely dents CYK (4.8%). The
  boundaries specifically carry the structural load.

---

## Why layer 5 (the patch site)

The probe reads the **last** block, but patching there is causally inert for
generation: after the last block only `ln_f` + `lm_head` remain, and they do not
mix positions, so a corrupted *prefix* state can never reach the token being
generated. The patch must sit at an **intermediate** layer with blocks still
above it, so attention can carry the corruption forward.

Two competing pressures fix the choice:

- **Too early** → NT5 not yet crystallized, so noise damages as much as the
  donor (poor specificity).
- **Too late** → few blocks above, so the corruption can't propagate (weak
  effect).

The sweet spot is the **earliest layer where NT5 is already strongly decodable**,
which maximizes the number of blocks above it. The per-layer diagonal NT5 probe
(`evaluation/probing_all.py`) gives, by layer:

```
0: 38.1   1: 26.3   2: 51.1   3: 85.2   4: 75.0(dip)
5: 98.6   6: 98.3   7: 99.7   8: 99.9(peak)   ...   11: 99.7
```

Layer 5 is the first plateau (98.6%). The N = 50 head-to-head confirmed it: layer
5 gave a larger donor effect than layer 6 at the same specificity gap, and both
beat the saturated-but-shallow layer 8 (peak decodability, but too few blocks
above to propagate). Layer 5 = earliest-strong = maximal propagation distance.

---

## Known limitation: NT6 co-location

The pairs design averages out contamination from levels 2–4 (their identities at
an NT5 boundary are independent across random donors → mean-zero), **but not
NT6**. An NT5 left-boundary is *nested co-located* with an NT6 left-boundary
(the start of an NT5 constituent is the start of its first NT6 child), so every
donor swap moves the co-located NT6 identity too, **systematically** — it cannot
be averaged away.

Why this does not threaten the claim:

1. Whether the causal variable is "NT5" or "the NT6 nested inside it," it is
   unambiguously a **high-level structural** encoding — the structure-vs-surface
   dissociation stands either way. The imprecision is *which deep level*, not
   *structure vs surface*.
2. We chose NT5 over NT6 **deliberately**. NT6 is cleanly isolable (nothing
   nested below it) but spans only 2–3 terminals, so corrupting it would move the
   *surface* statistics and collapse the dissociation. You can have clean
   isolation (NT6) or a clean structure/surface dissociation (NT5) — not both
   with this method. We picked the dissociation.
3. **Clean fix (future work):** patch only the projection of the residual onto
   the NT5 probe's decodable direction, leaving the NT6 subspace untouched.
4. **Cheap check on existing data:** stratify the 500 donor trials by whether NT6
   also flipped vs stayed the same, and compare the CYK drop. If the drop is
   similar in both subsets, NT5 — not the co-located NT6 — is doing the work.

---

## Summary

- The NT5-boundary hidden-state encoding of NT identity is **causally used** for
  grammatical generation (donor 57.6% CYK drop, ≈ 9–10 SE above the noise
  control).
- The effect is **specific** (beats same-position noise 2×) and **localized to
  the boundaries** (beats non-boundary noise ~6×).
- CYK and n-gram surface statistics form a **double dissociation**: the donor
  breaks global structure while leaving local surface intact; non-boundary noise
  does the reverse.
- Main open issue is NT6 co-location, addressable by an NT6 stratification check
  on the current data and, properly, by a subspace-projection patch (future
  work).
