# EXTENSION.md — Activation Patching of Boundary Hidden States

Extension to the Allen-Zhu & Li (2023) reproduction. Follow-up to Results 4-9:
tests whether the boundary-hidden-state encoding of NT identity is **causally**
load-bearing for structured generation, not just a correlate of it.

## 1. Hypothesis

Result 5 showed that for trained GPT_rot, NT identity at level ℓ is linearly
readable from the hidden state at positions where `boundaries[ℓ][i] == 1`.
Probing shows information is *present*. It doesn't show information is *used*.

**Causal claim being tested:** if we surgically swap those boundary hidden
states for a donor's, the model's structural generation breaks (CYK-invalid)
while its surface statistics (terminal-token marginals) stay close to baseline.
A *dissociation* between structural and surface metrics is what makes this a
use-test, not just "we perturbed the model and it got worse."

## 2. The intervention

Standard activation patching. Per trial:

1. Sample `x_clean` (valid CFG string). Take prefix = `[BOS] + x_clean[:50]`.
2. Sample a *boundary-matched donor* — another valid CFG string that shares
   **≥ 3 level-ℓ boundary positions with `x_clean` at the same absolute
   indices** within the prefix, and carries a **different valid NT-ℓ identity**
   at each shared position. Patch *exactly* those shared positions. The donor is
   drawn from the natural CFG length distribution, not length-matched (see §5).
3. Cache the donor's hidden state at layer ℓ\* at those shared positions.
4. Run autoregressive generation from the prefix, but with a forward hook on
   layer ℓ\* that — on every forward pass — replaces the residual stream at
   the shared level-ℓ boundary positions of the prefix with the donor's values.
   The continuation tokens are never directly patched; the effect reaches them
   only through attention back onto the patched prefix positions.
5. Score the completion (structural: CYK validity; surface: bigram KL vs.
   the CFG's true bigram transition distribution).

This is the same hook mechanism `probing.py:_make_hook` uses for reading, with
a `return` so PyTorch substitutes the modified tensor back into the graph.

### What "patching level ℓ" means

We don't surgically replace "level-ℓ bits" — there's one 768-d vector per
token, with no level slots. Level-ℓ specificity comes from *position selection*:

- We patch at positions where `boundaries[ℓ][i] == 1` (right edges of level-ℓ
  subtrees).
- The donor must also have level-ℓ boundaries at those positions, so we swap
  one valid level-ℓ NT identity for another (not erase it).

Caveat we accept: those positions also encode higher-level NT info, which gets
swapped too. Across N trials with random donors, that info varies randomly and
the systematic effect is the level-ℓ swap (which is enforced). A subspace
projection swap (only the probe direction) would be the surgical version —
flagged here as future work.

## 3. The critical dissociation: surface vs. structure

Without this dissociation, a "validity dropped" number is uninterpretable.

- **Structural (should break under boundary patch):** CYK validity of the
  generated completion (`dp/cyk.py:is_valid`).
- **Surface (should stay close to baseline):** bigram KL between the generated
  terminals' transition distribution and the **CFG's true bigram transition
  distribution** `P(next | prev)`. Optional: trigram KL (secondary), mean
  per-token entropy.

We reference the surface metric to the *grammar's* true distribution, not to the
clean-condition output, so that the structural and surface axes are symmetric:
both "did the structure break" and "did the surface drift" are measured against
the ground truth the model was trained to match.

Headline plot: for each condition, two bars — CYK validity drop vs. bigram-KL
drift. Boundary patch should show a large gap; the control should show similar
movement on both bars (or no movement on either).

## 4. The two controls

| Control | Manipulation | Prediction | What it establishes |
|---|---|---|---|
| **Noise @ boundary** | Same layer, same boundary positions, replace with Gaussian noise (zero-mean, variance matched to the residual stream) instead of a donor | Both CYK validity and the surface metric drop together — noise is non-specific | The donor patch's structural-only collapse is about swapping NT *identity*, not generic perturbation of those positions |
| **Noise @ non-boundary** | Same layer, same *number* of positions, Gaussian noise at non-boundary positions chosen at random | CYK validity stays high; surface metric may move slightly | If structure survives when everything *except* boundaries is corrupted, the boundary positions are *sufficient* to carry the structural load |

The two controls bracket the claim from both sides:

- **Noise @ boundary (specificity):** noise at boundaries is a generic
  perturbation and breaks everything; a donor patch at boundaries is a
  structured swap and breaks only structure. The gap between these two is the
  core causal evidence.
- **Noise @ non-boundary (sufficiency):** if structure survives noise
  everywhere but the boundaries, then the boundaries alone are doing the
  structural work — the model is reconstructing the parse from boundary
  representations.

The original draft's "donor patch at random positions" tested spatial
specificity but conflated identity-swap with perturbation. The two noise
controls replace it cleanly.

## 5. Decisions (locked)

- **Grammar: cfg3b.** Our checkpoint is trained on cfg3b.

- **Target level: NT5.** Diagonal probe @ boundary on the 6500-step checkpoint:

  | Level | Diag@boundary |
  |---|---|
  | NT2 | 87.1% |
  | NT3 | 66.3% |
  | NT4 | 89.5% |
  | NT5 | 99.9% |
  | NT6 | 91.9% |

  NT5 has the strongest representation, so it has the most signal for the
  causal test. (If results come in cleanly, a follow-up at NT3 — the weakest
  representation — would give a graded prediction. Not in scope here.)

  **Why not NT6, even though it isolates more cleanly.** Boundary nesting runs
  downward: a level-ℓ boundary is also a boundary at every *deeper* level
  ℓ+1…6. So at an NT5 boundary, NT6 *always* rides along in the same vector and
  is swapped every trial — the one confound pairs-averaging cannot remove (only
  subspace projection can; future work). NT6, being the deepest NT level (level
  7 = terminals), has *nothing* nested below it, so with pairs it is isolated
  with no un-averageable confound. That makes NT6 the cleaner target on the
  *isolation* axis — but the worse target on the *dissociation* axis, which
  matters more:

  - A level-6 NT expands directly to ~2–3 terminals, so NT6 essentially *is* the
    local surface (bigram) structure. Corrupting it moves the **surface metric
    too**, collapsing the structure-vs-surface dissociation that makes this a
    *use*-test rather than "we broke the model."
  - NT6 boundaries are dense (~every 2–3 tokens, ~40% of positions), so patching
    them is a sledgehammer, not a surgical probe — surface stats drift from sheer
    volume of overwriting.
  - NT6's representation (91.9%) is weaker than NT5's (99.9%).

  NT5 is the sweet spot: deep enough to carry a near-perfect, *structural*
  representation, shallow enough that corrupting it breaks the *global* parse
  while leaving *local* bigram texture intact. Its only cost is the NT5/NT6
  confound, which pairs narrows the culprit to {NT5, NT6} and subspace
  projection would finish separating. NT6 (and NT3) are therefore optional
  *contrast* panels, not the headline: if NT5 drops CYK with bigram flat while
  NT6 drops CYK *and* moves bigram, that contrast is itself evidence for the
  level interpretation.

- **ℓ\*: last layer (layer 11).** Existing `probing.py` results already probe
  the last hidden layer; the NT5 99.9% figure comes from there. `probing_all.py`
  would refine this if an intermediate layer turned out to be stronger, but
  given 99.9% there is little room to improve. Running probing_all.py first
  is not required.

- **Donor: boundary-matched, per-position, different-NT.** A donor is accepted
  if it shares **≥ 3 level-ℓ boundary positions with `x_clean` at the same
  absolute indices** (within the prefix window) and carries a **different valid
  NT-ℓ** at each shared position. We patch exactly the shared positions, swapping
  a valid NT identity for a *different* valid one — not erasing it.

  Three constraints drive this:

  1. **Same absolute position.** The model uses RoPE, so a hidden state carries
     relative-position information; injecting a donor's position-`j` state at
     clean's position `i ≠ j` adds a positional confound. So we require the donor
     to have a level-ℓ boundary at the *exact* index `i` we patch, which also
     forces `len(donor) > i`.
  2. **Different NT-ℓ.** If the donor carried the *same* NT-ℓ as clean, the patch
     would be a near no-op. The swap must change identity to test causal use.
  3. **Natural length distribution.** We do *not* length-match donor to clean —
     only `len(donor) > i` is mechanically required. We draw donors from the
     natural CFG length distribution so that the higher-level NT info that
     unavoidably rides along in the patched vector (the contamination noted in
     §2) varies *randomly* across trials instead of systematically. Biasing
     toward short donors would make that contamination systematic and weaken the
     control.

  **Sampling strategy A (default — single coherent donor):** reject-sample
  candidate strings; for each, intersect its level-ℓ boundary mask with
  `x_clean`'s over the prefix, keep positions where the NT differs, and accept if
  ≥ 3 such positions exist. This injects one *internally consistent* alternative
  parse, which is the cleaner manipulation. Cost: requiring ≥ 3 same-index
  coincidences with NT-difference rejects many candidates, so we sample a pool
  and keep qualifiers.

  **Sampling strategy B (fallback — per-position pool):** if strategy A's
  rejection rate is impractical, decouple the matching. Pre-sample a pool of
  valid strings and record each one's level-ℓ boundary positions and NT
  identities. Then, for *each* patched position `i` independently, pull a donor
  from the pool that has a level-ℓ boundary at exactly `i` with an NT different
  from clean's, and take *that* donor's hidden state at `i`. Different patched
  positions may draw from different donor strings. This trades the
  internally-consistent single parse (strategy A) for a much easier matching
  problem — each position only needs *one* coincidence instead of all of them
  simultaneously — at the cost that the injected boundary states no longer come
  from one coherent tree. Start with A; switch to B only if sampling A is too
  slow.

- **Measurement: CYK on continuation + bigram KL.** Bigram KL between the
  generated terminals' transition distribution and the **CFG's true bigram
  distribution** `P(next | prev)` (estimated once from a large sample of valid
  strings) is used instead of unigram KL. Referencing the grammar's true
  distribution — not the clean condition's output — keeps the structural and
  surface axes symmetric (both measured against ground truth).

  *Why not unigram:* measured on cfg3b (300 sampled strings), the terminal
  marginal is essentially uniform — `{1: 0.334, 2: 0.333, 3: 0.333}`. Unigram
  KL is therefore trivially ~0 for almost any output, including garbage, so it
  can't dissociate anything.

  *Why bigram works:* the bigram conditional `P(next | prev)` is strongly
  structured (a token rarely follows itself):

  | prev \ next | 1 | 2 | 3 |
  |---|---|---|---|
  | **1** | 0.102 | 0.476 | 0.422 |
  | **2** | 0.486 | 0.089 | 0.425 |
  | **3** | 0.417 | 0.435 | 0.148 |

  Mutual information I(prev; next) ≈ 0.135 nats — far from the 0 of independent
  terminals. So bigram KL *can* move, which makes "bigram KL stays near
  baseline" a falsifiable surface claim rather than a trivially-true one.

  *Why not trigram as the headline:* the surface metric must stay agnostic to
  tree structure (that's the dissociation). Higher n-gram orders capture
  longer-range correlations that increasingly *are* the structure CYK should
  own. In cfg3b the lowest-level constituents are tiny — a level-6 NT expands
  to only ~2–3 terminals — so a 3-token window can straddle an entire low-level
  subtree, making trigram KL partly a *structural* check. Bigram windows mostly
  sit inside constituents and capture local "what follows what" texture (the
  "right word by word" notion). Trigram also costs variance: 27 cells vs 9, and
  per-trial continuations are short. Bigram is therefore the headline.

  *Trigram as a secondary check:* we still report trigram KL as a robustness
  sanity check, with the explicit caveat that it partly reflects low-level
  constituent validity and so is expected to move somewhat more than bigram
  even under a clean structural dissociation.

  Both metrics are computed per trial; we report rates/means across N trials.

- **N trials: 500 per condition.** Each trial is 2–3 forward passes + 1 CYK
  check (~0.3s); 500 fits in well under an hour.

## 6. Suggested execution order

1. Identity-check the hook in `extension/patching.py` (no-op patch must
   produce logits identical to vanilla forward).
2. Build the corpus once: `extension/build_corpus.py` (samples M valid strings,
   reports the donor acceptance rate).
3. Freeze the trials: `extension/build_trials.py` (matches N clean/donor pairs).
4. Run the sweep at (NT5, layer −1): `extension/patch_experiment.py` — clean
   baseline, donor patch, noise@boundary, noise@non-boundary.
5. Plot: bar chart of CYK validity vs. bigram KL per condition.

## 7. What success looks like

Under boundary patching at ℓ\*, CYK validity drops substantially (target: ≥30
percentage points below clean baseline) while bigram KL stays close to
baseline. Under noise patching at the same boundary positions, both CYK
validity and bigram KL drop together (noise is non-specific). The contrast
between the two conditions is the result.

That's the result. One figure, one control, one sentence: "corrupting boundary
hidden states selectively breaks structural generation, providing causal
evidence that the NT-boundary encoding identified by Result 5 is used by the
model."

## 8. References to relevant code

The extension lives in `extension/` (see `extension/README.md`):

- Patch hook + identity check: `extension/patching.py`
- Corpus builder + donor matching: `extension/build_corpus.py`
- Trial freezer: `extension/build_trials.py`
- Experiment driver + metrics: `extension/patch_experiment.py`

Reused from the main reproduction:

- Per-layer probe scan (for picking ℓ\*): `evaluation/probing_all.py`
- Hidden-state extraction pattern: `evaluation/probing.py:41-78`
- Generation loop: `evaluation/evaluation.py:23-53`
- Boundary labels: `cfg/grammar.py` (`CFGSample.boundaries`)
- Structural oracle: `dp/cyk.py:is_valid`
- Model block list: `models/gpt_rot.py` (`model.blocks[ℓ]`)
