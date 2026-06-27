# Extension A — Synonymic Invariance in the CFG Transformer

*An analytical walkthrough of the proposed second extension, connecting our Allen‑Zhu
reproduction to the **Random Hierarchy Model** (RHM) paper
(Cagnetta, Petrini, Tomasini, Favero & Wyart, 2024).*

This document explains the idea slowly and from scratch, maps every concept onto code
we already have ([cfg/grammar.py](../cfg/grammar.py),
[evaluation/probing.py](../evaluation/probing.py)), and ends with a concrete
implementation plan. Read it top to bottom — each section builds on the previous one.

---

## 0. TL;DR (the one-paragraph version)

The RHM paper and the paper we reproduced study **the same object**: a layered random
grammar where every symbol can be expanded by several interchangeable rules. The RHM
calls those interchangeable rules **synonyms**, and its single most important
measurement is: *as you go deeper into a trained network, its internal representation
stops being able to tell synonyms apart* — it becomes **invariant** to swapping one
synonym for another. They quantify this with a number called **synonymic sensitivity**
`S_{k,ℓ}` and show that invariance to level‑ℓ synonyms switches on at network depth
`k ≈ ℓ+1`. **Extension A = measure exactly this `S_{k,ℓ}` curve on our already‑trained
GPT.** It needs no retraining, reuses our sampler and our hidden‑state hooks, and it
plugs a hole between our two existing results: probing showed the model *encodes* the
parse tree; synonymic invariance shows the model *throws away the irrelevant within‑tree
detail*.

---

## 1. Why the two papers are the same model wearing different clothes

Our reproduction (Allen‑Zhu & Li, *Physics of LMs, Part 1*) and the RHM are both
**L‑level layered random CFGs**:

| Concept | Allen‑Zhu CFG (us) | Random Hierarchy Model |
|---|---|---|
| Levels | `1` (root) … `7` (terminals) | `L+1` (class) … `1` (leaves) |
| A symbol expands into… | 2 or 3 children (a rule's RHS) | exactly `s` children (an `s`-tuple) |
| # of rules per symbol | variable (e.g. 4 rules for symbol 22) | exactly `m` rules ("synonyms") |
| What the network sees | flat terminal string | flat leaf string |
| **Task** | **next-token prediction** (language model) | **classify the root** |
| **Model** | **GPT-2 transformer** | deep CNN / MLP |
| Headline measurement | *Is the parse tree **encoded**?* (probing) | *Synonymic **invariance*** `S_{k,ℓ}` |

The only structural differences that matter for us:

1. **Branching is variable for us** (rules have length 2 *or* 3), constant (`s`) for them.
   This causes one technical wrinkle we handle in §6.
2. **They classify, we generate.** They have one representation vector per input; we have
   one hidden state *per position*. So "the representation" becomes "the hidden state at
   a chosen position" (§5).

Everything else — the hierarchy, the synonyms, the idea of invariance — carries over
directly.

---

## 2. What exactly is a "synonym"? (worked cfg3b example)

In cfg3b, look at the root symbol **22**. Our grammar file gives it four rules:

```
22 |-> 20 21
22 |-> 20 19 21
22 |-> 21 19 19
22 |-> 20 20
```

These four right-hand sides are the **synonyms of symbol 22**. They are *interchangeable*
in one precise sense: whichever one you pick, the node is still a "22". A 22 expanded as
`20 21` and a 22 expanded as `20 20` produce **different terminal strings underneath**,
but both are legitimate, grammatical ways of *being a 22*.

The RHM's parameter `m` is literally "how many synonyms each symbol has." In cfg3b `m`
is roughly 2–4 depending on the symbol.

> **Mental model.** Think of a non-terminal as a *meaning* and its rules as *different
> ways to say that meaning*. "I'm tired" / "I'm exhausted" / "I'm worn out" are synonyms:
> the surface words differ, the meaning (the parent label) is identical. The RHM asks
> whether a trained network, deep enough, learns to **collapse all three to the same
> internal representation** — i.e. to represent the *meaning*, not the *wording*.

---

## 3. The RHM claim, stated formally

The RHM defines **synonymic sensitivity** (their Eq. 8). For network layer `k` and
hierarchy level `ℓ`:

$$
S_{k,\ell} \;=\;
\frac{\big\langle\, \lVert f_k(x) - f_k(P_\ell\, x)\rVert^2 \,\big\rangle_{x,\,P_\ell}}
     {\big\langle\, \lVert f_k(x) - f_k(y)\rVert^2 \,\big\rangle_{x,\,y}}
$$

Read it piece by piece:

- `f_k(x)` — the network's representation at layer `k` for input `x`.
- `P_ℓ x` — the input `x` with **one level‑ℓ synonym swapped** for another (the *meaning*
  is unchanged, the *wording* of one sub-part is changed). `P_ℓ` is the "swap operator".
- **Numerator** — how much the layer‑`k` representation *moves* when we make a change
  that *shouldn't matter* (a synonym swap). Small ⇒ the layer ignores synonym identity ⇒
  invariant.
- **Denominator** — a normaliser: how much the representation moves between two
  *genuinely different* random inputs `x` and `y`. This sets the scale of "a lot".

So `S_{k,ℓ} ∈ [0, ~1]`:

- `S ≈ 1` → swapping a synonym moves the representation as much as a totally different
  input would. The layer is **fully sensitive** (it still "hears the wording").
- `S ≈ 0` → swapping a synonym barely moves the representation. The layer is
  **invariant** (it "hears only the meaning").

**Their key empirical law (Fig. 6):** invariance to level‑ℓ synonyms appears
*progressively with depth* — a given level becomes invariant starting around layer
`k = ℓ + 1`, and lower levels (closer to the leaves) become invariant in earlier layers
than higher levels. Invariance tracks generalisation: `S` drops exactly as the network
learns the task.

---

## 4. The level-numbering bridge (read carefully — easy to get backwards)

Their `ℓ` counts **up from the leaves** (`ℓ=1` = leaf tuples). Our levels count **down
from the root** (`ℓ=2` = top NTs, `ℓ=6` = NTs just above terminals). To avoid confusion
we **index a swap by the level of the parent symbol whose rule we change**, using *our*
numbering `ℓ ∈ {2,3,4,5,6}` — the same levels our probes already use.

Swapping the rule at a level‑ℓ parent changes its children at level ℓ+1 and everything
below. So:

| Our swap level ℓ | Symbol whose rule we swap | RHM equivalent | Predicted to become invariant in… |
|---|---|---|---|
| 6 | NT6 (just above terminals) | low level (`l=1`) | **shallow** transformer layers |
| 5 | NT5 | … | … |
| 4 | NT4 | … | … |
| 3 | NT3 | … | … |
| 2 | NT2 (top) | high level | **deep** transformer layers |

**Prediction translated to our setting:** invariance to *NT6* swaps should appear early
(low layers); invariance to *NT2* swaps should appear only in the deepest layers — a
clean monotone staircase across our 12 transformer blocks.

---

## 5. What is `f_k(x)` for a language model? (choosing the position)

The RHM has one representation per input (it classifies). We have one hidden state per
**position**. We need a principled choice of *which* position to read. Two options:

### 5a. Primary: the subtree's **boundary "summary" position** (recommended)

Our probing results (R4/R5) already established that the **last token of a level‑ℓ
subtree** — the position where `boundaries[ℓ][i] = 1` — is where the model **stores the
summary of that whole subtree**. That is precisely the place the RHM's claim is about:
the summary of a sub-constituent.

So define, for a subtree rooted at a level‑ℓ node with right-boundary position `j`:

$$
f_k(x) \;:=\; h_k^{(j)}(x) \quad\text{— the layer-}k\text{ hidden state at the subtree's boundary } j.
$$

We swap that one subtree's rule (a level‑ℓ synonym), keep everything else fixed, and ask:
*does the summary at `j` change?* If the model has learned to represent "a level‑ℓ
constituent of type `s`" and **not** "which particular rule built it", then `h_k^{(j)}`
should be (nearly) unchanged — `S_{k,ℓ} → 0` at deep `k`.

This is the cleanest, most interpretable version and it directly extends R5.

### 5b. Secondary: mean-pooled representation

To mirror the RHM's "whole-input representation" more literally, also report a version
where `f_k(x)` is the **mean over all positions** of `h_k`. Cheaper to reason about
globally, noisier locally. Good as a second panel / sanity check.

We lead with 5a and include 5b as corroboration.

---

## 6. The swap operator `P_ℓ` in our grammar (the one real subtlety)

We need to turn a sampled string `x` into `P_ℓ x`: same string, but one level‑ℓ subtree
rebuilt from a **different rule of the same parent symbol**.

### The procedure

1. Sample a base string `x` with its full parse (we already do this in
   `CFG.sample_string()` — it returns `ancestor_symbols`, `ancestor_indices`,
   `boundaries`).
2. Pick a target level `ℓ` and one node at that level — i.e. pick one value of the
   ancestor index `p_ℓ`. Its span is all positions `i` with `ancestor_indices[ℓ][i] ==`
   that value; its symbol is `s = ancestor_symbols[ℓ][i]`; its right boundary is the
   position `j` in the span with `boundaries[ℓ][j] == 1`.
3. Choose a **different rule** of symbol `s` (a different synonym).
4. Re-expand that node's subtree top-down with the new rule (random choices below),
   producing new terminals for that span only.
5. Splice the new terminals back in place of the old span → that's `P_ℓ x`.

### The subtlety: subtree length can change

Because our rules have length 2 **or** 3, a different rule (and its random descendants)
may yield a **different number of terminals**. If the swapped subtree has a different
length, every position *after* it shifts, and the boundary index `j` moves — contaminating
the comparison with a pure positional effect. (The RHM never hits this because its
branching `s` is constant — a genuine difference between the two settings, worth one
sentence in the report.)

**Clean fix — length-matched swaps.** Accept a swap only if the rebuilt subtree has
**the same terminal length** as the original. Then:

- the prefix before the subtree is byte-identical,
- the boundary position `j` is at the **same index** in `x` and `P_ℓ x`,
- and because the transformer is **causal**, `h_k^{(j)}` depends only on tokens `1..j`,
  so any change in `h_k^{(j)}` is caused **purely by the synonym swap** — nothing
  downstream can leak in.

Implement length-matching by **rejection sampling**: re-expand with a different top-rule,
keep regenerating until the subtree length matches; cap the attempts. This is cheap and
reliable for the **deeper levels** (ℓ = 4, 5, 6), whose subtrees are small (a handful of
terminals). For shallow levels (ℓ = 2, 3) subtrees are large (tens–hundreds of tokens)
and exact length-matching is rarer; for those, fall back to one of:

- restrict to parent symbols that *have* two equal-length rules and match only the
  immediate children length, accepting small downstream length changes and reading each
  string's *own* boundary index (note the caveat); or
- simply report fewer / noisier samples at ℓ=2,3 and say so.

Lead with length-matched deep levels; treat shallow levels as best-effort.

### Implementation note

`CFG._generate()` currently makes all rule choices internally and only returns
`level_syms` + `par_arrays`; it does **not** expose which rule built each node, and there
is no "re-expand one node" entry point. Extension A therefore needs a small new helper in
`cfg/` — easiest as a **tree-based generator**:

- build the parse as an explicit tree of nodes `(symbol, children)`,
- a function `resample_subtree(node, forbid_rule)` that re-expands one node with a forced
  different rule,
- a `flatten(tree)` that returns the terminal string **and** the same parse annotations
  `sample_string()` already produces (so the rest of the pipeline is unchanged).

This is additive — it does not touch the existing `sample_string()` path used by probing.

---

## 7. The metric, concretely, in our pipeline

For a fixed swap level `ℓ` and many sampled base strings:

1. Build pairs `(x, P_ℓ x)` (§6), all length-matched, recording the boundary index `j`.
2. One forward pass each through the **frozen** GPT, with a forward hook on **every**
   block `model.blocks[0..11]` (probing already hooks `blocks[-1]`; we just register on
   all of them) → hidden states `h_k^{(j)}(x)` and `h_k^{(j)}(P_ℓ x)` for `k = 1..12`
   (optionally `k=0` = embeddings).
3. Numerator: average `‖h_k^{(j)}(x) − h_k^{(j)}(P_ℓ x)‖²` over pairs.
4. Denominator: average `‖h_k^{(j)}(x) − h_k^{(j')}(y)‖²` over **unrelated** base strings
   `x, y` (boundary positions of the same level ℓ) — the natural scale.
5. `S_{k,ℓ} = numerator / denominator`.

Output: a `12 × 5` table / heatmap `S_{k,ℓ}` (layer × level). That single figure is the
deliverable.

> Tip: normalise hidden states per layer (e.g. divide by mean norm at that layer) before
> taking differences, so the ratio isn't dominated by raw activation-scale differences
> across depth. This mirrors the RHM's ratio definition, which already cancels scale.

---

## 8. What we expect to see (and the honest caveat)

If the RHM mechanism holds in a *language-model* transformer:

- **`S_{k,ℓ}` decreases with depth `k`** for every level — deeper layers are more
  invariant.
- **The staircase:** the depth at which `S` drops moves *later* as ℓ goes from 6 → 2
  (low levels invariant early, high levels invariant late) — the analogue of the RHM's
  `k ≈ ℓ+1` law.
- Invariance is **strongest where probing was strongest** (deep levels / boundary
  positions), tying the two results together.

**Caveat to state up front:** our GPT is **under-trained** (6 500 steps, KL still far from
the paper's target). The RHM shows `S` only collapses *as the task is learned*. So we may
see **partial** invariance — a softened staircase rather than a sharp one. That is itself
an RHM-consistent finding (invariance tracks learning), and it's the honest framing: we
test whether the *trend* emerges, not whether it saturates.

---

## 9. Why this is the right extension (how it closes the triangle)

We already have two results about the model's internals:

- **Probing (R4/R5):** the boundary hidden state **encodes** the parse tree — you can
  *read out* `s_ℓ` from it.
- **Activation patching ("present vs. used"):** which encoded information is actually
  **used** causally downstream.

Synonymic invariance is the third corner:

- **Probing** says *what is present.*
- **Invariance** says *what is discarded* — synonym identity is present in the input but
  **collapsed away** in the representation.
- **Patching** says *what is used.*

"Present", "discarded", and "used" are three faces of the same question about how the
model compresses a parse tree. Extension A is the missing face, it reuses the probing
infrastructure almost wholesale, and it imports the second paper's *central* measurable
rather than a peripheral one.

---

## 10. Implementation plan mapped to our files

| Step | Where | New / reuse |
|---|---|---|
| Tree-based generator + `resample_subtree` + length-matched `P_ℓ` swap | new `cfg/synonym_swap.py` | **new** (additive; leaves `grammar.py` untouched) |
| Correctness tests (swap preserves ancestors above ℓ, changes terminals below, length matches, parent symbol unchanged) | `tests/` | **new**, small |
| Hook **all** blocks for `h_k` (not just `blocks[-1]`) | adapt `extract_hidden_states_batch` in [evaluation/probing.py](../evaluation/probing.py) | **reuse + extend** |
| Compute `S_{k,ℓ}` (boundary version §5a + pooled §5b) | new `evaluation/synonymic_invariance.py` | **new**, ~150 lines |
| Plot `12×5` heatmap + per-level curves | mirror [evaluation/plot_attention.py](../evaluation/plot_attention.py) style | **new**, small |
| Write-up replacing the Lorem-ipsum §6 of the report | [report/report.tex](../report/report.tex) | **new** |

### Suggested split across the four of us (2–3 days)

1. **Sampler/swap** (`synonym_swap.py` + tests) — the only conceptually tricky part.
2. **Extraction** (all-layer hooks + pair batching).
3. **Metric + plots** (`synonymic_invariance.py`, heatmap).
4. **Write-up** (report section + figure captions, wiring to probing/patching narrative).

Steps 2–4 can start against a stub swap function while step 1 is finalised.

---

## 11. Deliverables

1. `S_{k,ℓ}` heatmap (12 layers × 5 levels) — the headline figure.
2. Per-level `S_{k,ℓ}` vs depth curves, with the `k ≈ ℓ+1` staircase (or its softened,
   under-trained version) annotated.
3. (Optional) effective-dimension panel: PCA participation ratio of `h_k` across layers —
   the RHM links invariance to a drop in effective dimensionality (their App. E); it is
   ~half a day on activations we already extract and makes a strong second figure.
4. A report section that frames invariance as the "what is discarded" complement to our
   probing ("what is present") and patching ("what is used") results.

---

*Next step options: (a) I scaffold `cfg/synonym_swap.py` with the tree generator and a
length-matched `P_ℓ`, plus the correctness tests; or (b) I draft the metric module
`evaluation/synonymic_invariance.py` against a stub. Say which and I'll make it concrete
against the real code.*
