# Probing: What It Is, Why It Works, and How the Code Does It

## The One-Line Summary

The model was never told what a parse tree is. We want to check if it figured one out anyway.
We do that by asking: *can a simple linear classifier read the parse tree off the frozen hidden states?*

---

## Part 1: What a Hidden State Actually Is

When GPT processes a sequence of tokens, every transformer block takes the current matrix of vectors and transforms it. After the **last** block, you have a matrix of shape `(sequence_length, 768)`. Row `i` of that matrix — call it `E_i` — is the model's internal representation of "token i, given everything before it."

`E_i` is just 768 floating-point numbers. It is not human-readable. The model never explicitly computed "I am inside NT-20." But to correctly predict next tokens on a hard CFG, the model *must* internally track which subtree it is in, because the legal continuations of the string depend on it. The question is: did that tracking happen, and is it stored in a format a simple function can read?

### What "linearly encoded" means

If you can train a **linear function** (a matrix multiply + softmax) that takes `E_i` and reliably outputs the correct ancestor NT, then the parse tree is *linearly encoded* in the hidden state. The information is there, and it's geometrically organized — one region of 768-dim space corresponds to NT-19, another to NT-20, another to NT-21.

A nonlinear classifier (e.g. a neural network with hidden layers) could potentially decode information that is tangled and scrambled. A linear probe is a strict test: the structure has to be *directly readable* with a hyperplane, no further computation allowed.

---

## Part 2: The Thing We Are Predicting

For each level `ℓ ∈ {2, 3, 4, 5, 6}` and each position `i` in the string, the ground truth label is:

```
s_ℓ(i) = which NT symbol at level ℓ generated the token at position i
```

For these CFGs, there are exactly 3 NTs per level:

| Level | NT symbols |
|-------|-----------|
| 2     | {19, 20, 21} → mapped to classes {0, 1, 2} |
| 3     | {16, 17, 18} → mapped to classes {0, 1, 2} |
| 4     | {13, 14, 15} → mapped to classes {0, 1, 2} |
| 5     | {10, 11, 12} → mapped to classes {0, 1, 2} |
| 6     | {7, 8, 9}   → mapped to classes {0, 1, 2} |

So `n_classes = 3` for every level. This is a **3-way classification** problem at every token position.

---

## Part 3: Why a Single Vector `E_i` Might Not Be Enough

Here is the subtlety that motivates the multi-head design.

Predicting `s_ℓ(i)` from `E_i` alone requires knowing which subtree token `i` belongs to at level `ℓ`. But in the middle of a subtree, the model hasn't seen the full subtree yet — it can't be certain which NT is responsible. The information about subtree identity gets assembled across multiple positions.

At **boundary positions** (the last token of a subtree), `E_i` should contain a complete summary of the entire subtree that just finished. But at **non-boundary positions**, the boundary hasn't been seen yet, and the full ancestry is ambiguous from left context alone.

The full probe handles this by being allowed to look at *all* positions `E_0, E_1, ..., E_{T-1}` when making its prediction for position `i`. This is like letting the probe "attend" to any part of the sequence.

---

## Part 4: The Probe Architecture — Step by Step

The probe is described by Equation 4.2 in the paper:

```
G_i(x) = Σ_{r ∈ [H], k ∈ [T]} w_{r,i→k} · f_r(E_k)
```

Let's unpack every symbol.

### `E_k` — hidden states (input, frozen)

`E_k` is the 768-dim hidden state at position `k`, extracted from the **last transformer block** of the frozen, pretrained GPT model. The GPT model's weights are locked. We never touch them during probe training.

For a sequence of length `T`, you have `T` of these vectors: `E_0, E_1, ..., E_{T-1}`.

### `f_r` — the linear maps (one per head, these are learned)

`f_r` is a linear function: `f_r : R^768 → R^3`.

It's just a matrix of shape `(768, 3)`. It transforms each hidden state into a 3-dim vector (a score for each of the 3 NT classes).

There are `H = 16` heads, so there are 16 such matrices. They are all **different** — each head learns a different linear projection of the hidden states.

In the code, all 16 are stored as one big `Linear(768, 16 * 3)` layer for efficiency:

```python
self.linear = nn.Linear(n_embd, n_heads * n_classes, bias=False)
# shape: (768, 48) — this contains all 16 matrices concatenated
```

### `w_{r,i→k}` — the attention weights (position-only, these are learned)

This is the key innovation of the probe. The weight `w_{r,i→k}` says: *when predicting the ancestor of position `i` using head `r`, how much should we rely on the hidden state at position `k`?*

Crucially, these weights depend **only on the positions `i` and `k`**, not on the content of the hidden states. This is what keeps the probe linear.

They are computed as:

```
w_{r,i→k} = softmax_k( <P_{i,r}, P_{k,r}> / sqrt(d') )
```

where `P_{i,r}` is a **learned position embedding** for position `i` in head `r`, and `d' = 1024` is its dimension. The `<·,·>` is a dot product.

**In plain English:** the probe learns "which positions matter for predicting the ancestor of position `i`." For a boundary position, it might learn to look mostly at itself. For a non-boundary position, it might learn to look ahead (to the next boundary) or across the whole subtree.

### There is no GPT attention layer here

The probe has its **own** attention mechanism, completely separate from GPT's attention. GPT's attention layers processed the tokens during pretraining and produced the hidden states `E_k`. The probe's attention is a new, lightweight mechanism that decides how to aggregate those already-computed hidden states when making a classification.

GPT's attention: content-driven, causal, trained on next-token prediction.  
Probe's attention: position-driven only, non-causal, trained on ancestor symbol prediction.

### Putting it together: what does `G_i` compute?

```
G_i(x) = Σ_{r=1..16} Σ_{k=0..T-1} w_{r,i→k} · f_r(E_k)
```

For each head `r`:
1. Apply the linear map `f_r` to every hidden state → 3 scores per position
2. Take a weighted average of those scores, weighted by `w_{r,i→k}`
3. This gives a 3-dim vector from head `r`

Then sum the 3-dim vectors from all 16 heads → final 3-dim output `G_i`.

`G_i` contains the raw logits. Argmax gives the predicted class (0, 1, or 2), which maps back to an NT symbol (19, 20, or 21 for level 2).

---

## Part 5: Walking Through the Code

```python
def forward(self, hidden, attn_mask=None):
    # hidden: (B, T, 768) — frozen GPT hidden states
    B, T, _ = hidden.shape
```

**Step 1: Compute position embeddings**

```python
    pos = torch.arange(T, device=hidden.device)
    P = self.pos_emb(pos).view(T, self.n_heads, self.d_pos)
    # P: (T, 16, 1024)
    # P[i, r, :] = position embedding for position i in head r
    
    P = P.permute(1, 0, 2)  # (16, T, 1024)
```

`self.pos_emb` is an `nn.Embedding` table of shape `(512, 16 * 1024)`. It contains learned vectors — the probe learns these during training. Each position gets a 1024-dim embedding for each of the 16 heads.

**Step 2: Compute attention scores**

```python
    scores = torch.bmm(P, P.transpose(-2, -1)) / math.sqrt(self.d_pos)
    # P:       (16, T, 1024)
    # P^T:     (16, 1024, T)
    # scores:  (16, T, T)
    # scores[r, i, k] = <P_{i,r}, P_{k,r}> / sqrt(1024)
```

This is a standard scaled dot-product attention score. For each head `r` and each pair of positions `(i, k)`, it computes how much position `i` should attend to position `k`. These scores come entirely from the **position embeddings** — no hidden state values involved.

**Step 3: Apply optional mask (for diagonal/local probe)**

```python
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    # weights: (16, T, T)
    # weights[r, i, k] = w_{r,i→k}
    # Softmax over k: weights sum to 1 for each (r, i) pair
```

For the full probe: `attn_mask = None`, all positions can attend to all positions.  
For the diagonal probe (δ=0): `attn_mask[i, k] = (i == k)`, so position `i` can only attend to itself.

**Step 4: Apply linear maps to hidden states**

```python
    transformed = self.linear(hidden).view(B, T, self.n_heads, self.n_classes)
    # self.linear: Linear(768, 16*3)
    # After linear: (B, T, 48)
    # After view:   (B, T, 16, 3)
    # transformed[b, k, r, c] = f_r(E_k)[c]
    #   = the c-th class score for position k under head r
```

**Step 5: Weighted sum**

```python
    output = torch.einsum('hik,bkhc->bihc', weights, transformed)
    # For each batch b, position i, head h, class c:
    # output[b, i, h, c] = Σ_k  weights[h, i, k] * transformed[b, k, h, c]
    #                     = Σ_k  w_{h,i→k} * f_h(E_k)[c]
    # output: (B, T, 16, 3)
    
    return output.sum(dim=2)
    # Sum over 16 heads → (B, T, 3)
    # Final logits for each position
```

---

## Part 6: Does Each Head Give Its Own NT Prediction?

Yes — sort of. But they are summed, not voted.

Each head `r` contributes:
```
contribution_r[i, c] = Σ_k  w_{r,i→k} * f_r(E_k)[c]
```

This is a 3-dim vector (one score per NT class) for each position. The 16 heads are summed to produce the final 3-dim logit vector for each position:

```
G_i[c] = Σ_{r=1..16}  contribution_r[i, c]
```

Think of each head as a *specialist* attending to different positional ranges. Head 1 might learn to look at nearby positions; head 2 might look at positions 20-50 away; etc. They are complementary, not redundant. The final answer aggregates all their contributions.

The 3-class classification is a joint decision of all 16 heads, not a majority vote.

---

## Part 7: How Many Times Do You Run Probing?

**Yes — once per level, so 5 times in total** (levels 2, 3, 4, 5, 6).

Each run trains **two probes**: a full probe and a diagonal probe. So you end up with 10 probe models per GPT checkpoint per grammar.

Here is the full breakdown:

```
For each level ℓ ∈ {2, 3, 4, 5, 6}:
    
    label_map = {19→0, 20→1, 21→2}  (or equivalent for that level)
    
    ┌─ Full probe (Result 4) ────────────────────────────────────────────┐
    │  - Attention: unrestricted, attends to all positions               │
    │  - Trained on: all positions in the string                         │
    │  - Evaluated on: all positions                                     │
    │  - Answers: "can the probe read s_ℓ(i) for any position i?"        │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌─ Diagonal probe (Result 5) ────────────────────────────────────────┐
    │  - Attention: diagonal only (δ=0), each position attends to itself  │
    │  - Trained on: all positions (including non-boundaries)             │
    │  - Evaluated on: ONLY positions where b_ℓ(i) = 1 (boundaries)      │
    │  - Answers: "at the last token of a subtree, does E_i alone encode  │
    │    which NT closed that subtree?"                                   │
    └────────────────────────────────────────────────────────────────────┘
```

### The complete execution plan

```
run_probing_experiment(...)
│
├── Level 2: NTs {19, 20, 21}
│   ├── train full_probe_L2    (30K iters, 60 seqs/iter)
│   ├── train diag_probe_L2    (30K iters, 60 seqs/iter)
│   └── evaluate both → ProbeResult(level=2, full_acc, boundary_acc)
│
├── Level 3: NTs {16, 17, 18}
│   ├── train full_probe_L3
│   ├── train diag_probe_L3
│   └── evaluate → ProbeResult(level=3, ...)
│
├── Level 4: NTs {13, 14, 15} ...
├── Level 5: NTs {10, 11, 12} ...
└── Level 6: NTs {7, 8, 9}   ...
```

Final printed table matches paper Figure 5 (Result 4) and Figure 7 (Result 5).

---

## Part 8: The Control Experiment (GPT_rand)

Pass `--random_gpt` to run the same probing on a GPT model with **random, untrained weights**.

If the probe still achieves high accuracy, that would mean the *architecture itself* encodes structure, regardless of training. That would invalidate the whole finding.

Expected results:
- **Trained GPT_rot on cfg3f:** ~100% (NT6), ~97% (NT5), ~95% (NT4), ~93% (NT3), ~94% (NT2)
- **GPT_rand (random weights):** ~33% for all levels (chance level — 1-in-3 random guess)

The gap between these two numbers is the proof that the structure is **learned through next-token prediction training**, not baked in by the architecture.

---

## Part 9: What the Results Mean

### Result 4 (full probe, all positions)

High accuracy here means the hidden states, taken together across the full sequence, linearly encode the complete parse tree. The model has discovered hierarchical structure without ever being shown parse annotations.

### Result 5 (diagonal probe at boundaries)

High accuracy here is the deeper finding. It means: the single hidden state at the *last token of a subtree* — just 768 numbers — contains enough information to identify which NT generated that entire subtree, with no help from neighboring positions.

This is the "summary position" phenomenon: as the model processes the last token of a subtree, it accumulates the entire subtree's identity into that one vector. This is precisely what a dynamic programming algorithm would do: when the DP table entry `DP(i, j, A)` is computed at position `j`, it summarizes everything that happened from `i` to `j`.

### Why it fails at non-boundary positions (if you tested there)

A token in the middle of a subtree has only seen the subtree's prefix. The subtree's NT is not yet determined — different NTs could generate the same prefix and different suffixes. The information simply does not exist in the left context, so no linear probe (or any probe) can recover it. This is an information-theoretic impossibility, not a limitation of the probe design.

---

## Part 10: Data Flow Summary

```
CFG grammar
    │
    ▼
cfg.sample_string()  →  [1, 3, 1, 2, 3, ...]  +  ancestor_symbols[ℓ][i]  +  boundaries[ℓ][i]
    │                       token sequence            ground truth labels        boundary flags
    │
    ▼
[BOS, 1, 3, 1, 2, 3, ...]
    │
    ▼ (frozen GPT_rot forward pass)
    │
    ▼
E_0, E_1, E_2, ...  ← shape: (T, 768) — last-layer hidden states, frozen
    │
    ├──────────────────────────────────────────────────────────────────────────────┐
    │                                                                              │
    ▼                                                                              ▼
[Full probe] MultiHeadLinearProbe                          [Diag probe] MultiHeadLinearProbe
  - pos_emb learns: "to predict s_ℓ(i),                     - pos_emb forced: "to predict s_ℓ(i),
    attend to positions k that carry                           only E_i is available"
    subtree-boundary information"
  - linear learns: "in which direction                       - linear learns: "from E_i alone,
    of hidden state space does NT identity                     at a boundary position, read
    live?"                                                     which NT just closed"
    │                                                                              │
    ▼                                                                              ▼
G_i ∈ R^3 for every position i                             G_i ∈ R^3 for boundary positions only
argmax → predicted NT class                                argmax → predicted NT class
    │                                                                              │
    ▼                                                                              ▼
compare to s_ℓ(i) → full accuracy (Result 4)              compare to s_ℓ(i) where b_ℓ(i)=1
                                                           → boundary accuracy (Result 5)
```

---

## Quick Reference: Key Numbers

| Symbol | Value | Meaning |
|--------|-------|---------|
| `n_embd` | 768 | Hidden state dimension (GPT-2 small) |
| `n_heads` | 16 | Number of probe attention heads |
| `d_pos` | 1024 | Position embedding dimension per head |
| `n_classes` | 3 | NT symbols per level |
| `n_iters` | 30,000 | Probe training iterations |
| `batch_size` | 60 | Sequences per training step |
| `levels` | {2,3,4,5,6} | 5 separate probe experiments |
| `probes per level` | 2 | full + diagonal |
| `total probe models` | 10 | per GPT checkpoint per grammar |
