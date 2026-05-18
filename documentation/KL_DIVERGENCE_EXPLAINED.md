# KL Divergence for CFG Evaluation — Complete Explanation

## What We Are Measuring

After training GPT on CFG strings, we want to know: **how close is the model's learned distribution to the true CFG distribution?**

KL divergence measures exactly this. At each position `c` in a test string, the model predicts a probability distribution over the next token. The true CFG also induces a distribution over the next token. KL divergence tells us how much the model's distribution diverges from the truth.

**Paper target (Result 3, Figure 4):**

| Grammar | GPT (absolute) KL | GPT_rot KL |
|---------|-------------------|------------|
| cfg3b   | 0.00008           | 0.00008    |
| cfg3f   | 0.00455           | 0.00455    |

A KL of 0.00455 nats/token means GPT_rot has almost perfectly learned cfg3f's distribution. Vanilla GPT does worse on hard grammars.

---

## Part 1: What KL Divergence Is

KL divergence from distribution P to distribution Q is:

```
KL(P || Q) = Σ_x  P(x) × log( P(x) / Q(x) )
```

**Key properties:**
- Always ≥ 0
- = 0 only when P = Q exactly
- NOT symmetric: KL(P||Q) ≠ KL(Q||P)
- Undefined (= ∞) if Q(x) = 0 for any x where P(x) > 0

**Which direction does the paper use?**

```
KL(P_true || P_model)
```

This is "true distribution diverges from model." The true CFG distribution is P, the model is Q.

**Why this direction and not the other?**

- If the model assigns P_model(t) = 0 to a token that P_true(t) > 0, KL = ∞. This is a severe penalty: the model is asserting something is *impossible* that the grammar says *can happen*.
- If the model assigns mass to tokens the grammar forbids (P_true(t) = 0), this direction ignores it (since the term 0 × log(...) = 0). The model is uncertain about impossible things, but the grammar doesn't penalise that.

**In nats or bits?**

The paper uses nats (natural log). Bits use log base 2. Convert: 1 nat = 1/ln(2) ≈ 1.44 bits.

**Averaged over positions and strings:**

```
KL_avg = (1 / total_positions) × Σ_{strings x} Σ_{positions c} KL(P_true(·|x[0..c-1]) || P_model(·|x[0..c-1]))
```

---

## Part 2: The Two Distributions We Need

At each position `c` in test string `x[0..n-1]`, we need:

### P_true(t | x[0..c-1]) — The True CFG Distribution

This is the probability that the **next token is `t`**, given the prefix seen so far, under the true grammar.

Formally:
```
P_true(t | x[0..c-1]) = Σ_{s ∈ L(G), s starts with x[0..c-1]+t} P_CFG(s)
                         ─────────────────────────────────────────────────────
                         Σ_{s ∈ L(G), s starts with x[0..c-1]}   P_CFG(s)
```

Both numerator and denominator are sums over **infinitely many complete strings** — all valid CFG strings that share the given prefix. You cannot enumerate them; you need a DP algorithm (see Part 3).

### P_model(t | x[0..c-1]) — The Model's Distribution

This is simple: run GPT on the token sequence `[BOS, x[0], ..., x[c-1]]` and take the softmax of the output logits at position `c`.

The model's vocabulary includes BOS (0), terminals {1, 2, 3}, EOS (4). Since the CFG distribution is only over terminals {1, 2, 3} at mid-string positions, we **renormalise** the model's probabilities over {1, 2, 3} only:

```python
probs_terminals = softmax(logits)[[1, 2, 3]]
P_model = probs_terminals / probs_terminals.sum()
```

---

## Part 3: Why the Empirical Approach Is Wrong

Consider this implementation attempt:

```python
empirical_counts = defaultdict(int)
for _ in range(3000):
    sample = cfg.sample_string().string
    if sample[:c] == x[:c]:          # does this string share our prefix?
        empirical_counts[sample[c]] += 1
```

**This is fundamentally broken.** Here is why:

### Problem 1: The prefix is astronomically rare

For a grammar with 3 terminals and strings of length ~300, the number of distinct valid strings is enormous. The probability of a *specific* prefix of length `c` is roughly:

```
P(prefix x[0..c-1]) ≈ (1/3)^c    (very roughly, ignoring grammar structure)
```

For c = 100:  P ≈ (1/3)^100 ≈ 2 × 10^{−48}

You would need approximately **10^48 samples** to get even one match. With 3000 samples, you get zero matches for essentially every position `c > 15` or so.

### Problem 2: The `matches < 10` filter removes almost all positions

The code skips any prefix where fewer than 10 strings happened to match. Since nearly every prefix of length ≥ 20 gets zero matches in 3000 samples, the code would skip virtually all positions in the test string. You'd only evaluate the model at the first few positions, which is not meaningful.

### Problem 3: Even where it works, it has enormous variance

Suppose by miracle you get 10 matches out of 3000 samples. Then your estimate of P_true(t) is based on 10 observations. That gives a standard error of roughly:

```
SE ≈ sqrt(p(1-p)/10) ≈ 0.16
```

A standard error of 0.16 on a probability that might be 0.3 is completely unreliable for computing KL divergence.

### Problem 4: It gives wrong numbers even in theory

The empirical estimate is correct in the limit of infinite samples, but "converges" at a rate of 1/sqrt(matches). To get 1% accuracy on P_true, you need ~10,000 matches. Since each string matches with probability ~10^{-48} for typical c, you'd need 10^{52} samples. This is not a runtime optimization problem — it's a **fundamental impossibility**.

### The correct answer: use DP

The inside algorithm computes P_true(t | prefix) **exactly** (up to floating point) in O(n³) time using dynamic programming. There is no approximation, no sampling, no variance. See Part 4.

---

## Part 4: The DP Algorithm for P_true

We want: **P_true(next = t | prefix x[0..c-1])** for each t ∈ {1, 2, 3}.

### Step 1: Build the inside table for the full test string

The inside algorithm fills a table `alpha[i, j, A]` = P(A generates exactly x[i..j]):

```
alpha[i, i, PT_t] = 1.0   if x[i] == t   (base case: preterminals)

alpha[i, j, A]    = Σ_{A→BC} p × Σ_{k=i}^{j-1} alpha[i, k, B] × alpha[k+1, j, C]
```

Complexity: O(n³ × R) where R = number of rules.

This table is computed **once per test string** and shared across all positions c.

### Step 2: The `right` function

For each candidate next token t, define the extended prefix `y = x[0..c-1] + [t]` (length c+1).

We want: P(root generates some string that *starts with* y[0..c]).

The key function:

```
right(A, start) = P(A generates any string starting with y[start..c])
```

Base cases:
```
right(A,   c+1) = 1.0  for all A        ← no prefix left; any continuation is valid
right(PT_t, c)  = 1.0  if y[c] == t     ← preterminal for t at the last position
right(PT_t, c)  = 0.0  if y[c] != t
right(PT_t, s<c)= 0.0                   ← PT generates 1 token; can't start a 2+ token prefix
```

Recursive case for binary rule A → B C (prob p):
```
right(A, start) = Σ_{A→BC} p × [  right(B, start)                                  ← B covers all remaining prefix
                                  + Σ_{k=start}^{c-1} alpha[start,k,B] × right(C,k+1)  ← B covers part of prefix
                                ]
```

The two terms are **mutually exclusive**:
- First term: the split between B and C is at or after position c (B overshoots or exactly covers y[start..c])
- Second term: the split is before c (B generates exactly y[start..k] for k < c)

### Step 3: Why `right(B, start)` matters — the overshooting problem

This is the subtlety that makes the implementation hard. For high-level NTs:

- Level-2 NTs generate strings of length 32–243 tokens
- Level-3 NTs generate strings of length 16–81 tokens
- The root generates strings of length 64–729 tokens

At a prefix position c = 96 with the root's left child (a level-2 NT), that NT **always** generates more than 96 tokens. Its span goes beyond position c. The exact-span probability `alpha[0, 96, level2_NT]` could be zero (if that NT never generates exactly 97 tokens), but the **prefix probability** `right(level2_NT, 0)` is non-zero (it generates more than 96 tokens, which still starts with the prefix).

**If you mistakenly use `alpha[0, c, B]` instead of `right(B, start)`**, you get:
- Zero probability for tokens that are clearly valid continuations
- The actual next token from a valid CFG string gets probability 0
- KL divergence becomes undefined (or infinite)

This was the bug in the first implementation attempt.

### Step 4: Solving the right recursion efficiently

The same-start dependency (`right(A, start)` depends on `right(B, start)` where B is A's left child) creates a system of equations. Since the grammar is acyclic (hierarchical levels), this system has a unique solution.

In matrix form:
```
right[start] = IS[start] + W × right[start]

where:
  W[A, B]    = Σ_{A→BC rules} p     (left-child weight matrix)
  IS[start]  = Σ_{A→BC} p × Σ_{k=start}^{c-1} alpha[start,k,B] × right[k+1,C]
                                    (inner k-sum, uses right values from later starts)

Solution: right[start] = (I − W)^{−1} × IS[start]
                       = M × IS[start]
```

**M = (I − W)^{−1}** is precomputed once per grammar. For each start, it is a single matrix-vector multiply: O(N²).

The matrix (I − W) is non-singular because:
- W has zeros on the diagonal (no NT is its own left child in an acyclic grammar)
- (I − W) is unit lower-triangular in topological order
- det(I − W) = 1

### Step 5: Unnormalized probability and normalization

```python
unnorm[t] = right[0, root_id]   # right(root, 0) with extended prefix ending in t

P_true(t | prefix) = unnorm[t] / Σ_{t'} unnorm[t']
```

We compute this for each t ∈ {1, 2, 3} and normalize.

---

## Part 5: Why P_true Sums to 1 (Sanity Check)

`Σ_t P_true(t | prefix)` should equal 1.0. This is guaranteed because:

```
Σ_t unnorm(t) = Σ_t P(root generates any string starting with x[0..c-1]+t)
              = P(root generates any string starting with x[0..c-1])
```

Since x[0..c-1] is a prefix of a valid CFG string, this probability is strictly positive. After normalization, the distribution sums to exactly 1.

---

## Part 6: Full Algorithm Summary

```
Input: test string x[0..n-1], trained GPT model, binarized CFG

Precompute once per grammar:
  M = (I − W)^{−1}   where W[A,B] = Σ_{A→BC rules} p

For each test string x:

  1. Build inside table alpha[i,j,A] for x in O(n³)

  2. Get model's distributions in ONE forward pass:
       feed [BOS, x[0], ..., x[n-1]] to GPT
       extract logits at positions 0..n-1
       softmax → renormalise over {1,2,3} → p_model[c, :] for each c

  3. For each position c from 0 to n-1:

       For each candidate terminal t ∈ {1, 2, 3}:
         a. Base: right[c+1, A] = 1.0 for all A
         b. Base: right[c, PT_t] = 1.0 (matches y[c] = t), 0 for other preterminals
         c. Propagate: right[c] = M @ right_base[c]   ← (I−W)^{-1} handles left-child overshoot
         d. For start from c-1 down to 0:
              IS[start, A] = Σ_{A→BC} p × Σ_k alpha[start,k,B] × right[k+1,C]
              right[start] = M @ IS[start]
         e. unnorm[t] = right[0, root]

       Normalize: P_true[c, t] = unnorm[t] / Σ unnorm[t']

       KL[c] = Σ_t P_true[c,t] × log(P_true[c,t] / P_model[c,t])

  total_KL += Σ_c KL[c]

Result: total_KL / total_positions   (nats per token)
```

---

## Part 7: Complexity

| Step | Complexity per string |
|------|-----------------------|
| Inside table | O(n³ × R) |
| GPT forward pass | O(n × d²) one pass |
| Right function (all c, all t) | O(n² × V × (c × R + N²)) |
| Total | O(n³ × R × V) |

For n=300, R=60, V=3, N=40:
- Inside: ~300³ × 60 ≈ 1.6B ops
- Right function: dominates, ~2–3 minutes per string

For 200 strings: ~6–10 hours CPU, or ~30–60 minutes with GPU for the GPT part.

**Practical advice**: Use `--n_strings 50` for a quick sanity check. The paper likely used a compiled C++/CUDA implementation.

---

## Part 8: What the Numbers Mean

**KL ≈ 0 (GPT_rot on cfg3b):**
GPT_rot has almost perfectly learned cfg3b. At every position, the model's next-token distribution is nearly identical to the true CFG distribution. The model has compressed the entire grammar into its weights.

**KL ≈ 0.00455 (GPT_rot on cfg3f):**
Still very small. The model is very close to the true distribution. The small error comes from the hardness of cfg3f (maximum ambiguity, hardest to parse), where the model makes rare mistakes in extreme grammatical configurations.

**KL ≈ 0.455 (vanilla GPT on cfg3f):**
100x larger. Vanilla GPT (absolute positional embeddings) struggles on hard CFGs because:
- The CFG structure is relative (subtree spans repeat at the same depth regardless of absolute position)
- Absolute embeddings encode the wrong coordinate system
- The model makes systematic errors in ambiguous regions

**KL = ∞ in theory:**
Would happen if the model assigns zero probability to a token the grammar allows. In practice, softmax never gives exactly 0, so KL is always finite. The renormalisation over {1,2,3} also prevents this.

---

## Part 9: Connection to Training Loss

Training loss is cross-entropy:

```
loss = -log P_model(x[c] | x[0..c-1])
     = -Σ_t P_actual(t) × log P_model(t)    where P_actual is a delta at x[c]
```

Cross-entropy = entropy + KL divergence:

```
H(P_true, P_model) = H(P_true) + KL(P_true || P_model)
```

So:

```
training loss → H(P_true) + KL(P_true || P_model)
```

- H(P_true) is the irreducible entropy of the CFG — the theoretical minimum loss
- KL(P_true || P_model) is the extra loss due to the model not matching the true distribution

**This is why the teammate was right that training loss can't go below ~0.74 for cfg3f**: 0.74 is the estimated entropy H(P_true) of cfg3f. The KL term can approach 0 with enough training, but the entropy floor is fixed.

```
training loss = H(P_true) + KL ≈ 0.74 + KL
```

When training loss ≈ 0.76:
```
KL ≈ 0.76 − 0.74 = 0.02 nats/token
```

The paper reports KL ≈ 0.00455 for GPT_rot on cfg3f, suggesting the paper's model trained much longer/harder than the current 20K-step checkpoint. The current checkpoint has KL ≈ 0.02, which is ~4× worse than the paper.

---

## Part 10: Common Mistakes

| Mistake | Effect |
|---------|--------|
| Using empirical sampling | Zero matches for most positions; skips almost all data |
| Using `alpha[start,c,B]` instead of `right(B,start)` | Zero probability for actual next token at many positions |
| Not renormalising model probs over {1,2,3} | EOS/BOS tokens pollute the distribution |
| Computing KL(P_model \|\| P_true) instead of KL(P_true \|\| P_model) | Different quantity; model errors are penalised differently |
| Using greedy decoding instead of softmax | P_model would be a delta function, not a distribution |
| Forgetting the (I−W)^{-1} correction | Prefix probability underestimated; NTs that overshoot prefix get 0 weight |
