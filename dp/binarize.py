"""
Convert a CFG to binary form (Chomsky Normal Form) for use with CYK / inside algorithms.

Two transformations applied:
  1. Preterminals  — for each terminal t, introduce a fresh NT PT_t with rule PT_t -> t.
                     Replace every occurrence of t in rule RHS with PT_t.
  2. Binarization  — for every ternary rule A -> B C D, introduce a fresh auxiliary NT X
                     and replace the rule with  A -> B X  and  X -> C D.

After both transforms every rule is either:
  - A -> B C   (binary, both children are NTs)
  - PT_t -> t  (unit, stored separately in BinarizedCFG.preterminals)

Rule probabilities:
  - Original NT A: P(A -> rule) = 1 / |R(A)|  (uniform over original rules)
  - Auxiliary / preterminal NTs: P = 1.0       (only one rule each)
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass
class BinarizedCFG:
    binary_rules: List[Tuple[int, int, int, float]]  # (A, B, C, prob)
    preterminals: Dict[int, int]                      # terminal -> preterminal NT id
    root: int
    terminals: FrozenSet[int]
    all_nts: FrozenSet[int]                           # all NT ids (original + aux + preterminal)


def binarize(cfg) -> BinarizedCFG:
    next_id = max(cfg.nonterminals) + 1

    # Step 1: create one preterminal NT per terminal symbol
    preterminals: Dict[int, int] = {}
    for t in sorted(cfg.terminals):
        preterminals[t] = next_id
        next_id += 1

    binary_rules: List[Tuple[int, int, int, float]] = []

    # Step 2: process each original rule
    for lhs, rhs_list in cfg.rules.items():
        p = 1.0 / len(rhs_list)
        for rhs in rhs_list:
            # Replace terminals with their preterminal NTs
            nt_rhs = tuple(preterminals.get(sym, sym) for sym in rhs)

            if len(nt_rhs) == 2:
                binary_rules.append((lhs, nt_rhs[0], nt_rhs[1], p))
            elif len(nt_rhs) == 3:
                # A -> B C D  =>  A -> B X,  X -> C D
                aux = next_id
                next_id += 1
                binary_rules.append((lhs, nt_rhs[0], aux, p))
                binary_rules.append((aux, nt_rhs[1], nt_rhs[2], 1.0))
            else:
                raise ValueError(f"Rule length {len(nt_rhs)} not supported: {lhs} -> {rhs}")

    all_nts: set = set()
    for A, B, C, _ in binary_rules:
        all_nts.update((A, B, C))

    return BinarizedCFG(
        binary_rules=binary_rules,
        preterminals=preterminals,
        root=cfg.root,
        terminals=cfg.terminals,
        all_nts=frozenset(all_nts),
    )
