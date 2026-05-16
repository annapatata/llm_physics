import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass
class CFGSample:
    string: List[int]
    length: int
    ancestor_symbols: Dict[int, List[int]]
    ancestor_indices: Dict[int, List[int]]
    boundaries: Dict[int, List[int]]
    deepest_boundary: List[int] # This needs to be calculated


def load_cfg(filepath):
    """
    Load a CFG from the .txt grammar format.
    Each line: <idx>\t<LHS>|-><RHS symbols space-separated>
    Returns a CFG instance.
    """
    rules = defaultdict(list)
    all_lhs = set()
    all_rhs = set()

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                _, rule_str = line.split("\t", 1)
            else:
                rule_str = line
            lhs_str, rhs_str = rule_str.split("|->")
            lhs = int(lhs_str.strip())
            rhs = tuple(int(x) for x in rhs_str.split())
            rules[lhs].append(rhs)
            all_lhs.add(lhs)
            all_rhs.update(rhs)

    terminals = all_rhs - all_lhs
    nonterminals = all_lhs
    root = (nonterminals - all_rhs).pop()

    return CFG(dict(rules), root, frozenset(terminals), frozenset(nonterminals))


class CFG:
    def __init__(self, rules, root, terminals, nonterminals):
        self.rules = rules          # dict: int -> list of tuples
        self.root = root            # int
        self.terminals = terminals  # frozenset of int
        self.nonterminals = nonterminals  # frozenset of int


    def sample_string(self) -> CFGSample:
        """
        Sample (x, s, p, b) from the CFG.

        Returns
        -------
        x : list[int]
            The terminal string; x[i] = x_{i+1} in paper notation.
        s : dict[int, list[int]]
            s[ℓ][i] = s_ℓ(i+1) — NT ancestor symbol at level ℓ for position i+1.
        p : dict[int, list[int]]
            p[ℓ][i] = p_ℓ(i+1) — NT ancestor index at level ℓ for position i+1.
        b : dict[int, list[int]]
            b[ℓ][i] = b_ℓ(i+1) — NT-end boundary indicator, defined for ℓ ∈ {1,…,L-1}.
        deepest_boundary: list[int]
            deepest_boundary[i] = b♯(i)

        Convention: levels ℓ are 1-indexed (1 = root, L = terminals).
        Lists are 0-indexed (index i = paper position i+1).
        """
        level_syms, par_arrays = self._generate()
        L = len(level_syms) # Number of levels, including the terminal level
        x = level_syms[L - 1]
        n = len(x)

        # ---- ancestor indices p[ℓ] ----------------------------------------
        # p[L][i] = i+1  (leaf position is its own index)
        # p[ℓ][i] = par_arrays[ℓ-1][ p[ℓ+1][i] - 1 ]   for ℓ < L
        #
        # par_arrays[k][j]  =  1-indexed parent at level k+1
        #                       of the (j+1)-th symbol at level k+2
        p = {}
        # Adjust L to be 1-indexed for the levels given in CONTEXT.md (L=7 for terminals)
        # L in CONTEXT.md refers to the number of *levels*, where Level 7 is terminals.
        # level_syms has L elements, indexed 0 to L-1. So level_syms[L-1] is the terminal level.
        # If L is the number of levels, then the highest level index is L.
        effective_L = L # Use L directly from `len(level_syms)`

        p[effective_L] = list(range(1, n + 1))
        for ell in range(effective_L - 1, 0, -1):
            p[ell] = [par_arrays[ell - 1][p[ell + 1][i] - 1] for i in range(n)]

        # ---- ancestor symbols s[ℓ] ----------------------------------------
        s = {}
        for ell in range(1, effective_L + 1):
            s[ell] = [level_syms[ell - 1][p[ell][i] - 1] for i in range(n)]

        # ---- NT-end boundary indicators b[ℓ] -------------------------------
        # b_ℓ(i) = 1  iff  p_ℓ(i) ≠ p_ℓ(i+1)  or  i is the last position
        b = {}
        for ell in range(1, effective_L): # Boundaries are defined for ℓ ∈ {1,...,L-1}
            b[ell] = [
                1 if (i == n - 1 or p[ell][i] != p[ell][i + 1]) else 0
                for i in range(n)
            ]

        # ---- deepest_boundary b♯(i) ----------------------------------------
        # b♯(i) — deepest NT-end: the smallest ℓ ∈ {2,...,L-1} such that b_ℓ(i)=1.
        # The CONTEXT.md says {2,...,L-1}, so we should iterate from 2 up to effective_L - 1.
        deepest_boundary = [0] * n # Initialize with a default value (e.g., 0 or -1)
        for i in range(n):
            deepest = 0
            for ell in range(2, effective_L): # Iterate from 2 to L-1 (inclusive)
                if b[ell][i] == 1:
                    deepest = ell
                    break # Smallest ℓ
            deepest_boundary[i] = deepest

        return CFGSample(string=x, length=n, ancestor_symbols=s,
                         ancestor_indices=p, boundaries=b,
                         deepest_boundary=deepest_boundary)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Top-down generation of the full level-symbol sequences.

        Returns
        -------
        level_syms : list[list[int]]
            level_syms[k] = symbol sequence at level k+1 (0-indexed k).
        par_arrays : list[list[int]]
            par_arrays[k][j] = 1-indexed parent position at level k+1
                               of the (j+1)-th symbol at level k+2.
        """
        level_syms = [[self.root]]
        par_arrays = []

        while not all(sym in self.terminals for sym in level_syms[-1]):
            current = level_syms[-1]
            next_seq = []
            par_next = []
            for i, sym in enumerate(current, start=1):  # i is 1-indexed
                rhs = random.choice(self.rules[sym]) if sym in self.nonterminals else (sym,)
                for child in rhs:
                    next_seq.append(child)
                    par_next.append(i)
            par_arrays.append(par_next)
            level_syms.append(next_seq)

        return level_syms, par_arrays