"""
Tree-based CFG generator and synonym-swap operator P_ℓ (Extension A).

build_tree  — builds an explicit parse tree recording which rule was used at each node
flatten     — converts a parse tree back to a CFGSample (same fields as CFG.sample_string())
synonym_swap — samples (x, P_ℓ x): a base string and a synonym-swapped partner
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cfg.grammar import CFG, CFGSample


@dataclass
class Node:
    symbol: int
    level: int        # 1-indexed depth (root = 1)
    rule_idx: int     # index into cfg.rules[symbol]; -1 for terminals
    children: List[Node] = field(default_factory=list)
    start: int = field(default=-1, repr=False)  # first terminal index (set by _assign_offsets)
    end: int = field(default=-1, repr=False)    # exclusive end terminal index


@dataclass
class SwapResult:
    base: CFGSample
    swapped: CFGSample
    parent_symbol: int
    span: Tuple[int, int]          # (start, end_inclusive) of swapped subtree in base
    base_boundary_index: int       # = span[1] — last terminal of swapped subtree in base
    swapped_boundary_index: int    # last terminal of swapped subtree in swapped string
    length_matched: bool


# ── Tree construction ─────────────────────────────────────────────────────────

def build_tree(cfg: CFG, symbol: int = None, level: int = 1) -> Node:
    """Build a full random parse tree from `symbol` (defaults to cfg.root)."""
    if symbol is None:
        symbol = cfg.root
    if symbol in cfg.terminals:
        return Node(symbol=symbol, level=level, rule_idx=-1)
    rules = cfg.rules[symbol]
    rule_idx = random.randrange(len(rules))
    rhs = rules[rule_idx]
    node = Node(symbol=symbol, level=level, rule_idx=rule_idx)
    for child_sym in rhs:
        node.children.append(build_tree(cfg, child_sym, level + 1))
    return node


def _build_subtree(cfg: CFG, symbol: int, rule_idx: int, level: int) -> Node:
    """Build a subtree using a specific rule, with random expansions below."""
    rhs = cfg.rules[symbol][rule_idx]
    node = Node(symbol=symbol, level=level, rule_idx=rule_idx)
    for child_sym in rhs:
        node.children.append(build_tree(cfg, child_sym, level + 1))
    return node


def _assign_offsets(node: Node, offset: int = 0) -> int:
    """Assign terminal start/end offsets to every node. Returns exclusive end."""
    node.start = offset
    if not node.children:
        node.end = offset + 1
        return offset + 1
    cur = offset
    for c in node.children:
        cur = _assign_offsets(c, cur)
    node.end = cur
    return cur


def _count_terminals(node: Node) -> int:
    if not node.children:
        return 1
    return sum(_count_terminals(c) for c in node.children)


def _get_terminals(node: Node) -> List[int]:
    if not node.children:
        return [node.symbol]
    out = []
    for c in node.children:
        out.extend(_get_terminals(c))
    return out


def _collect_nodes_at_level(node: Node, target_level: int) -> List[Node]:
    if node.level == target_level:
        return [node]
    out = []
    for c in node.children:
        out.extend(_collect_nodes_at_level(c, target_level))
    return out


def _find_node_by_pos(node: Node, level: int, start: int) -> Optional[Node]:
    if node.level == level and node.start == start:
        return node
    for c in node.children:
        found = _find_node_by_pos(c, level, start)
        if found is not None:
            return found
    return None


# ── Flatten: tree → CFGSample ─────────────────────────────────────────────────

def flatten(root: Node) -> CFGSample:
    """
    Convert a parse tree to a CFGSample with the same fields as CFG.sample_string().
    """
    _assign_offsets(root)
    terminals = _get_terminals(root)
    n = len(terminals)

    # Collect all nodes grouped by level, sorted left-to-right.
    level_nodes: dict = {}

    def collect(node: Node) -> None:
        level_nodes.setdefault(node.level, []).append(node)
        for c in node.children:
            collect(c)

    collect(root)
    for lv in level_nodes:
        level_nodes[lv].sort(key=lambda nd: nd.start)

    max_level = max(level_nodes)

    # ancestor_symbols[ℓ][i] and ancestor_indices[ℓ][i]
    ancestor_symbols: dict = {}
    ancestor_indices: dict = {}
    for lv, nodes in level_nodes.items():
        syms = [0] * n
        idxs = [0] * n
        for pos, nd in enumerate(nodes, start=1):   # 1-indexed position within level
            for i in range(nd.start, nd.end):
                syms[i] = nd.symbol
                idxs[i] = pos
        ancestor_symbols[lv] = syms
        ancestor_indices[lv] = idxs

    # boundaries[ℓ][i] = 1 iff terminal i is the last in its level-ℓ subtree
    boundaries: dict = {}
    for lv in range(1, max_level):   # exclude terminal level
        if lv not in ancestor_indices:
            continue
        p = ancestor_indices[lv]
        boundaries[lv] = [
            1 if (i == n - 1 or p[i] != p[i + 1]) else 0
            for i in range(n)
        ]

    # deepest_boundary[i] = smallest ℓ ∈ {2,...,L-1} where b_ℓ(i)=1
    deepest_boundary = [0] * n
    for i in range(n):
        for lv in range(2, max_level):
            if lv in boundaries and boundaries[lv][i] == 1:
                deepest_boundary[i] = lv
                break

    return CFGSample(
        string=terminals,
        length=n,
        ancestor_symbols=ancestor_symbols,
        ancestor_indices=ancestor_indices,
        boundaries=boundaries,
        deepest_boundary=deepest_boundary,
    )


# ── Synonym swap P_ℓ ──────────────────────────────────────────────────────────

def synonym_swap(
    cfg: CFG,
    level: int,
    max_attempts: int = 200,
    require_length_match: bool = True,
) -> Optional[SwapResult]:
    """
    Sample a base parse tree, pick a node at `level` whose symbol has >1 rule,
    re-expand it with a different rule, and return a SwapResult.

    With require_length_match=True:  only returns length-matched pairs (or None).
    With require_length_match=False: prefers matched; falls back to best-effort.
    Returns None if no eligible swap exists.
    """
    base_root = build_tree(cfg)
    _assign_offsets(base_root)

    candidates = [
        nd for nd in _collect_nodes_at_level(base_root, level)
        if nd.symbol in cfg.rules and len(cfg.rules[nd.symbol]) > 1
    ]
    if not candidates:
        return None

    random.shuffle(candidates)
    best_effort: Optional[tuple] = None   # (target, new_sub, new_rule_idx)

    for target in candidates:
        base_subtree_len = target.end - target.start
        alt_rules = [i for i in range(len(cfg.rules[target.symbol])) if i != target.rule_idx]

        for _ in range(max_attempts):
            new_rule_idx = random.choice(alt_rules)
            new_sub = _build_subtree(cfg, target.symbol, new_rule_idx, target.level)
            new_len = _count_terminals(new_sub)
            length_matched = (new_len == base_subtree_len)

            if not length_matched:
                if not require_length_match and best_effort is None:
                    best_effort = (target, new_sub, new_rule_idx)
                continue   # keep trying for a matched swap

            # Length-matched: assemble and return.
            return _make_result(cfg, base_root, target, new_sub, new_rule_idx, length_matched=True)

    # Exhausted all candidates — fall back to best-effort if allowed.
    if not require_length_match and best_effort is not None:
        target, new_sub, new_rule_idx = best_effort
        return _make_result(cfg, base_root, target, new_sub, new_rule_idx, length_matched=False)

    return None


def _make_result(
    cfg: CFG,
    base_root: Node,
    target: Node,
    new_sub: Node,
    new_rule_idx: int,
    length_matched: bool,
) -> SwapResult:
    swapped_root = copy.deepcopy(base_root)
    swapped_target = _find_node_by_pos(swapped_root, target.level, target.start)
    swapped_target.rule_idx = new_rule_idx
    swapped_target.children = new_sub.children
    _assign_offsets(swapped_root)

    base_sample = flatten(base_root)
    swapped_sample = flatten(swapped_root)

    base_b = target.end - 1
    swapped_b = swapped_target.end - 1

    return SwapResult(
        base=base_sample,
        swapped=swapped_sample,
        parent_symbol=target.symbol,
        span=(target.start, base_b),
        base_boundary_index=base_b,
        swapped_boundary_index=swapped_b,
        length_matched=length_matched,
    )
