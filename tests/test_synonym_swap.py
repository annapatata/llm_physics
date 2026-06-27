"""
Correctness tests for the length-matched synonym swap P_ℓ (Extension A).

Properties asserted (documentation/extension_A_synonymic_invariance.md §6, §10):
  1. flatten(build_tree(...)) reproduces a CFG.sample_string()-shaped annotation
     (lengths, keys, boundary formula) and the string is grammatical.
  2. A swap keeps the parent symbol unchanged (still "a 22" / still "an NT5").
  3. Length-matched: x and P_ℓ x have equal length; the prefix up to the boundary
     is byte-identical EXCEPT inside the swapped span; everything after the span is
     identical too.
  4. Ancestors strictly ABOVE level ℓ are unchanged; the terminals INSIDE the span
     actually change (a real swap, not a no-op).

Run:  python tests/test_synonym_swap.py
"""

import os
import sys
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg
from cfg.synonym_swap import build_tree, flatten, synonym_swap
from dp.cyk import is_valid

CFG_PATH = os.path.join(project_root, "cfg", "grammars", "cfg3b.txt")


def test_flatten_matches_sample_shape(cfg):
    tree = build_tree(cfg, cfg.root, 1)
    s = flatten(tree)
    L = max(s.ancestor_symbols)
    assert s.length == len(s.string)
    assert is_valid(s.string, cfg), "flattened tree must be a grammatical string"
    for ell in range(1, L + 1):
        assert len(s.ancestor_symbols[ell]) == s.length
        assert len(s.ancestor_indices[ell]) == s.length
    for ell in range(1, L):
        assert len(s.boundaries[ell]) == s.length
        # last position is always a boundary
        assert s.boundaries[ell][-1] == 1
    print("  [OK] flatten reproduces CFGSample shape and a grammatical string")


def test_length_matched_swaps_are_clean(cfg):
    """Levels 2-5: exact length-matched swaps are local, causal-clean, grammatical."""
    random.seed(0)
    checked = 0
    for level in (2, 3, 4, 5):
        got = 0
        for _ in range(3000):
            res = synonym_swap(cfg, level, require_length_match=True)
            if res is None:
                continue
            x, px = res.base, res.swapped
            a, b = res.span

            assert res.length_matched and x.length == px.length
            # identical outside the swapped span (prefix AND suffix byte-identical)
            assert x.string[:a] == px.string[:a], "prefix before span changed"
            assert x.string[b + 1:] == px.string[b + 1:], "suffix after span changed"
            # shared boundary index at the right edge of the span
            assert res.base_boundary_index == b == res.swapped_boundary_index
            assert x.boundaries[level][b] == 1, "boundary index is not a level-ℓ boundary"
            # parent symbol unchanged
            assert x.ancestor_symbols[level][b] == res.parent_symbol
            assert px.ancestor_symbols[level][b] == res.parent_symbol
            # ancestors strictly ABOVE ℓ are identical everywhere
            for up in range(1, level):
                assert x.ancestor_symbols[up] == px.ancestor_symbols[up], \
                    f"ancestor at level {up} changed by a level-{level} swap"
            # the swap actually changed terminals inside the span
            assert x.string[a:b + 1] != px.string[a:b + 1], "swap was a no-op"
            assert is_valid(px.string, cfg), "P_ℓ x is not grammatical"

            got += 1
            if got >= 30:
                break
        assert got > 0, f"no length-matched swaps produced at ℓ={level}"
        checked += got
        print(f"  [OK] level {level}: {got} length-matched swaps clean & grammatical")
    print(f"  [OK] {checked} length-matched swaps validated (levels 2-5)")


def test_level6_best_effort(cfg):
    """ℓ=6 (NT6) has no equal-length rules in cfg3b → only best-effort swaps exist."""
    random.seed(0)
    # strict must be impossible
    assert all(synonym_swap(cfg, 6, require_length_match=True) is None
               for _ in range(200)), "unexpected length-matched swap at ℓ=6"
    got = 0
    for _ in range(500):
        res = synonym_swap(cfg, 6, require_length_match=False)
        if res is None:
            continue
        assert not res.length_matched
        assert res.parent_symbol == res.base.ancestor_symbols[6][res.base_boundary_index]
        assert res.parent_symbol == res.swapped.ancestor_symbols[6][res.swapped_boundary_index]
        assert is_valid(res.swapped.string, cfg)
        got += 1
        if got >= 30:
            break
    assert got > 0
    print(f"  [OK] level 6: {got} best-effort swaps (length-mismatch expected, own index)")


if __name__ == "__main__":
    cfg = load_cfg(CFG_PATH)
    print("Synonym-swap correctness tests (cfg3b):")
    test_flatten_matches_sample_shape(cfg)
    test_length_matched_swaps_are_clean(cfg)
    test_level6_best_effort(cfg)
    print("\nAll synonym-swap tests passed.")
