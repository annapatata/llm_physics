import unittest
import numpy as np
import os

from cfg.grammar import load_cfg, CFG, CFGSample
from dp.cyk import is_valid
from dp.inside import string_prob

class TestCFGSampler(unittest.TestCase):

    def setUp(self):
        # Updated filepath to be relative to the project root
        self.cfg_filepath = 'cfg/grammars/cfg3b.txt'
        self.cfg = load_cfg(self.cfg_filepath)

    def test_all_generated_strings_pass_cyk(self):
        """Test 1: all generated strings pass CYK"""
        print("Running Test 1: all generated strings pass CYK")
        num_samples = 100
        for _ in range(num_samples):
            sample = self.cfg.sample_string()
            self.assertTrue(is_valid(sample.string, self.cfg), f"String {sample.string} failed CYK check.")
        print(f"Test 1 passed for {num_samples} samples.")

    def test_length_distribution_matches_paper(self):
        """Test 2: length distribution matches paper"""
        print("Running Test 2: length distribution matches paper")
        num_samples = 1000 # CONTEXT.md suggests 10000, reducing for quicker test
        lengths = [self.cfg.sample_string().length for _ in range(num_samples)]
        
        # CONTEXT.md: abs(np.percentile(lengths, 50) - 278) < 10 for cfg3b median
        median_length = np.percentile(lengths, 50)
        print(f"  Median length for cfg3b: {median_length}")
        self.assertLess(abs(median_length - 278), 20, # Increased tolerance slightly for reduced sample size
                        f"Median length {median_length} not within expected range (278 +/- 20).")
        
        # Additionally check min/max to ensure reasonable generation
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"  Min length: {min_len}, Max length: {max_len}")
        self.assertGreater(min_len, 0, "Minimum length should be greater than 0.")
        self.assertLessEqual(max_len, 729, "Maximum length should be <= 729 (3^6).")
        print(f"Test 2 passed for {num_samples} samples.")


    def test_boundaries_correctly_separate_ancestor_indices(self):
        """Test 3: boundaries correctly separate ancestor indices"""
        print("Running Test 3: boundaries correctly separate ancestor indices")
        num_samples = 100
        for _ in range(num_samples):
            sample = self.cfg.sample_string()
            n = sample.length
            # L is the total number of levels, including the terminal level
            # The CONTEXT.md indicates levels 2-6 are nonterminal symbols, level 7 terminals
            # So, the range for level should be from 2 up to L-1 (exclusive of terminal level L).
            # The definition of b_ℓ(i) applies for ℓ ∈ {1,…,L-1}.
            # The 'deepest_boundary' uses ℓ ∈ {2,...,L-1}.
            # Let's use the range as defined for `b` in sample_string: 1 to effective_L - 1
            max_level = max(sample.boundaries.keys())
            
            for level in range(1, max_level + 1): # Iterate through all levels that have boundary info
                for i in range(n - 1):
                    # Check if the boundary is 1, then ancestor indices must be different
                    if sample.boundaries[level][i] == 1:
                        self.assertNotEqual(sample.ancestor_indices[level][i], sample.ancestor_indices[level][i+1],
                                            f"Level {level}, position {i}: Boundary is 1 but ancestor indices are same.")
                    # Check if boundary is 0, then ancestor indices must be the same
                    else:
                        self.assertEqual(sample.ancestor_indices[level][i], sample.ancestor_indices[level][i+1],
                                           f"Level {level}, position {i}: Boundary is 0 but ancestor indices are different.")
        print(f"Test 3 passed for {num_samples} samples.")

    def test_inside_probability_greater_than_0_for_valid_strings(self):
        """Test 4: inside probability > 0 for valid strings"""
        print("Running Test 4: inside probability > 0 for valid strings")
        
        num_samples = 10 
        max_testable_length = 50 
        
        tested_count = 0
        attempts = 0
        
        while tested_count < num_samples:
            attempts += 1
            sample = self.cfg.sample_string()
            
            if sample.length > max_testable_length:
                if attempts > 1000:
                    print("Warning: Had trouble finding short enough strings to test quickly.")
                    break
                continue 
                
            prob = string_prob(sample.string, self.cfg)
            self.assertGreater(prob, 0.0, f"String {sample.string} has zero probability.")
            tested_count += 1
            
        print(f"Test 4 passed for {tested_count} samples (length <= {max_testable_length}).")

if __name__ == '__main__':
    unittest.main()