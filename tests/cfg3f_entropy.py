import os
import math
import numpy as np
from tqdm import tqdm
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg
from dp.inside import string_prob

def estimate_cfg_entropy(cfg_path, num_samples=1000, max_len=512):
    """
    Estimates the per-token entropy (in nats) of a Context-Free Grammar.
    This value represents the absolute theoretical minimum loss your model can achieve.
    """

    print(f"Loading grammar from {cfg_path}...")

    cfg= load_cfg(os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt'))
    
    total_entropy_rate = 0.0
    valid_samples = 0
    
    print(f"Sampling {num_samples} strings to estimate entropy...")
    for _ in tqdm(range(num_samples)):
        # 1. Sample a string from the true distribution
        sample = cfg.sample_string()
        x = sample.string
        n = len(x)
        
        # Skip strings that exceed your model's context window
        if n > max_len or n == 0:
            continue
            
        # 2. Compute the exact marginal probability P(x) using the Inside algorithm
        # Note: string_prob computes the probability under uniform rule choices.
        p_x = string_prob(x, cfg)
        
        if p_x <= 0.0:
            # This should theoretically never happen for a sampled string, 
            # but floating point underflow could cause it for extremely long strings.
            continue
            
        # 3. Calculate the negative log-probability (in nats)
        # Using math.log (natural logarithm) to match PyTorch's CrossEntropyLoss
        log_p = math.log(p_x)
        
        # 4. Normalize by the sequence length to get per-token entropy
        # We add 1 to the length to account for the EOS token you use in training
        per_token_entropy = -log_p / (n + 1)
        
        total_entropy_rate += per_token_entropy
        valid_samples += 1

    if valid_samples == 0:
        print("Error: No valid samples processed. Check max_len or underflow issues.")
        return None

    # Calculate the expected value
    estimated_entropy_nats = total_entropy_rate / valid_samples
    estimated_entropy_bits = estimated_entropy_nats / math.log(2)
    
    print("\n--- Entropy Estimation Results ---")
    print(f"Valid sequences processed : {valid_samples}")
    print(f"Theoretical Loss Floor    : {estimated_entropy_nats:.4f} nats/token")
    print(f"Information Entropy       : {estimated_entropy_bits:.4f} bits/token")
    print("----------------------------------")
    
    return estimated_entropy_nats

if __name__ == "__main__":
    # Adjust the path to where your cfg3f.txt is located
    cfg_file_path = os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt')
    
    if not os.path.exists(cfg_file_path):
        # Fallback if running from a different directory
        cfg_file_path = 'cfg/grammars/cfg3f.txt'
        
    estimate_cfg_entropy(cfg_file_path, num_samples=500)