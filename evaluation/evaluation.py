import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Adjust these imports if your folder structure differs
from cfg.grammar import load_cfg
from dp.cyk import is_valid
from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0
EOS_TOKEN = 4

MODEL_MAX_SEQ_LEN = 512  # must match RoPEAttention(max_seq_len=...) in gpt_rot.py
CFG_MAX_STRING_LEN = 729  # 3^6, theoretical maximum for these CFGs

@torch.no_grad()
def generate_autoregressive(model, prefix_tokens, temperature=1.0, device='cuda'):
    """
    Generates a sequence autoregressively using multinomial sampling.
    Temperature τ=1.0 is used (per the paper) to avoid greedy decoding.

    max_new_tokens is derived from the model's context window so that idx
    never exceeds MODEL_MAX_SEQ_LEN, preventing RoPE/mask buffer overflow.
    The CFG can produce strings up to 729 tokens; 256 would silently truncate
    them and cause valid long strings to fail the CYK check.
    """
    model.eval()
    idx = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Budget: how many new tokens fit before hitting the context-window ceiling
    max_new_tokens = MODEL_MAX_SEQ_LEN - len(prefix_tokens)

    generated = []
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())

        if next_token.item() == EOS_TOKEN:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return prefix_tokens + generated

def evaluate_completion_accuracy(model, cfg, num_samples=100, prefix_len=50, device='cuda'):
    print(f"\nEvaluating Completion Accuracy (N={num_samples}, Prefix Length={prefix_len})...")
    correct_completions = 0
    truncations = 0
    
    for i in tqdm(range(num_samples)):
        # 1. Force the grammar to give us a sequence that fits comfortably within the 512 limit
        while True:
            sample = cfg.sample_string()
            full_string = sample.string
            if len(full_string) <= 100:
                break
                
        cut_idx = min(prefix_len, len(full_string))
        prefix = [BOS_TOKEN] + full_string[:cut_idx]
        
        # 2. Generate completion
        completed_sequence = generate_autoregressive(model, prefix, device=device)
        
        # 3. Diagnostic Check: Did it hit the 512 wall?
        hit_wall = len(completed_sequence) == MODEL_MAX_SEQ_LEN
        if hit_wall:
            truncations += 1
            
        # Print the first 3 attempts to visually inspect what the model is outputting
        if i < 3:
            print(f"\n--- Debug Sample {i+1} ---")
            print(f"Target GT Length: {len(full_string)}")
            print(f"Generated Length: {len(completed_sequence)}")
            print(f"Hit 512 Wall?   : {hit_wall}")
            if hit_wall:
                print(f"Last 10 tokens  : {completed_sequence[-10:]} (Notice no EOS token '4')")
            print("------------------------")
            
        # Strip special tokens safely from anywhere in the string
        content = [t for t in completed_sequence if t != BOS_TOKEN and t != EOS_TOKEN]
            
        if len(content) > 0 and is_valid(content, cfg):
            correct_completions += 1
            
    accuracy = correct_completions / num_samples
    print(f"Completion Accuracy: {accuracy * 100:.2f}% ({correct_completions}/{num_samples})")
    
    if truncations > 0:
        print(f"\nWARNING: {truncations}/{num_samples} sequences hit the 512-token limit.")
        print("If truncations are high despite filtering for GT < 100, your model has NOT grokked the grammar.")
        print("It is babbling local patterns without closing the global brackets. You must train for more steps or use a larger batch size.")
        
    return accuracy


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on {device}")
    
    # 1. Initialize CFG
    cfg_path = os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt')
    my_cfg = load_cfg(cfg_path)
    
    # 2. Initialize Model (ensure vocab_size matches training: 5)
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    
    # 3. Load Weights
    weights_path = os.path.join(project_root, 'model.pt')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Weights not found at {weights_path}. Evaluating UNTRAINED model.")
        
    model.to(device)

    # Result 1 — Completion Accuracy (paper uses 20,000 samples; use fewer for quick checks)
    # c=0: full generation from scratch
    evaluate_completion_accuracy(model, my_cfg, num_samples=200, prefix_len=0, device=device)

    # c=50: completion from a 50-token prefix
    evaluate_completion_accuracy(model, my_cfg, num_samples=200, prefix_len=50, device=device)
