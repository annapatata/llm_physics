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
from models import build_model, load_model_weights, available_models

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
            if len(full_string) <= (MODEL_MAX_SEQ_LEN - 2): 
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
    import argparse

    grammars_dir = os.path.join(project_root, 'cfg', 'grammars')
    available = [f.replace('.txt', '') for f in os.listdir(grammars_dir) if f.endswith('.txt')]

    parser = argparse.ArgumentParser(description="Completion accuracy evaluation (Result 1)")
    parser.add_argument("--model", required=True, choices=available_models(),
                        help=f"Model architecture. Available: {', '.join(sorted(available_models()))}")
    parser.add_argument("--model_weights", required=True, help="Path to .pt model weights file")
    parser.add_argument("--cfg", required=True, choices=available,
                        help=f"Grammar to use. Available: {', '.join(sorted(available))}")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Running evaluation on {args.device}")

    my_cfg = load_cfg(os.path.join(grammars_dir, f'{args.cfg}.txt'))

    model = build_model(args.model)
    load_model_weights(model, args.model_weights)
    print("Model weights loaded successfully.")
    model.to(args.device)

    evaluate_completion_accuracy(model, my_cfg, num_samples=args.n_samples, prefix_len=0, device=args.device)
    evaluate_completion_accuracy(model, my_cfg, num_samples=args.n_samples, prefix_len=50, device=args.device)
