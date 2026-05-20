"""
Greedy-decoding twin of evaluation/evaluation.py.

Same completion-accuracy pipeline as the original, but each next-token choice
is argmax over the logits instead of multinomial sampling at temperature 1.

NOTE  Allen-Zhu & Li 2023 report Result 1 with multinomial sampling at τ = 1.
      Greedy decoding usually OVERSTATES accuracy because the model never
      samples from low-probability tails — useful as a complementary "best-case"
      check, not as a replacement for the paper's official metric.
"""

import os
import sys
import torch
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
def generate_greedy(model, prefix_tokens, device='cuda'):
    """
    Generates a sequence autoregressively using GREEDY decoding (argmax).
    No temperature, no sampling — at each step we pick the most probable token.

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
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        if next_token.item() == EOS_TOKEN:
            break

        idx = torch.cat((idx, next_token), dim=1)

    return prefix_tokens + generated

def evaluate_completion_accuracy_greedy(model, cfg, num_samples=100, prefix_len=50, device='cuda'):
    print(f"\nEvaluating Completion Accuracy [GREEDY] (N={num_samples}, Prefix Length={prefix_len})...")
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

        # 2. Generate completion (greedy)
        completed_sequence = generate_greedy(model, prefix, device=device)

        # 3. Diagnostic Check: Did it hit the 512 wall?
        hit_wall = len(completed_sequence) == MODEL_MAX_SEQ_LEN
        if hit_wall:
            truncations += 1

        # Print the first 3 attempts to visually inspect what the model is outputting
        if i < 3:
            print(f"\n--- Debug Sample {i+1} [greedy] ---")
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
    print(f"Completion Accuracy [GREEDY]: {accuracy * 100:.2f}% ({correct_completions}/{num_samples})")

    if truncations > 0:
        print(f"\nWARNING: {truncations}/{num_samples} sequences hit the 512-token limit.")
        print("If truncations are high despite filtering for GT < 100, your model has NOT grokked the grammar.")
        print("It is babbling local patterns without closing the global brackets. You must train for more steps or use a larger batch size.")
        print("(Note: greedy decoding can also induce repetitive loops that hit the wall — compare against the multinomial run.)")

    return accuracy


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running GREEDY evaluation on {device}")
    print("Reminder: paper Result 1 uses multinomial sampling at τ=1; greedy is a "
          "complementary best-case probe and is NOT the official metric.")

    # 1. Initialize CFG
    cfg_path = os.path.join(project_root, 'cfg', 'grammars', 'cfg3b.txt')
    my_cfg = load_cfg(cfg_path)

    # 2. Initialize Model (ensure vocab_size matches training: 5)
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)

    # 3. Load Weights
    weights_path = os.path.join(project_root, 'model.pt') # Or gpt_checkpoint_step_100000.pt
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)

        # Handle both raw state_dict and checkpoint dictionary formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Model weights loaded successfully.")

    # 4. Move model to the evaluation device. Without this the prefix tensor
    #    (created on `device`) and the model weights (on cpu) live on different
    #    devices and the embedding lookup raises a device-mismatch RuntimeError.
    model.to(device)

    # Result 1 — Completion Accuracy via GREEDY decoding (paper uses τ=1 multinomial)
    # c=0: full generation from scratch
    evaluate_completion_accuracy_greedy(model, my_cfg, num_samples=200, prefix_len=0, device=device)

    # c=50: completion from a 50-token prefix
    evaluate_completion_accuracy_greedy(model, my_cfg, num_samples=200, prefix_len=50, device=device)
