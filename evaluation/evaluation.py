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

@torch.no_grad()
def generate_autoregressive(model, prefix_tokens, max_new_tokens=256, temperature=1.0, device='cuda'):
    """
    Generates a sequence autoregressively using multinomial sampling.
    Temperature τ=1.0 is used (per the paper) to avoid greedy decoding.
    """
    model.eval()
    idx = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    for _ in range(max_new_tokens):
        logits = model(idx)
        # Scale by temperature and isolate the final token's logits
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Multinomial sampling
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        
        if next_token.item() == EOS_TOKEN:
            break
            
        idx = torch.cat((idx, next_token), dim=1)
        
    return prefix_tokens + generated

def evaluate_completion_accuracy(model, cfg, num_samples=100, prefix_len=50, device='cuda'):
    """
    Evaluates completion accuracy.
    Prefix length c=0 means full generation from scratch.
    """
    print(f"\nEvaluating Completion Accuracy (N={num_samples}, Prefix Length={prefix_len})...")
    correct_completions = 0
    
    for _ in tqdm(range(num_samples)):
        # Sample a valid string to extract a guaranteed valid prefix
        sample = cfg.sample_string()
        full_string = sample.string
        
        cut_idx = min(prefix_len, len(full_string))
        prefix = [BOS_TOKEN] + full_string[:cut_idx]
        
        # Generate completion
        completed_sequence = generate_autoregressive(model, prefix, device=device)
        
        # Strip BOS and EOS for CYK validation
        if completed_sequence[0] == BOS_TOKEN:
            completed_sequence = completed_sequence[1:]
        if completed_sequence[-1] == EOS_TOKEN:
            completed_sequence = completed_sequence[:-1]
            
        # Check validity using the CYK parser
        if is_valid(completed_sequence, cfg):
            correct_completions += 1
            
    accuracy = correct_completions / num_samples
    print(f"Completion Accuracy: {accuracy * 100:.2f}% ({correct_completions}/{num_samples})")
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

evaluate_completion_accuracy(model, my_cfg, num_samples=2000, prefix_len=0, device=device)
    
# Completion from prefix (prefix c=50)
evaluate_completion_accuracy(model, my_cfg, num_samples=2000, prefix_len=50, device=device)
