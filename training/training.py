import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader  
from tqdm import tqdm

# Add the project root to the Python path so it can find 'cfg' and 'models'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 1. Import from the cfg folder
from cfg.grammar import load_cfg 

# 2. Import the NEW iterable dataset
from dataset import InfiniteCFGDataset

# 3. Import the model from the models folder
from models.gpt_rot import GPT2Rotary

def get_infinite_batches(dataloader):
    """Yields batches indefinitely so we can train by steps, not epochs."""
    while True:
        for batch in dataloader:
            yield batch

def train_gpt_pretraining(model, dataloader, total_iterations=100_000, device='cuda'):
    model.to(device)
    
    # 1. Optimizer strictly following paper's hyperparameters
    optimizer = AdamW(
        model.parameters(), 
        lr=0.0003, 
        betas=(0.9, 0.98), 
        weight_decay=0.1
    )
    
    # 2. Linear Learning Rate Decay 
    # Starts at 1.0 * lr (0.0003) and decays linearly to 0.0 * lr over total_iters
    scheduler = LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.0, 
        total_iters=total_iterations
    )
    
    loss_fn = nn.CrossEntropyLoss()
    batch_iterator = get_infinite_batches(dataloader)
    
    model.train()
    
    # Setup tqdm for 100,000 steps
    progress_bar = tqdm(range(1, total_iterations + 1), desc="Pre-training")
    
    running_loss = 0.0
    log_interval = 500 # Print average loss every 500 steps
    
    for step in progress_bar:
        batch = next(batch_iterator).to(device)
        
        # Slicing for Next-Token Prediction
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward Pass
        logits = model(inputs)
        
        # Reshape for CrossEntropyLoss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Calculate Loss
        loss = loss_fn(logits_flat, targets_flat)
        
        # Backward Pass
        loss.backward()
        
        # Update weights and learning rate
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        # Logging & Progress Bar Updates
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            current_lr = scheduler.get_last_lr()[0]
            
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}", 
                "LR": f"{current_lr:.6f}"
            })
            running_loss = 0.0

    print("Training complete! 100,000 iterations finished.")
    return model




if __name__ == "__main__":
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # 2. Load Grammar and Data
    my_cfg = load_cfg(os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt'))

    # Use the new Infinite dataset (no total_target_tokens needed)
    dataset = InfiniteCFGDataset(my_cfg, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=96, pin_memory=True)

    # 3. Initialize Model
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)

    # 4. Run the 1,000 step smoke test and capture the returned model
    model = train_gpt_pretraining(model, dataloader, total_iterations=1_000, device=device)
    
    # 5. Save the weights so they aren't deleted from memory
    torch.save(model.state_dict(), "gpt2_cfg3f_1k_smoketest.pt")
    print("Smoke test complete. Model saved successfully!")