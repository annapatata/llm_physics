import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader  
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg 
from dataset import InfiniteCFGDataset
from models.gpt_rot import GPT2Rotary

def get_infinite_batches(dataloader):
    """Yields batches indefinitely so we can train by steps, not epochs."""
    while True:
        for batch in dataloader:
            yield batch

def train_gpt_pretraining(model, dataloader, total_iterations=100_000, accumulation_steps=8, save_path="model.pt", device='cuda'):
    model.to(device)
    
# 1. Lower the peak learning rate for phase 2 training
    optimizer = AdamW(
        model.parameters(), 
        lr=0.00005, # Dropped significantly from 0.0003
        betas=(0.9, 0.98), 
        weight_decay=0.1
    )

    effective_optimizer_steps = total_iterations // accumulation_steps
    
    # 2. Gentle Warmup: Start at 1% of the new, lower LR to safely rebuild Adam's momentum
    warmup_steps = int(effective_optimizer_steps * 0.05)
    decay_steps = effective_optimizer_steps - warmup_steps
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, decay_scheduler], 
        milestones=[warmup_steps]
    )
    
    loss_fn = nn.CrossEntropyLoss()
    batch_iterator = get_infinite_batches(dataloader)
    
    model.train()
    progress_bar = tqdm(range(1, total_iterations + 1), desc="Training")
    
    running_loss = 0.0
    log_interval = 50 
    save_interval = 5_000 # Save checkpoint every 5,000 steps
    
    optimizer.zero_grad()
    
    for step in progress_bar:
        batch = next(batch_iterator).to(device)
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = model(inputs)
        
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Scale loss for gradient accumulation
        loss = loss_fn(logits_flat, targets_flat)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients on deep CFG trees
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Re-multiply by accumulation_steps for accurate logging
        running_loss += loss.item() * accumulation_steps
        
        # Step the optimizer only after accumulating enough gradients
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # Logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}", 
                "LR": f"{current_lr:.6f}"
            })

            print(f"Step {step}/{total_iterations} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}", flush=True)

            running_loss = 0.0
            
        # Periodic saving
        if step % save_interval == 0:
            temp_save_path = save_path.replace(".pt", f"_step{step}.pt")
            torch.save(model.state_dict(), temp_save_path)

    # Final save
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Final model saved to {save_path}")
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # 1. Load Grammar and Data
    my_cfg = load_cfg(os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt'))
    dataset = InfiniteCFGDataset(my_cfg, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=12, pin_memory=True)

    # 2. Initialize Model
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    
    # 3. Load previous weights if they exist
    # Make sure this matches the filename you saved yesterday
    checkpoint_path = os.path.join(project_root, "model.pt") 
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Loading weights to resume training...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No previous checkpoint found. Starting training from scratch.")

    final_save_path = os.path.join(project_root, "model2.pt") 

    # 4. Train the model (100k steps with accumulation = 1M effective forward passes)
    model = train_gpt_pretraining(
        model, 
        dataloader, 
        total_iterations=10_000, 
        accumulation_steps=8, 
        save_path=final_save_path,
        device=device
    )