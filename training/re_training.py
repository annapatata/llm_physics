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
    while True:
        for batch in dataloader:
            yield batch

def train_gpt_pretraining(
    model, 
    dataloader, 
    total_iterations=20_000, 
    accumulation_steps=16, 
    full_save_path="full_checkpoint.pt", 
    device='cuda'
):
    model.to(device)
    
    # 1. Final Annealing Learning Rate
    optimizer = AdamW(model.parameters(), lr=0.000005, betas=(0.9, 0.98), weight_decay=0.1)

    # 2. Scheduler Math
    effective_steps = total_iterations // accumulation_steps
    # 10% warmup is crucial here to let AdamW safely rebuild its momentum from scratch
    warmup_steps = int(effective_steps * 0.10) 
    
    # Starts at 10% of 5e-6 (which is 5e-7), giving a very gentle on-ramp
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=effective_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
    
    start_step = 1

    # 3. Smart Resume Logic
    if os.path.exists(full_save_path):
        print(f"Restoring full session from {full_save_path}...")
        checkpoint = torch.load(full_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Optimizer and Scheduler states fully restored!")
    else:
        print("No full checkpoint found. Starting optimizer and scheduler from scratch based on loaded model weights.")

    loss_fn = nn.CrossEntropyLoss()
    batch_iterator = get_infinite_batches(dataloader)
    
    model.train()
    progress_bar = tqdm(range(start_step, total_iterations + 1), desc="Annealing Phase")
    
    running_loss = 0.0
    optimizer.zero_grad()
    
    for step in progress_bar:
        batch = next(batch_iterator).to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) / accumulation_steps
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        running_loss += loss.item() * accumulation_steps
        
        if step % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if step % 100 == 0:
            avg_loss = running_loss / 100
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.8f}"})
            print(f"Step {step}/{total_iterations} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}", flush=True)

            running_loss = 0.0

        # Periodic full state save
        if step % 5000 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, full_save_path)

    # Final Weights-Only Save (for evaluation)
    final_weights_path = os.path.join(project_root, "model_final_annealed.pt")
    torch.save(model.state_dict(), final_weights_path)
    print(f"Training complete! Final weights saved to {final_weights_path}")
    
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    my_cfg = load_cfg(os.path.join(project_root, 'cfg', 'grammars', 'cfg3f.txt'))
    dataset = InfiniteCFGDataset(my_cfg, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=12, pin_memory=True)

    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    
    full_checkpoint_path = os.path.join(project_root, "full_checkpoint.pt")
    best_weights_path = os.path.join(project_root, "model.pt")
    
    # 4. Initialization Routing
    if not os.path.exists(full_checkpoint_path):
        if os.path.exists(best_weights_path):
            print(f"Loading weights from {best_weights_path} to begin annealing...")
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
        else:
            print(f"WARNING: {best_weights_path} not found. Starting from random initialization.")

    train_gpt_pretraining(
        model, 
        dataloader, 
        total_iterations=20_000, 
        accumulation_steps=16, 
        full_save_path=full_checkpoint_path,
        device=device
    )