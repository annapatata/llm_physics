import sys
import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cfg.grammar import load_cfg
from dataset import InfiniteCFGDataset
from models import build_model, load_model_weights, available_models

CHECKPOINTS_DIR = "checkpoints"


def save_checkpoint(model, optimizer, scheduler, step, output_name):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tracker_path = os.path.join(CHECKPOINTS_DIR, "last_checkpoint.json")

    # Delete previous checkpoint
    if os.path.exists(tracker_path):
        with open(tracker_path) as f:
            prev = json.load(f).get("last_checkpoint")
        if prev and os.path.exists(prev):
            os.remove(prev)

    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{output_name}_checkpoint_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)

    with open(tracker_path, 'w') as f:
        json.dump({"last_checkpoint": checkpoint_path}, f, indent=2)

    print(f"\n[Checkpoint] Saved to {checkpoint_path}")


def get_infinite_batches(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train_gpt_pretraining(model, dataloader, output_name, total_iterations=10_000, accumulation_steps=8, device='cuda'):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.98), weight_decay=0.1)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iterations)

    loss_fn = nn.CrossEntropyLoss()
    batch_iterator = get_infinite_batches(dataloader)

    model.train()
    progress_bar = tqdm(range(1, total_iterations + 1), desc="Pre-training")
    running_loss = 0.0
    log_interval = 10

    optimizer.zero_grad()

    for step in progress_bar:
        step_loss = 0.0

        for _ in range(accumulation_steps):
            batch = next(batch_iterator).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            step_loss += scaled_loss.item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += step_loss

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{current_lr:.6f}"})
            print(f"Step {step}/{total_iterations} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}", flush=True)
            running_loss = 0.0

        if step % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, step, output_name)

    print(f"Training complete! {total_iterations} iterations finished.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT CFG Pretraining")
    grammars_dir = os.path.join(project_root, 'cfg', 'grammars')
    available_grammars = [f.replace('.txt', '') for f in os.listdir(grammars_dir) if f.endswith('.txt')]
    parser.add_argument("--model", required=True, choices=available_models(),
                        help=f"Model architecture. Available: {', '.join(sorted(available_models()))}")
    parser.add_argument("--model_weights", default=None,
                        help="Path to .pt weights file to resume training from (optional)")
    parser.add_argument("--n_iters", type=int, default=6_500, help="Number of training iterations")
    parser.add_argument("--output", default="my_model", help="Base name for checkpoint and final weights files")
    parser.add_argument("--cfg", required=True, choices=available_grammars,
                        help=f"Grammar to use. Available: {', '.join(sorted(available_grammars))}")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    cfg_path = os.path.join(grammars_dir, f'{args.cfg}.txt')
    my_cfg = load_cfg(cfg_path)

    dataset = InfiniteCFGDataset(my_cfg, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=12, pin_memory=True)

    model = build_model(args.model)
    if args.model_weights is not None:
        load_model_weights(model, args.model_weights)
        print(f"Loaded weights from {args.model_weights}")

    model = train_gpt_pretraining(
        model,
        dataloader,
        output_name=args.output,
        total_iterations=args.n_iters,
        accumulation_steps=8,
        device=device,
    )

    final_path = f"{args.output}_weights.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")