import torch
from torch.utils.data import Dataset, DataLoader
from grammar import load_cfg  # Assuming your grammar file is imported here

BOS_TOKEN = 0
EOS_TOKEN = 4
VOCAB_SIZE = 5

class TrainingCFGDataset(Dataset):
    def __init__(self, cfg, total_target_tokens=1_000_000, seq_len=512):
        self.seq_len = seq_len
        
        print(f"Generating training data... Target: ~{total_target_tokens} tokens.")
        all_tokens = []
        
        # 1. Generate a massive continuous stream of tokens
        while len(all_tokens) < total_target_tokens:
            x, _, _, _ = cfg.sample_string() 
            # We completely ignore s, p, and b here. 
            # We only care about the terminal string 'x' for training.
            
            # Sandwich the sentence with BOS and EOS and add it to the pile
            all_tokens.extend([BOS_TOKEN] + x + [EOS_TOKEN])
            
        print(f"Generated {len(all_tokens)} total tokens.")
        
        # 2. Calculate how many perfect 512-token windows we can make
        num_full_sequences = len(all_tokens) // seq_len
        
        # Drop the remainder tokens at the very end that don't fit into a 512 window
        usable_length = num_full_sequences * seq_len
        
        # 3. Convert to a PyTorch tensor and reshape it
        # .view() instantly reshapes the 1D array into a 2D matrix of shape [num_sequences, 512]
        self.data = torch.tensor(all_tokens[:usable_length], dtype=torch.long)
        self.data = self.data.view(num_full_sequences, seq_len)
        
        print(f"Created {num_full_sequences} individual sequences of length {seq_len}.")

    def __len__(self):
        # Returns the total number of 512-length sequences available
        return self.data.size(0)

    def __getitem__(self, idx):
        # Returns one single 512-length sequence
        return self.data[idx]


