import torch
from torch.utils.data import IterableDataset, DataLoader

BOS_TOKEN = 0
EOS_TOKEN = 4

class InfiniteCFGDataset(IterableDataset):
    def __init__(self, cfg, seq_len=512):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = seq_len + 1

    def __iter__(self):
        token_buffer = []
        
        while True:
            # 1. Capture the entire CFGSample object
            sample = self.cfg.sample_string()
            
            # 2. Extract the terminal string
            x = sample.string 
            
            # Sandwich with BOS and EOS and add to the running buffer
            token_buffer.extend([BOS_TOKEN] + x + [EOS_TOKEN])
            
            while len(token_buffer) >= self.chunk_size:
                chunk = token_buffer[:self.chunk_size]
                token_buffer = token_buffer[self.chunk_size:]
                
                yield torch.tensor(chunk, dtype=torch.long)