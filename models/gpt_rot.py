import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEAttention(nn.Module):
    def __init__(self, n_embd=768, n_head=12, max_seq_len=512):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.IS_RESIDUAL_PROJECTION = True # Tag for init scaling

        # 1. Precompute Causal Mask
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

        # 2. Precompute RoPE frequencies
        head_dim = n_embd // n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("sin_emb", emb.sin().unsqueeze(0).unsqueeze(0))
        self.register_buffer("cos_emb", emb.cos().unsqueeze(0).unsqueeze(0))

    def apply_rotary_emb(self, q, k, T):
        sin = self.sin_emb[:, :, :T, :]
        cos = self.cos_emb[:, :, :T, :]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

    def forward(self, x, return_attention=False):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply buffered RoPE
        q, k = self.apply_rotary_emb(q, k, T)

        # Apply buffered Causal Mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        att_weights = F.softmax(att, dim=-1)
        
        y = att_weights @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        output = self.c_proj(y)
        
        if return_attention:
            return output, att_weights
        return output

class GPTBlock(nn.Module):
    def __init__(self, n_embd=768, n_head=12):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.mlp[-1].IS_RESIDUAL_PROJECTION = True # Tag for init scaling

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, att_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, att_weights
            
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Rotary(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768):
        super().__init__()
        self.n_layer = n_layer # Save n_layer for the init function
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 1. Weight Tying
        self.lm_head.weight = self.wte.weight
        
        # Apply standard HF initialization with custom scaling
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard Hugging Face GPT-2 weight initialization with residual scaling."""
        if isinstance(module, nn.Linear):
            std = 0.02
            # 2. Residual scaling to prevent early instability
            if hasattr(module, 'IS_RESIDUAL_PROJECTION'):
                std *= (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, return_all_attentions=False):
        # Extract base token embeddings
        x = self.wte(idx)
        
        attentions = []
        # Pass through all transformer blocks
        for block in self.blocks:
            if return_all_attentions:
                x, att = block(x, return_attention=True)
                attentions.append(att) # Stores A_{l,h} for each layer
            else:
                x = block(x)
                
        # Final layer norm and projection to vocabulary size
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if return_all_attentions:
            return logits, attentions
        return logits