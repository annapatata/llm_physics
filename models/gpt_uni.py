import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformAttention(nn.Module):
    """
    Fixed uniform attention — no Q or K matrices, weights never updated.
    Head h (0-indexed) uniformly averages the previous 2^h tokens (including current).
    Window sizes: 1, 2, 4, 8, 16, 32, 64, 128.
    """

    N_HEADS = 8
    WINDOWS = [2 ** h for h in range(N_HEADS)]  # [1, 2, 4, ..., 128]

    def __init__(self, n_embd=1024, max_seq_len=512):
        super().__init__()
        self.n_embd = n_embd

        # Precompute fixed attention masks: (n_heads, max_seq_len, max_seq_len)
        masks = torch.zeros(self.N_HEADS, max_seq_len, max_seq_len)
        for h, w in enumerate(self.WINDOWS):
            for i in range(max_seq_len):
                start = max(0, i - w + 1)
                masks[h, i, start:i + 1] = 1.0 / (i - start + 1)
        self.register_buffer("attn_weights", masks)  # fixed, never updated

    def forward(self, x, return_attention=False):
        B, T, C = x.size()

        weights = self.attn_weights[:, :T, :T]           # (H, T, T)
        weights = weights.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, T, T)

        # Each head operates on the full hidden state, then we average across heads
        x_heads = x.unsqueeze(1).expand(-1, self.N_HEADS, -1, -1)  # (B, H, T, C)
        out = (weights.unsqueeze(-1) * x_heads).sum(dim=2)          # (B, H, C) — weighted avg per position...

        # Actually: for each position i, weighted sum over T
        # weights: (B, H, T, T) — weights[b,h,i,k] = weight from i to k
        out = torch.einsum('bhtk,bkc->bhtc', weights, x)  # (B, H, T, C)
        out = out.mean(dim=1)                              # (B, T, C) — average across heads

        if return_attention:
            return out, weights
        return out


class GPTUniBlock(nn.Module):
    def __init__(self, n_embd=1024):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = UniformAttention(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.mlp[-1].IS_RESIDUAL_PROJECTION = True

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, att_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, att_weights

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Uniform(nn.Module):
    """
    GPT with fixed uniform multi-scale attention and no learned Q/K.
    Uses 1024 dimensions (vs 768) to match parameter count of other models.
    All learning happens in the MLP layers.
    """

    def __init__(self, vocab_size, n_layer=12, n_embd=1024):
        super().__init__()
        self.n_layer = n_layer
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([GPTUniBlock(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
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
        x = self.wte(idx)

        attentions = []
        for block in self.blocks:
            if return_all_attentions:
                x, att = block(x, return_attention=True)
                attentions.append(att)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_all_attentions:
            return logits, attentions
        return logits
