import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DisentangledRelativeAttention(nn.Module):
    """
    DeBERTa-style disentangled relative attention.
    Attention score = (c2c + c2p + p2c) / sqrt(3 * head_dim)

    c2c: Q_content · K_content  (standard content attention)
    c2p: Q_content · K_rel      (content query attends to relative position key)
    p2c: Q_rel     · K_content  (relative position query attends to content key)
    """

    def __init__(self, n_embd=768, n_head=12, max_seq_len=512):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.max_rel_dist = max_seq_len

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.IS_RESIDUAL_PROJECTION = True

        # Relative position embeddings: one table for keys, one for queries
        # Size: (2 * max_rel_dist, head_dim) — covers distances [-max, +max]
        self.rel_key_emb = nn.Embedding(2 * max_seq_len, self.head_dim)
        self.rel_qry_emb = nn.Embedding(2 * max_seq_len, self.head_dim)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def _rel_idx(self, T, device):
        # Returns (T, T) tensor of relative position indices into the embedding table.
        # rel_idx[i, j] = clip(j - i + max_rel_dist, 0, 2*max_rel_dist - 1)
        positions = torch.arange(T, device=device)
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T), delta[i,j] = i-j
        idx = (-delta + self.max_rel_dist).clamp(0, 2 * self.max_rel_dist - 1)
        return idx

    def forward(self, x, return_attention=False):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        qkv = self.c_attn(x)
        q_c, k_c, v = qkv.split(C, dim=2)

        q_c = q_c.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k_c = k_c.view(B, T, H, D).transpose(1, 2)
        v   = v.view(B, T, H, D).transpose(1, 2)

        rel_idx = self._rel_idx(T, x.device)  # (T, T)

        # c2c: (B, H, T, T)
        c2c = q_c @ k_c.transpose(-2, -1)

        # c2p: Q_content[i] · K_rel[delta(i,j)]
        # k_rel: (T, T, D) → need (H, T, T) after dot with q_c
        k_rel = self.rel_key_emb(rel_idx)           # (T, T, D)
        k_rel = k_rel.unsqueeze(0).expand(H, -1, -1, -1)  # (H, T, T, D)
        c2p = torch.einsum('bhtd,htsd->bhts', q_c, k_rel)  # (B, H, T, T) — q[i]·k_rel[i,j]

        # p2c: Q_rel[delta(j,i)] · K_content[j]
        # Note: p2c uses the transposed delta: delta(j,i) = -delta(i,j)
        rel_idx_t = self._rel_idx(T, x.device).T.contiguous()  # (T, T), delta(j,i)
        q_rel = self.rel_qry_emb(rel_idx_t)         # (T, T, D)
        q_rel = q_rel.unsqueeze(0).expand(H, -1, -1, -1)  # (H, T, T, D)
        p2c = torch.einsum('bhtd,htsd->bhts', k_c, q_rel).transpose(-2, -1)  # (B, H, T, T)

        scale = math.sqrt(3 * D)
        att = (c2c + c2p + p2c) / scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
        self.attn = DisentangledRelativeAttention(n_embd, n_head)
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


class GPT2Relative(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768):
        super().__init__()
        self.n_layer = n_layer
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([GPTBlock(n_embd, n_head) for _ in range(n_layer)])
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
