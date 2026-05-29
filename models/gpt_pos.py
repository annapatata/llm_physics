import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionOnlyAttention(nn.Module):
    """
    Attention where scores depend only on relative position, not content.
    Score = Q_rel[delta(i,j)] · K_rel[delta(j,i)] / sqrt(head_dim)
    Values are still content-based.
    """

    def __init__(self, n_embd=768, n_head=12, max_seq_len=512):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.v_proj = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.IS_RESIDUAL_PROJECTION = True

        self.rel_qry_emb = nn.Embedding(2 * max_seq_len, self.head_dim)
        self.rel_key_emb = nn.Embedding(2 * max_seq_len, self.head_dim)
        self.max_rel_dist = max_seq_len

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def _rel_idx(self, T, device):
        positions = torch.arange(T, device=device)
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T), delta[i,j] = i-j
        return (-delta + self.max_rel_dist).clamp(0, 2 * self.max_rel_dist - 1)

    def forward(self, x, return_attention=False):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        rel_idx = self._rel_idx(T, x.device)      # (T, T)
        rel_idx_t = rel_idx.T.contiguous()

        q_rel = self.rel_qry_emb(rel_idx)          # (T, T, D) — q position embedding at delta(i,j)
        k_rel = self.rel_key_emb(rel_idx_t)        # (T, T, D) — k position embedding at delta(j,i)

        # p2p score: q_rel[i,j] · k_rel[j,i] for each head
        # q_rel/k_rel are shared across heads — broadcast over H
        att = torch.einsum('ijd,ijd->ij', q_rel, k_rel) / math.sqrt(D)  # (T, T)
        att = att.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)        # (B, H, T, T)
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
        self.attn = PositionOnlyAttention(n_embd, n_head)
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


class GPT2Position(nn.Module):
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
