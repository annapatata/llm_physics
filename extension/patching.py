"""
Activation patching of NT-boundary hidden states (EXTENSION.md).

Tests whether the boundary-hidden-state encoding of NT identity (Result 5) is
*causally* used for structured generation, not just decodable from it.

This file is built piece by piece:
  Piece 1 (here) — the patching hook + an identity check.
  ... later pieces add donor sampling, the generation sweep, and metrics.

──────────────────────────────────────────────────────────────────────────────
Piece 1: the mechanism
──────────────────────────────────────────────────────────────────────────────
A forward hook on `model.blocks[ℓ]` sees the residual stream *after* block ℓ —
the same tensor `probing.py` reads at `blocks[-1]`. If the hook *returns* a
tensor, PyTorch substitutes it for the block's output, so it flows into block
ℓ+1 and onward. That's all activation patching is: overwrite the residual at
chosen (position, layer), let the rest of the forward pass propagate it.

Two correctness requirements baked in here:
  - Re-apply every forward pass. During autoregressive generation the model is
    called once per new token; each call recomputes the residual from scratch,
    so the patch must be re-stamped on every call. The hook does this by living
    on the module for the whole generation, keyed on *absolute* positions that
    stay valid as the sequence grows to the right.
  - Identity safety. Patching a position with the value it already had must be a
    perfect no-op (bit-identical logits). `identity_check()` asserts this; if it
    fails, the hook is corrupting the forward pass and every downstream number
    is meaningless.
"""

import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.gpt_rot import GPT2Rotary

BOS_TOKEN = 0
EOS_TOKEN = 4


# ── The patcher ─────────────────────────────────────────────────────────────────

class ActivationPatcher:
    """
    Context manager that overwrites the residual stream after `layer` at fixed
    absolute positions with cached values, on every forward pass.

        with ActivationPatcher(model, layer=11) as patcher:
            patcher.set_patch(positions=[3, 7, 12], values=donor_states)
            out = generate(...)          # patch applied on each forward call
            patcher.clear()              # disable without removing the hook

    `values` is (n_positions, n_embd) — one residual vector per patched position,
    broadcast over the batch dimension. `positions` are indices into the *token
    sequence* (so index 0 is BOS), and must satisfy position < T on every call.
    """

    def __init__(self, model: GPT2Rotary, layer: int):
        self.model = model
        self.layer = layer
        self.positions = None   # LongTensor (n_positions,)
        self.values = None      # FloatTensor (n_positions, n_embd)
        self.enabled = False
        self._handle = None

    def _hook(self, module, inputs, output):
        # During plain generation the block returns a bare tensor; during
        # attention analysis it returns (tensor, weights). Handle both, but we
        # only ever patch the residual tensor.
        if not self.enabled or self.positions is None:
            return output
        is_tuple = isinstance(output, tuple)
        resid = output[0] if is_tuple else output  # (B, T, n_embd)

        T = resid.shape[1]
        # Positions beyond the current sequence length can't be patched yet
        # (shouldn't happen for prefix patching, but guard anyway).
        valid = self.positions < T
        pos = self.positions[valid].to(resid.device)
        vals = self.values[valid].to(resid.device, resid.dtype)

        patched = resid.clone()
        patched[:, pos, :] = vals  # broadcast (n, n_embd) over batch dim
        return (patched, *output[1:]) if is_tuple else patched

    def set_patch(self, positions, values):
        """positions: iterable of ints. values: (len(positions), n_embd) tensor."""
        self.positions = torch.as_tensor(positions, dtype=torch.long)
        self.values = torch.as_tensor(values)
        assert self.values.shape[0] == self.positions.shape[0], \
            "need one value vector per patched position"
        self.enabled = True

    def clear(self):
        """Disable patching but keep the hook registered (cheap toggle)."""
        self.enabled = False
        self.positions = None
        self.values = None

    def __enter__(self):
        self._handle = self.model.blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


# ── Residual-stream capture (to cache donor / clean states) ─────────────────────

@torch.no_grad()
def capture_residual(model: GPT2Rotary, token_ids, layer: int, device: str):
    """
    Run `token_ids` through the model and return the residual stream after
    `layer`, shape (T, n_embd) on CPU. This is the tensor a patch would overwrite
    — captured here so a donor's values can be injected into a different run.
    """
    store = {}

    def grab(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        store['resid'] = out.detach()

    handle = model.blocks[layer].register_forward_hook(grab)
    idx = torch.as_tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    model(idx)
    handle.remove()
    return store['resid'][0].cpu()  # (T, n_embd)


# ── Identity check (EXTENSION.md §6.1) ──────────────────────────────────────────

@torch.no_grad()
def identity_check(model: GPT2Rotary, token_ids, layer: int, positions, device: str):
    """
    Patch `positions` with the values they already hold and assert the logits are
    bit-identical to an unpatched forward pass. A clean pass is the only way to
    know the hook itself adds no distortion before we trust any real patch.
    """
    model.eval()
    idx = torch.as_tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 1) Unpatched logits.
    clean_logits = model(idx)

    # 2) Capture the residual we're about to re-inject unchanged.
    clean_resid = capture_residual(model, token_ids, layer, device)  # (T, n_embd)
    pos = torch.as_tensor(positions, dtype=torch.long)
    values = clean_resid[pos]  # (n_positions, n_embd)

    # 3) Patch with the identical values → must reproduce clean_logits exactly.
    with ActivationPatcher(model, layer) as patcher:
        patcher.set_patch(pos, values.to(device))
        patched_logits = model(idx)

    max_abs_diff = (clean_logits - patched_logits).abs().max().item()
    ok = torch.equal(clean_logits, patched_logits)
    print(f"  layer={layer}  positions={[int(p) for p in pos]}")
    print(f"  max abs logit diff = {max_abs_diff:.3e}   bit-identical = {ok}")
    assert max_abs_diff == 0.0, "no-op patch changed the logits -- hook is unsafe"
    print("  [OK] identity check passed: the hook is a perfect no-op when "
          "patching a value with itself.")
    return ok


if __name__ == "__main__":
    # The identity check is weight-agnostic, so a random model on CPU is enough
    # to validate the mechanism. (Real experiments load the trained checkpoint.)
    device = "cpu"
    torch.manual_seed(0)
    model = GPT2Rotary(vocab_size=5, n_layer=12, n_head=12, n_embd=768)
    model.eval().to(device)

    # A toy [BOS] + terminals sequence; positions chosen anywhere in it.
    token_ids = [BOS_TOKEN, 1, 2, 3, 1, 1, 2, 3, 2, 1, 3]
    print("Identity check (random weights, CPU) -- mechanism validation only:")
    # layer=-1 is the last block (== blocks[11]) -- the exact residual stream
    # probing.py reads from, so the patch site matches the NT5 probe site.
    identity_check(model, token_ids, layer=-1, positions=[2, 5, 8], device=device)
