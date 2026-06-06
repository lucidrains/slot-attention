import torch
from torch.nn import Module

from einops import rearrange, einsum

# helpers

def exists(v):
    return v is not None

# class

class OrthoSlotWrapper(Module):
    def __init__(
        self,
        slot_attention: Module
    ):
        super().__init__()
        self.slot_attention = slot_attention
        self.register_buffer('mask', None, persistent = False)

    def forward(
        self,
        *args,
        **kwargs
    ):
        out = self.slot_attention(*args, **kwargs)

        is_tuple = isinstance(out, tuple)
        slots = out[0] if is_tuple else out

        b, n, d, device = *slots.shape, slots.device

        # lazily instantiate mask if not present or shape mismatch

        if not exists(self.mask) or self.mask.shape[-1] != n:
            mask = ~torch.eye(n, dtype = torch.bool, device = device)
            self.register_buffer('mask', mask, persistent = False)

        # center the representations

        mean = slots.mean(dim = 1, keepdim = True)
        centered_slots = slots - mean

        # compute pairwise inner product

        sim = einsum(centered_slots, centered_slots, 'b i d, b j d -> b i j')

        # mask out diagonal and compute loss

        off_diag_sim = sim[:, self.mask]
        ortho_loss = off_diag_sim.pow(2).mean()

        if is_tuple:
            return (*out, ortho_loss)

        return slots, ortho_loss
