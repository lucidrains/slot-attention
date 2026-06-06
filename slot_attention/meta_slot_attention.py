from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

import einx
from einops import rearrange, einsum

# vq pytorch lib

from vector_quantize_pytorch import VectorQuantize

# helpers

def exists(v):
    return v is not None

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t + eps, p = 1, dim = dim)

# class

class MetaSlotAttention(Module):
    def __init__(
        self,
        slot_attention: Module,
        *,
        codebook_size: int = 1024,
        vq_decay = 0.9,
        **vq_kwargs
    ):
        super().__init__()
        self.slot_attention = slot_attention

        dim = slot_attention.dim

        vq_defaults = dict(
            decay = vq_decay,
            kmeans_init = True,
            kmeans_iters = 10,
            threshold_ema_dead_code = 2,
            **vq_kwargs
        )

        vq_defaults.update(vq_kwargs)

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = codebook_size,
            **vq_defaults
        )

    def forward(
        self,
        inputs,
        **slot_attn_kwargs
    ):
        slots = self.slot_attention(inputs, **slot_attn_kwargs)

        b, n, d, device = *slots.shape, slots.device

        # quantize to shared codebook

        _, indices, vq_loss = self.vq(slots)

        # pairwise index match -> merge duplicates by averaging

        match = einx.equal('b i, b j -> b i j', indices, indices)

        attn = l1norm(match.float())
        merged = einsum(attn, slots, 'b i j, b j d -> b i d')

        # identify and mask duplicates, keeping first occurrence

        is_duplicate = torch.tril(match, diagonal = -1).any(dim = -1)
        mask = ~is_duplicate

        # pack unique slots left, duplicates right

        sort_indices = is_duplicate.long().argsort(dim = -1, stable = True)
        batch_indices = rearrange(torch.arange(b, device = device), 'b -> b 1')

        dedup_slots = merged[batch_indices, sort_indices]
        mask = mask[batch_indices, sort_indices]

        # zero out padded positions

        dedup_slots = einx.where('b n, b n d, -> b n d', mask, dedup_slots, 0.)

        return dedup_slots, mask, vq_loss
