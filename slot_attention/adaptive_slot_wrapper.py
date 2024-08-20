from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from slot_attention.slot_attention import SlotAttention
from slot_attention.multi_head_slot_attention import MultiHeadSlotAttention

# functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_softmax(logits, temperature = 1.):
    dtype, size = logits.dtype, logits.shape[-1]

    assert temperature > 0

    scaled_logits = logits / temperature

    # gumbel sampling and derive one hot

    noised_logits = scaled_logits + gumbel_noise(scaled_logits)

    indices = noised_logits.argmax(dim = -1)

    hard_one_hot = F.one_hot(indices, size).type(dtype)

    # get soft for gradients

    soft = scaled_logits.softmax(dim = -1)

    # straight through

    hard_one_hot = hard_one_hot + soft - soft.detach()

    # return indices and one hot

    return hard_one_hot, indices

# wrapper

class AdaptiveSlotWrapper(Module):
    def __init__(
        self,
        slot_attn: SlotAttention | MultiHeadSlotAttention,
        temperature = 1.
    ):
        super().__init__()

        self.slot_attn = slot_attn
        dim = slot_attn.dim

        self.temperature = temperature
        self.pred_keep_slot = nn.Linear(dim, 2, bias = False)

    def forward(
        self,
        x,
        **slot_kwargs
    ):

        slots = self.slot_attn(x, **slot_kwargs)

        keep_slot_logits = self.pred_keep_slot(slots)

        keep_slots, _ = gumbel_softmax(keep_slot_logits, temperature = self.temperature)

        # just use last column for "keep" mask

        keep_slots = keep_slots[..., -1]  # Float["batch num_slots"] of {0., 1.}

        return slots, keep_slots
