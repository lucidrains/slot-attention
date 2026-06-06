import torch
from slot_attention import SlotAttention, MultiHeadSlotAttention

def test_slot_attention():
    slot_attn = SlotAttention(num_slots = 5, dim = 64, iters = 3)
    inputs = torch.randn(2, 10, 64)
    slots = slot_attn(inputs)
    assert slots.shape == (2, 5, 64)

def test_slot_attention_override_num_slots():
    slot_attn = SlotAttention(num_slots = 5, dim = 64, iters = 3)
    inputs = torch.randn(2, 10, 64)
    slots = slot_attn(inputs, num_slots = 8)
    assert slots.shape == (2, 8, 64)

def test_multi_head_slot_attention():
    slot_attn = MultiHeadSlotAttention(num_slots = 5, dim = 64, heads = 4, iters = 3)
    inputs = torch.randn(2, 10, 64)
    slots = slot_attn(inputs)
    assert slots.shape == (2, 5, 64)

def test_multi_head_slot_attention_override_num_slots():
    slot_attn = MultiHeadSlotAttention(num_slots = 5, dim = 64, heads = 4, iters = 3)
    inputs = torch.randn(2, 10, 64)
    slots = slot_attn(inputs, num_slots = 8)
    assert slots.shape == (2, 8, 64)
