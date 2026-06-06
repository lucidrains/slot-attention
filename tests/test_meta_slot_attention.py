import torch
from slot_attention import SlotAttention, MultiHeadSlotAttention, MetaSlotAttention

def test_meta_slot_attention():
    slot_attn = SlotAttention(num_slots = 5, dim = 64, iters = 3)
    meta_slot = MetaSlotAttention(slot_attn, codebook_size = 32)

    inputs = torch.randn(2, 10, 64)
    dedup_slots, mask, vq_loss = meta_slot(inputs)

    assert dedup_slots.shape == (2, 5, 64)
    assert mask.shape == (2, 5)
    assert vq_loss.numel() == 1

def test_meta_multi_head_slot_attention():
    multi_slot = MultiHeadSlotAttention(num_slots = 4, dim = 128, heads = 4, iters = 3)
    meta_slot = MetaSlotAttention(multi_slot, codebook_size = 16)

    inputs = torch.randn(2, 20, 128)
    dedup_slots, mask, vq_loss = meta_slot(inputs)

    assert dedup_slots.shape == (2, 4, 128)
    assert mask.shape == (2, 4)
    assert vq_loss.numel() == 1
