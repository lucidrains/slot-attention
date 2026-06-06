import torch
from slot_attention import SlotAttention
from slot_attention.ortho_slot_wrapper import OrthoSlotWrapper

def test_ortho_slots():
    slot_attn = SlotAttention(
        num_slots = 5,
        dim = 512,
        iters = 3
    )

    ortho_slots = OrthoSlotWrapper(slot_attn)

    inputs = torch.randn(2, 1024, 512)
    slots, ortho_loss = ortho_slots(inputs)

    assert slots.shape == (2, 5, 512)
    assert ortho_loss.numel() == 1
    assert ortho_loss.item() >= 0

    # check if gradients flow
    ortho_loss.backward()
