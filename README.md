<img src="./diagram.jpeg" width="600px" style="border: 1px solid #ccc"></img>

## Slot Attention

Implementation of <a href="https://arxiv.org/abs/2006.15055">Slot Attention</a> from the paper 'Object-Centric Learning with Slot Attention' in Pytorch. Here is a <a href="https://www.youtube.com/watch?v=DYBmD88vpiA">video</a> that describes what this network can do.

Update: The official repository has been released <a href="https://github.com/google-research/google-research/tree/master/slot_attention">here</a>

## Install

```bash
$ pip install slot_attention
```

## Usage

```python
import torch
from slot_attention import SlotAttention

slot_attn = SlotAttention(
    num_slots = 5,
    dim = 512,
    iters = 3   # iterations of attention, defaults to 3
)

inputs = torch.randn(2, 1024, 512)
slot_attn(inputs) # (2, 5, 512)
```

After training, the network is reported to be able to generalize to slightly different number of slots (clusters). You can override the number of slots used by the `num_slots` keyword in forward.

```python
slot_attn(inputs, num_slots = 8) # (2, 8, 512)
```

To use the <a href="https://arxiv.org/abs/2406.09196">adaptive slot</a> method for generating a differentiable one hot mask for whether to use a slot, just do the following

```python
import torch
from slot_attention import MultiHeadSlotAttention, AdaptiveSlotWrapper

# define slot attention

slot_attn = MultiHeadSlotAttention(
    dim = 512,
    num_slots = 5,
    iters = 3,
)

# wrap the slot attention

adaptive_slots = AdaptiveSlotWrapper(
    slot_attn,
    temperature = 0.5 # gumbel softmax temperature
)

inputs = torch.randn(2, 1024, 512)

slots, keep_slots = adaptive_slots(inputs) # (2, 5, 512), (2, 5)

# the auxiliary loss in the paper for minimizing number of slots used for a scene would simply be

keep_aux_loss = keep_slots.sum()  # add this to your main loss with some weight
```

## Citation

```bibtex
@misc{locatello2020objectcentric,
    title   = {Object-Centric Learning with Slot Attention},
    author  = {Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
    year    = {2020},
    eprint  = {2006.15055},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{Fan2024AdaptiveSA,
    title   = {Adaptive Slot Attention: Object Discovery with Dynamic Slot Number},
    author  = {Ke Fan and Zechen Bai and Tianjun Xiao and Tong He and Max Horn and Yanwei Fu and Francesco Locatello and Zheng Zhang},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.09196},
    url     = {https://api.semanticscholar.org/CorpusID:270440447}
}
```
