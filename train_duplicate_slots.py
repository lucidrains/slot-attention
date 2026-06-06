# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "einx",
#   "vector-quantize-pytorch",
#   "wandb",
#   "fire",
#   "tqdm"
# ]
# ///

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

import fire
import wandb
from tqdm import tqdm

from einops import rearrange, einsum

from slot_attention import SlotAttention, MetaSlotAttention

# helpers

def exists(v):
    return v is not None

# positional grid

def build_grid(h, w):
    y, x = torch.meshgrid(torch.linspace(0., 1., h), torch.linspace(0., 1., w), indexing = 'ij')
    return torch.stack([y, x, 1. - y, 1. - x], dim = -1)

# toy data - colored rectangles on black background

def generate_shapes(batch_size, img_size = 16, num_shapes = 2):
    colors = torch.tensor([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.]
    ])

    images = torch.zeros(batch_size, 3, img_size, img_size)

    for b in range(batch_size):
        for k in range(num_shapes):
            color = colors[k % len(colors)]
            x, y = torch.randint(0, img_size - 4, (2,)).tolist()
            images[b, :, y:y+4, x:x+4] = rearrange(color, 'c -> c 1 1')

    return images

# encoder / decoder

class CNNEncoder(Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding = 1),
            nn.ReLU()
        )
        self.pos_emb = nn.Linear(4, dim)

    def forward(self, x):
        x = self.conv(x)
        h, w = x.shape[-2:]

        grid = build_grid(h, w).to(x.device)
        grid_emb = rearrange(self.pos_emb(grid), 'h w d -> 1 d h w')

        return rearrange(x + grid_emb, 'b d h w -> b (h w) d')

class SpatialBroadcastDecoder(Module):
    def __init__(self, dim, resolution = 16):
        super().__init__()
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.pos_emb = nn.Linear(4, dim)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, 4, 3, padding = 1)
        )

    def forward(self, slots, padding_mask = None):
        b, k, d = slots.shape
        h, w = self.resolution

        slots = rearrange(slots, 'b k d -> (b k) d 1 1').expand(-1, -1, h, w)

        grid = build_grid(h, w).to(slots.device)
        grid_emb = rearrange(self.pos_emb(grid), 'h w d -> 1 d h w')

        out = self.conv(slots + grid_emb)
        out = rearrange(out, '(b k) c h w -> b k c h w', b = b)

        rgbs, masks = out[:, :, :3], out[:, :, 3:]

        if exists(padding_mask):
            masks = masks.masked_fill(rearrange(~padding_mask, 'b k -> b k 1 1 1'), float('-inf'))

        masks = masks.softmax(dim = 1)
        return (rgbs * masks).sum(dim = 1)

# slot autoencoder

class SlotAutoEncoder(Module):
    def __init__(self, num_slots, dim, img_size, use_vq = True):
        super().__init__()
        self.use_vq = use_vq
        self.encoder = CNNEncoder(dim)

        slot_attn = SlotAttention(num_slots = num_slots, dim = dim, iters = 3)
        self.slot_attention = MetaSlotAttention(slot_attn, codebook_size = 64) if use_vq else slot_attn

        self.decoder = SpatialBroadcastDecoder(dim, resolution = img_size)

    def forward(self, x):
        enc = self.encoder(x)

        if self.use_vq:
            slots, padding_mask, vq_loss = self.slot_attention(enc)
        else:
            slots = self.slot_attention(enc)
            padding_mask = None
            vq_loss = torch.tensor(0., device = x.device)

        recon = self.decoder(slots, padding_mask = padding_mask)
        return recon, slots, padding_mask, vq_loss

# evaluation

@torch.no_grad()
def evaluate_duplicates(model, img_size, num_shapes = 2, num_eval = 10):
    model.eval()
    images = generate_shapes(num_eval, img_size, num_shapes)

    recon, slots, padding_mask, _ = model(images)
    b, k, _ = slots.shape

    active = padding_mask.sum(dim = 1).float().mean().item() if exists(padding_mask) else float(k)
    print(f'active slots: {active:.1f} (expected ~{num_shapes + 1})')

    slots_norm = F.normalize(slots, dim = -1)
    sim = einsum(slots_norm, slots_norm, 'b i d, b j d -> b i j')

    sim = sim * (1. - torch.eye(k, device = sim.device))

    if exists(padding_mask):
        valid = rearrange(padding_mask, 'b i -> b i 1') & rearrange(padding_mask, 'b j -> b 1 j')
        sim = sim * valid.float()

    num_dup = (sim.amax(dim = (-1, -2)) > 0.9).sum().item()
    print(f'batches with duplicates (cos > 0.9): {num_dup}/{b}')

    print('\npairwise sims (sample 0):')
    for i in range(k):
        for j in range(i + 1, k):
            print(f'  slot {i} vs {j}: {sim[0, i, j]:.4f}')

    return dict(active_slots = active, duplicate_batches = num_dup)

# train

def train(
    use_vq: bool = True,
    use_wandb: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_slots: int = 6,
    dim: int = 8,
    img_size: int = 16,
    num_shapes: int = 3,
    seed: int = 42
):
    torch.manual_seed(seed)
    name = 'vq-dedup' if use_vq else 'baseline'

    if use_wandb:
        wandb.init(project = 'slot-attention-shapes', name = name, reinit = True)

    print(f'generating data ({num_shapes} shapes, {num_slots} slots)')
    images = generate_shapes(500, img_size, num_shapes)
    loader = DataLoader(TensorDataset(images), batch_size = batch_size, shuffle = True)

    model = SlotAutoEncoder(num_slots, dim, img_size, use_vq = use_vq)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    model.train()

    for epoch in tqdm(range(epochs), desc = name):
        total_recon = total_vq = active_slots = 0.

        for (batch,) in loader:
            optim.zero_grad()
            recon, slots, padding_mask, vq_loss = model(batch)

            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + vq_loss
            loss.backward()
            optim.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            active_slots += padding_mask.sum(dim = 1).float().mean().item() if exists(padding_mask) else float(slots.shape[1])

        n = len(loader)
        avg_recon, avg_vq, avg_active = total_recon / n, total_vq / n, active_slots / n

        if (epoch + 1) % 10 == 0:
            tqdm.write(f'epoch {epoch + 1:3d} | recon {avg_recon:.4f} | vq {avg_vq:.4f} | active {avg_active:.1f}')

        if use_wandb:
            wandb.log(dict(recon_loss = avg_recon, vq_loss = avg_vq, active_slots = avg_active, epoch = epoch + 1))

    print()
    stats = evaluate_duplicates(model, img_size, num_shapes)

    if use_wandb:
        wandb.log(stats)
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(train)
