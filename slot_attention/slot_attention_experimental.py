import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, init

from einops import einsum

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t + eps, p = 1, dim = dim)

# weighted attention with configurable softmax / mean dims

class WeightedAttention(Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

    def forward(self, inputs, context):
        inputs = self.norm_input(inputs)
        context = self.norm_context(context)

        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = einsum(q, k, 'b i d, b j d -> b i j') * self.scale
        attn = dots.softmax(dim = self.softmax_dim)
        attn = l1norm(attn, dim = self.weighted_mean_dim, eps = self.eps)

        updates = einsum(v, attn, 'b j d, b i j -> b i d')
        return updates

# residual wrappers

class GatedResidual(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.fn = fn

    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )
        return inputs.reshape(b, -1, d)

# feedforward

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

# class

class SlotAttentionExperimental(Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps))
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps, softmax_dim = 2, weighted_mean_dim = 1))
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = default(num_slots, self.num_slots)

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)
            slots = self.slots_ff(slots)

            inputs = self.inputs_to_slots_attn(inputs, slots)
            inputs = self.inputs_ff(inputs)

        return slots, inputs
