from torch import nn
import torch

class WeightedAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

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

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim = self.softmax_dim) + self.eps
        attn = attn / attn.sum(dim = self.weighted_mean_dim, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        return updates

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRU(dim, dim)
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs, _ = self.gru(
            updates.reshape(1, -1, d),
            inputs.reshape(1, -1, d)
        )
        return inputs.reshape(b, -1, d)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

class SlotAttentionExperimental(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps))
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps, softmax_dim = 2, weighted_mean_dim = 1))
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)
            slots = self.slots_ff(slots)

            inputs = self.inputs_to_slots_attn(inputs, slots)
            inputs = self.inputs_ff(inputs)

        return slots, inputs
