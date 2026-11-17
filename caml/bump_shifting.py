"""Velocity-dependent bump-shifting models and checkpoint loading."""
import torch
from torch import nn
from torch.nn import functional as F
from .io import read_checkpoint, sha256


class LinearBumpShifting(nn.Module):
    """G_j(h) = h @ W_j.T approximates d enc(x) / dx_j (positive sign)."""
    def __init__(self, n, dimensions):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimensions, n, n))
        for w in self.weight:
            nn.init.xavier_uniform_(w, gain=0.01)

    def forward(self, h, velocity):
        return sum(velocity[:, j:j+1] * F.linear(h, w)
                   for j, w in enumerate(self.weight))


class PlaceBumpShifting(nn.Module):
    def __init__(self, n, hidden=(1024, 512, 256), clip_value=50.0):
        super().__init__()
        self.clip_value = clip_value
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(n, width), nn.GELU()])
            n = width
        self.shared = nn.Sequential(*layers)
        # Output dimension is set by the input layer, independent of hidden widths.
        population_size = layers[0].in_features
        self.head_x = nn.Linear(n, population_size)
        self.head_y = nn.Linear(n, population_size)

    def forward(self, h, velocity):
        z = self.shared(h)
        gx = self.head_x(z).clamp(-self.clip_value, self.clip_value)
        gy = self.head_y(z).clamp(-self.clip_value, self.clip_value)
        return velocity[:, :1] * gx + velocity[:, 1:] * gy


def load_bump_shifting(path, pop, energy_path):
    data = read_checkpoint(path, str(pop.B.device))
    if data["energy_sha256"] != sha256(energy_path):
        raise ValueError("Bump shifting checkpoint was trained with a different energy checkpoint.")
    if data["kind"] == "place":
        model = PlaceBumpShifting(pop.pop_dim, data["hidden"], data["clip_value"])
    else:
        model = LinearBumpShifting(pop.pop_dim, 1 if pop.kind == "hd" else 2)
    model.to(pop.B.device).load_state_dict(data["state_dict"])
    model.eval().requires_grad_(False)
    return model, data


