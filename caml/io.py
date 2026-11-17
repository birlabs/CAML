"""Versioned checkpoints containing energy weights and population geometry."""
import hashlib
import torch

from .model import HDPopulation, GridPopulation, PlacePopulation, _FourierBasis


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path, device="cpu"):
    # All supported files contain tensors and primitive metadata only.
    return torch.load(path, map_location=device, weights_only=True)




def make_population(kind, config):
    cls = {"hd": HDPopulation, "grid": GridPopulation, "place": PlacePopulation}[kind]
    import inspect
    allowed = inspect.signature(cls).parameters
    return cls(**{k: v for k, v in config.items() if k in allowed})




def load_population(path, device="cpu"):
    data = read_checkpoint(path, device)
    kind = data.get("kind", "grid" if "grid_n1" in data or "n1" in data else "place")
    cfg = dict(data)
    for old, new in (("grid_n1", "n1"), ("grid_n2", "n2"), ("grid_kappa", "kappa")):
        if old in cfg:
            cfg[new] = cfg[old]
    pop = make_population(kind, cfg)
    # Allocate the basis, then restore its stored weights and population geometry.
    pop._basis = _FourierBasis(pop.pop_dim, pop.fourier_m).to(device)
    if kind == "place":
        if "centers" not in data or "free_xy" not in data:
            raise ValueError("Legacy place checkpoint omits tuning centers. Recover the original geometry before importing; do not resample centers.")
        pop.centers = data["centers"].to(device)
        pop.free_xy = data["free_xy"].to(device)
        pop._mu = pop.centers
    else:
        pop._mu = pop._build_mu(device)
    if pop.energy_mode in ("both", "hopfield"):
        pop._build_hopfield(device)
    pop.load_state_dict(data["state_dict"])
    pop.to(device).eval()
    pop.requires_grad_(False)
    return pop



