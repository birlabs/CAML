#!/usr/bin/env python3
"""CAML model — Population = neural population code with built-in Fourier energy.

A Population instance IS the complete model: energy, force, projection,
sampling, encoding/decoding, and roll-based bump shifting dynamics.

Concrete types: HDPopulation (1-D ring), GridPopulation (2-D torus).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

_EPS = 1e-8


# ===========================================================================
# Utilities
# ===========================================================================

def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


# ===========================================================================
# Fourier basis (internal)
# ===========================================================================

class _FourierBasis(nn.Module):
    """E(h) = e0 + Σ a_m·cos(h @ B_m + b_m),  F(h) = -dE/dh."""

    def __init__(self, dim: int, m: int):
        super().__init__()
        self.dim, self.m = int(dim), int(m)
        B = torch.randn(m, dim) / math.sqrt(max(1, dim))
        self.B = nn.Parameter(B, requires_grad=False)
        self.b = nn.Parameter(2 * math.pi * torch.rand(m), requires_grad=False)
        self.a = nn.Parameter(torch.zeros(m))
        self.e0 = nn.Parameter(torch.zeros(()))

    def phase(self, h):
        return h @ self.B.T + self.b

    def energy(self, h):
        return self.e0 + torch.cos(self.phase(h)) @ self.a

    def force(self, h):
        return (torch.sin(self.phase(h)) * self.a) @ self.B


# ===========================================================================
# Hopfield basis (classical attractor energy)
# ===========================================================================

class _HopfieldBasis(nn.Module):
    """Classical Hopfield energy: E(h) = -½ hᵀ W h + e0_h,  F(h) = W h.

    W is initialised randomly (like Fourier B) and trained via gradient descent
    to fit the target energy landscape.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        # Random symmetric init (same spirit as Fourier random B)
        W = torch.randn(dim, dim) / math.sqrt(max(1, dim))
        W = 0.5 * (W + W.T)
        self.W = nn.Parameter(W.float(), requires_grad=True)
        self.e0_h = nn.Parameter(torch.zeros(()), requires_grad=True)

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """E(h) = -½ hᵀ W h + e0_h  (per sample)."""
        Fh = h @ self.W
        return -0.5 * (h * Fh).sum(dim=-1) + self.e0_h

    def force(self, h: torch.Tensor) -> torch.Tensor:
        """F(h) = W h  (e0_h has no effect on force)."""
        return h @ (0.5 * (self.W + self.W.T))


# ===========================================================================
# Abstract Population
# ===========================================================================

class Population(nn.Module, ABC):
    """Neural population with built-in Fourier energy and roll-based bump shifting.

    Subclasses implement: encode, decode, random_latent, _dxdt,
    and the metadata properties (kind, pop_dim, latent_dim).
    project, sample_clean, target_energy, target_force are derived automatically.
    """

    def __init__(self, fourier_m: int = 8192, seed: int = 0,
                 energy_mode: str = "fourier"):
        super().__init__()
        self.fourier_m = int(fourier_m)
        self.seed = seed
        self.gain = 1.0  # attractor strength (energy/force scale), does not affect bump shifting
        self.energy_mode = energy_mode  # "fourier" | "hopfield" | "both"
        self.register_buffer("_mu", None)
        self._hopfield_basis = None     # loaded for quadratic or combined energy

    # -- metadata (abstract) -----------------------------------------------

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @property
    @abstractmethod
    def pop_dim(self) -> int: ...

    @property
    @abstractmethod
    def latent_dim(self) -> int: ...

    @property
    def E_scale(self) -> float:
        return math.sqrt(self.pop_dim)

    # -- basis -------------------------------------------------------------


    def _build_hopfield(self, device: str):
        """Construct Hopfield basis. Override in subclass."""
        raise NotImplementedError(
            f"_build_hopfield not implemented for {type(self).__name__}")

    # -- energy / force ----------------------------------------------------

    def energy(self, h):
        e = self._basis.energy(h) if self.energy_mode != "hopfield" else h.new_zeros(h.shape[0])
        if self._hopfield_basis is not None:
            e = e + self._hopfield_basis.energy(h)
        return self.gain * e

    def force(self, h):
        f = self._basis.force(h) if self.energy_mode != "hopfield" else torch.zeros_like(h)
        if self._hopfield_basis is not None:
            f = f + self._hopfield_basis.force(h)
        return self.gain * f

    def phase(self, h):
        return self._basis.phase(h)

    a  = property(lambda s: s._basis.a)
    e0 = property(lambda s: s._basis.e0)
    B  = property(lambda s: s._basis.B)
    b  = property(lambda s: s._basis.b)

    # -- manifold (derived) ------------------------------------------------

    @abstractmethod
    def encode(self, *args: torch.Tensor) -> torch.Tensor:
        """latent → population state."""
        ...

    @abstractmethod
    def decode(self, h: torch.Tensor) -> tuple:
        """population state → latent."""
        ...

    @abstractmethod
    def random_latent(self, batch: int) -> tuple:
        """Sample random latent coordinates. Returns tuple for encode(*...)."""
        ...

    def sample_clean(self, batch: int) -> torch.Tensor:
        """On-manifold states via encode(random_latent(batch))."""
        return self.encode(*self.random_latent(batch))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Decode-re-encode map (not necessarily a Euclidean nearest point)."""
        decoded = self.decode(h)
        if not isinstance(decoded, tuple):
            decoded = (decoded,)
        return self.encode(*decoded)

    # -- reference energy and force --------------------------------------------------

    def target_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Map raw activity to the encoded manifold for reference comparisons."""
        return self.project(h)

    def target_energy(self, h: torch.Tensor) -> torch.Tensor:
        return ((h - self.target_projection(h).detach()) ** 2).sum(dim=-1) / self.E_scale

    def target_force(self, h: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.target_projection(h) - h) / self.E_scale

    # -- manifold geometry -------------------------------------------------

    @abstractmethod
    def _build_mu(self, device: str) -> torch.Tensor:
        """Build preferred coordinates, shape (pop_dim,). Used when loading a checkpoint."""
        ...


    # -- dynamics (roll-based, no dense bump shifting matrices) ----------------

    @abstractmethod
    def _dxdt(self, state: torch.Tensor, velocity) -> torch.Tensor:
        """Population-specific derivative using torch.roll for bump shifting."""
        ...

    def rk4_step(self, state: torch.Tensor, velocity=None, dt: float = 0.05) -> torch.Tensor:
        if velocity is None:
            return rk4(self.force, state, dt)
        # ensure velocity has batch-dim shape for grid (broadcast if scalar)
        if velocity.dim() == 1 and self.latent_dim > 2:
            velocity = velocity.unsqueeze(-1).expand(-1, self.latent_dim - 1)
        def f(s): return self._dxdt(s, velocity)
        k1 = f(state); k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2); k4 = f(state + dt * k3)
        return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # -- serialization -----------------------------------------------------

    def state_dict(self, *args, **kwargs):
        d = self._basis.state_dict(*args, **kwargs)
        if self._hopfield_basis is not None:
            hd = self._hopfield_basis.state_dict(*args, **kwargs)
            d.update({f"_hopfield.{k}": v for k, v in hd.items()})
        return d

    def load_state_dict(self, state_dict, strict=True):
        basis_keys = {k: v for k, v in state_dict.items()
                      if not k.startswith("_hopfield.")}
        hopfield_keys = {k[len("_hopfield."):]: v
                         for k, v in state_dict.items()
                         if k.startswith("_hopfield.")}
        self._basis.load_state_dict(basis_keys, strict=strict)
        if hopfield_keys and self._hopfield_basis is not None:
            self._hopfield_basis.load_state_dict(hopfield_keys, strict=strict)


# ===========================================================================
# HD Population
# ===========================================================================

class HDPopulation(Population):
    """Head-direction ring attractor."""

    def __init__(self, n: int = 64, kappa: float = 4.0,
                 fourier_m: int = 4096, seed: int = 0):
        super().__init__(fourier_m=fourier_m, seed=seed)
        self.n, self.kappa = int(n), float(kappa)

    kind = property(lambda s: "hd")
    pop_dim = property(lambda s: s.n)
    latent_dim = property(lambda s: 2)

    # -- manifold helpers --------------------------------------------------

    def _build_mu(self, dev):
        return torch.linspace(-math.pi, math.pi, self.n + 1, device=dev)[:-1]  # (n,)

    def _von_mises(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.kappa * d.cos() - self.kappa)

    # -- encode / decode ---------------------------------------------------

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        return self._von_mises(wrap_angle(theta[:, None] - self._mu[None, :]))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.atan2((h * self._mu.sin()).sum(-1), (h * self._mu.cos()).sum(-1))

    def random_latent(self, batch: int) -> tuple:
        return (2 * math.pi * torch.rand(batch, device=self._basis.B.device) - math.pi,)


    # -- dynamics (roll-based bump shifting) -----------------------------------

    def _dxdt(self, h, velocity):
        dth = 2 * math.pi / self.n
        bump_shifting = (torch.roll(h, -1, dims=-1) - torch.roll(h, 1, dims=-1)) / (2 * dth)
        return self.force(h) - velocity[:, None] * bump_shifting


# ===========================================================================
# Grid Population
# ===========================================================================

class GridPopulation(Population):
    """Grid-cell torus attractor."""

    def __init__(self, n1: int = 64, n2: int = 64, kappa: float = 2.0,
                 grid_code: str = "hex3", fourier_m: int = 16384, seed: int = 0):
        super().__init__(fourier_m=fourier_m, seed=seed)
        self.n1, self.n2 = int(n1), int(n2)
        self.kappa = float(kappa)
        self.grid_code = str(grid_code)

    kind = property(lambda s: "grid")
    pop_dim = property(lambda s: s.n1 * s.n2)
    latent_dim = property(lambda s: 3)

    # -- manifold helpers --------------------------------------------------

    def _build_mu(self, dev):
        t1 = torch.linspace(-math.pi, math.pi, self.n1 + 1, device=dev)[:-1]
        t2 = torch.linspace(-math.pi, math.pi, self.n2 + 1, device=dev)[:-1]
        T1, T2 = torch.meshgrid(t1, t2, indexing="ij")
        return torch.stack([T1.reshape(-1), T2.reshape(-1)], dim=-1)  # (n1*n2, 2)

    def _rate(self, d1, d2):
        if self.grid_code == "product2":
            return torch.exp(self.kappa * (d1.cos() + d2.cos() - 2.0))
        return torch.exp(self.kappa * (d1.cos() + d2.cos() + wrap_angle(d2 - d1).cos() - 3.0))

    # -- encode / decode ---------------------------------------------------

    def encode(self, th1: torch.Tensor, th2: torch.Tensor) -> torch.Tensor:
        return self._rate(wrap_angle(th1[:, None] - self._mu[:, 0]),
                          wrap_angle(th2[:, None] - self._mu[:, 1]))

    def decode(self, g: torch.Tensor):
        return (torch.atan2((g * self._mu[:, 0].sin()).sum(-1), (g * self._mu[:, 0].cos()).sum(-1)),
                torch.atan2((g * self._mu[:, 1].sin()).sum(-1), (g * self._mu[:, 1].cos()).sum(-1)))

    def random_latent(self, batch: int) -> tuple:
        th = 2 * math.pi * torch.rand(batch, 2, device=self._basis.B.device) - math.pi
        return th[:, 0], th[:, 1]


    # -- dynamics (roll-based bump shifting) -----------------------------------

    def _dxdt(self, g, velocity):
        n1, n2 = self.n1, self.n2
        dth1, dth2 = 2 * math.pi / n1, 2 * math.pi / n2
        g2 = g.reshape(-1, n1, n2)
        t1 = (torch.roll(g2, -1, 1) - torch.roll(g2, 1, 1)) / (2 * dth1)
        t2 = (torch.roll(g2, -1, 2) - torch.roll(g2, 1, 2)) / (2 * dth2)
        tx = velocity[:, 0, None, None] * t1 + velocity[:, 1, None, None] * t2
        return self.force(g) - tx.reshape(g.shape[0], -1)


# ===========================================================================
# Place Population
# ===========================================================================

class PlacePopulation(Population):
    """Place-cell 2D spatial attractor with Gaussian place fields.

    Fixed bump shifting uses local finite differences between tuning centers.
    Learned nonlinear bump shifting is supplied separately by caml.bump_shifting.
    """

    def __init__(self, n: int = 256, sigma: float = 0.08,
                 arena_size: float = 1.0, wall_margin: float = 0.1,
                 min_center_dist: float = 0.05,
                 fourier_m: int = 8192, seed: int = 0,
                 energy_mode: str = "fourier"):
        super().__init__(fourier_m=fourier_m, seed=seed,
                         energy_mode=energy_mode)
        self.n = int(n)
        self.sigma = float(sigma)
        self.arena_size = float(arena_size)
        self.wall_margin = float(wall_margin)
        self.min_center_dist = float(min_center_dist)
        self.register_buffer("centers", None)
        self.register_buffer("free_xy", None)

    kind       = property(lambda s: "place")
    pop_dim    = property(lambda s: s.n)
    latent_dim = property(lambda s: 2)

    # -- centre placement --------------------------------------------------

    def _build_mu(self, dev: str) -> torch.Tensor:
        """Use the place-field centers stored in the checkpoint."""
        if self.centers is None:
            raise ValueError("Load place-field centers from a checkpoint first.")
        return self.centers.to(dev)


    # -- Hopfield ----------------------------------------------------------

    def _build_hopfield(self, device: str):
        self._hopfield_basis = _HopfieldBasis(self.n).to(device)

    # -- encode / decode --------------------------------------------------

    def _gaussian(self, d2: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * d2 / max(self.sigma ** 2, _EPS))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        d2 = ((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return self._gaussian(d2)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Fit a position inside the free map without clipping activity.

        Accept a reliable in-map log-space fit. Otherwise choose the encoding
        nearest to the raw activity on the checkpoint's free-position grid.
        Signed activities are not valid population-vector averaging weights.
        """
        if h.ndim != 2 or h.shape[1] != self.n or not torch.isfinite(h).all():
            raise ValueError("Place decoding requires finite (batch, n) activity")
        lo = self.wall_margin * self.arena_size
        hi = self.arena_size - lo
        device = h.device
        sigma2 = max(self.sigma ** 2, _EPS)
        thresh = 0.01

        c = self.centers                # (N, 2)
        c2 = (c ** 2).sum(-1)           # (N,)
        mask = h > thresh               # (B, N)

        # A_i = [-2c_ix, -2c_iy, 1],  b_i = -2σ²·log(h_i) - |c_i|²
        # A: (N, 3), b_full: (B, N)
        A = torch.cat([-2.0 * c, torch.ones(self.n, 1, device=device)], dim=-1)  # (N, 3)

        # Batched masked least squares; solves the same selected-row problem.
        # Float64 normal equations avoid a Python loop and per-cell GPU synchronizations.
        weights = mask.to(torch.float64)
        design = A.to(torch.float64)
        rhs = (-2.0 * sigma2 * h.clamp_min(_EPS).log() - c2).to(torch.float64)
        lhs = torch.einsum("bn,ni,nj->bij", weights, design, design)
        atb = torch.einsum("bn,ni,bn->bi", weights, design, rhs)
        solution, info = torch.linalg.solve_ex(lhs, atb)
        x_ls = solution[:, :2].to(h.dtype)
        pred = -0.5 * ((c[None] - x_ls[:, None])**2).sum(-1) / sigma2
        residual = (h.clamp_min(_EPS).log() - pred).square()
        rms = ((residual * mask).sum(-1) / mask.sum(-1).clamp_min(1)).sqrt()
        valid = ((mask.sum(-1) >= 4) & (info == 0) & (rms < 0.5)
                 & torch.isfinite(x_ls).all(-1)
                 & ((x_ls >= lo) & (x_ls <= hi)).all(-1))
        if not valid.all():
            positions = self.free_xy
            codes = self.encode(positions)
            # ||h||^2 is constant across candidate positions; signed h is intact.
            code_norm = codes.square().sum(-1)
            for indices in (~valid).nonzero(as_tuple=True)[0].split(1024):
                distance = code_norm - 2 * (h[indices] @ codes.T)
                x_ls[indices] = positions[distance.argmin(-1)]
        return x_ls

    def random_latent(self, batch: int) -> tuple:
        """Grid-plus-Gaussian positions conditioned on the free square.

        Rejection resamples the full position proposal; coordinates and neural
        activities are never clipped to make an out-of-map sample valid.
        """
        lo = self.wall_margin * self.arena_size
        hi = self.arena_size - lo
        idx = torch.randint(0, self.free_xy.shape[0], (batch,),
                            device=self.centers.device)
        x = self.free_xy[idx] + 0.02 * torch.randn(
            batch, 2, device=self.centers.device)
        outside = ((x < lo) | (x > hi)).any(-1)
        while outside.any():
            count = int(outside.sum())
            idx = torch.randint(0, self.free_xy.shape[0], (count,),
                                device=self.centers.device)
            x[outside] = self.free_xy[idx] + 0.02 * torch.randn(
                count, 2, device=self.centers.device)
            outside = ((x < lo) | (x > hi)).any(-1)
        return (x,)


    # -- bump shifting operators -----------------------------------------------

    def _build_bump_shifting_operators(self, dx: float = 0.05):
        """Precompute FD operators D_x, D_y from nearest-neighbor centers."""
        dev = self.centers.device; n = self.n; c = self.centers
        D_x = torch.zeros(n, n, device=dev)
        D_y = torch.zeros(n, n, device=dev)
        for i in range(n):
            ci = c[i]
            d_plus = (c - (ci + torch.tensor([dx, 0.0], device=dev))).norm(dim=-1)
            d_minus = (c - (ci - torch.tensor([dx, 0.0], device=dev))).norm(dim=-1)
            d_plus[i] = float('inf'); d_minus[i] = float('inf')
            jp = int(d_plus.argmin()); jm = int(d_minus.argmin())
            D_x[i, jp] = 1.0 / (2 * dx)
            D_x[i, jm] = -1.0 / (2 * dx)
            d_plus = (c - (ci + torch.tensor([0.0, dx], device=dev))).norm(dim=-1)
            d_minus = (c - (ci - torch.tensor([0.0, dx], device=dev))).norm(dim=-1)
            d_plus[i] = float('inf'); d_minus[i] = float('inf')
            jp = int(d_plus.argmin()); jm = int(d_minus.argmin())
            D_y[i, jp] = 1.0 / (2 * dx)
            D_y[i, jm] = -1.0 / (2 * dx)
        self.register_buffer("D_x", D_x.contiguous())
        self.register_buffer("D_y", D_y.contiguous())
        print(f"  PlacePopulation: built FD operators D_x, D_y "
              f"({n}×{n}, dx={dx})")

    def _dxdt(self, h, velocity):
        if not hasattr(self, "D_x"):
            self._build_bump_shifting_operators()
        return self.force(h) - velocity[:, :1] * (h @ self.D_x.T) - velocity[:, 1:] * (h @ self.D_y.T)

    def rk4_step(self, state: torch.Tensor, velocity=None,
                 dt: float = 0.05) -> torch.Tensor:
        """Override to handle 2-D velocity (batch, 2)."""
        if velocity is None:
            return rk4(self.force, state, dt)
        elif velocity.dim() == 1:
            velocity = velocity.unsqueeze(-1).expand(-1, 2)
        def f(s): return self._dxdt(s, velocity)
        k1 = f(state); k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2); k4 = f(state + dt * k3)
        return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4(field, h, dt):
    k1 = field(h)
    k2 = field(h + dt * k1 / 2)
    k3 = field(h + dt * k2 / 2)
    k4 = field(h + dt * k3)
    return h + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
