"""Checkpoint evaluation for the main recovery and path-integration results."""
import math

import numpy as np
import torch

from .bump_shifting import load_bump_shifting
from .io import load_population
from .model import rk4, wrap_angle


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("Non-finite evaluation result")
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=0)),
            "n": int(values.size)}


def _finite(h):
    if not torch.isfinite(h).all():
        raise FloatingPointError("Non-finite population activity")


@torch.no_grad()
def recovery(ckpt, *, device="cuda", points=None, runs=5, steps=None,
             dt=0.05, amplitudes=(0.1, 0.5), seed=104):
    """Table 1: mean and SD of per-state normalized population RMSE."""
    pop = load_population(ckpt, device)
    if pop.kind not in ("hd", "grid"):
        raise ValueError("Use an HD or Grid checkpoint")
    points = points if points is not None else (2000 if pop.kind == "hd" else 500)
    steps = steps if steps is not None else (500 if pop.kind == "hd" else 4000)
    records = []
    for amplitude in amplitudes:
        errors = []
        for evaluation_seed in range(seed, seed + runs):
            torch.manual_seed(evaluation_seed)
            clean = pop.sample_clean(points)
            rms = clean.square().mean(-1).sqrt().mean()
            h = clean + amplitude * rms * torch.randn_like(clean)
            for _ in range(steps):
                h = pop.rk4_step(h, dt=dt)
            _finite(h)
            error = (h.double() - clean.double()).square().mean(-1).sqrt() / rms.double()
            errors.append(error.cpu().numpy())
            print(f"{pop.kind.upper()} noise={amplitude:g}: {len(errors)*points}/{runs*points} states", flush=True)
        records.append({"population": pop.kind, "noise_sd": float(amplitude),
                        **_summary(np.concatenate(errors))})
    return records


def _velocity_inputs(amplitudes, runs, steps, dt, dims, seed):
    trajectories = []
    for amplitude in amplitudes:
        for evaluation_seed in range(seed, seed + runs):
            gx = torch.Generator().manual_seed(evaluation_seed)
            gy = torch.Generator().manual_seed(evaluation_seed + 10000)
            vx, vy, values = 0.0, 0.0, []
            for i in range(steps + 1):
                scale = 0.3 * math.sqrt(dt if i else 0)
                vx = np.clip(vx + scale * float(torch.randn(1, generator=gx)), -amplitude, amplitude)
                if dims == 2:
                    vy = np.clip(vy + scale * float(torch.randn(1, generator=gy)), -amplitude, amplitude)
                values.append([vx] if dims == 1 else [vx, vy])
            trajectories.append(values)
    return torch.tensor(trajectories, dtype=torch.float32).transpose(0, 1)


@torch.no_grad()
def path_integration(ckpt, shifting_ckpt, *, device="cuda", duration=None,
                     runs=100, dt=0.05, amplitudes=(0.5, 1.0), seed=0):
    """Table 2: paired fixed/learned tracking, with one velocity update per step."""
    pop = load_population(ckpt, device)
    if pop.kind not in ("hd", "grid"):
        raise ValueError("Use an HD or Grid checkpoint")
    shifting, metadata = load_bump_shifting(shifting_ckpt, pop, ckpt)
    if metadata["config"]["gain"] != pop.gain:
        raise ValueError("Energy and bump-shifting gains differ")
    duration = duration if duration is not None else (20 if pop.kind == "hd" else 40)
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("Duration must be a positive integer multiple of dt")
    dims = 1 if pop.kind == "hd" else 2
    velocity = _velocity_inputs(amplitudes, runs, steps, dt, dims, seed).to(device)
    truth = torch.zeros(velocity.shape[1], dims, device=device)
    states = {name: pop.encode(*truth.unbind(-1)) for name in ("fixed", "learned")}
    decoded = {name: [] for name in states}
    targets = []
    for i in range(steps + 1):
        targets.append(truth.cpu().numpy().copy())
        for name, h in states.items():
            phase = pop.decode(h)
            phase = phase[:, None] if dims == 1 else torch.stack(phase, -1)
            decoded[name].append(phase.cpu().numpy())
        if i == steps:
            break
        v = velocity[i]
        states["fixed"] = pop.rk4_step(states["fixed"], v[:, 0] if dims == 1 else v, dt)
        states["learned"] = rk4(lambda h: pop.force(h) + shifting(h, v), states["learned"], dt)
        truth = wrap_angle(truth + dt * v)
        if (i + 1) % max(1, steps // 10) == 0:
            print(f"{pop.kind.upper()} tracking: {i+1}/{steps} steps", flush=True)
    truth = np.asarray(targets)
    burn = int(0.1 * (steps + 1))
    records = []
    for method, values in decoded.items():
        _finite(states[method])
        delta = np.asarray(values) - truth
        error = np.abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))))
        for j, amplitude in enumerate(amplitudes):
            per_axis = error[burn:, j*runs:(j+1)*runs].mean(0)
            # Average Grid axes within each trajectory before computing its SD.
            per_trajectory = np.asarray(per_axis, dtype=np.float64).mean(-1)
            records.append({"population": pop.kind, "velocity_limit": float(amplitude),
                            "method": method, "unit": "degree",
                            **_summary(per_trajectory)})
    return records


@torch.no_grad()
def place_tracking(ckpt, shifting_ckpt, *, device="cuda", runs=100,
                   duration=20, dt=0.05, seed=104):
    """Table 2: Combined Place with the bounded-map decoder."""
    pop = load_population(ckpt, device)
    if pop.kind != "place" or pop.energy_mode != "both":
        raise ValueError("Use the Combined Place checkpoint")
    pop.gain = 2
    shifting, metadata = load_bump_shifting(shifting_ckpt, pop, ckpt)
    if metadata["config"]["gain"] != pop.gain:
        raise ValueError("Energy and bump-shifting gains differ")
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("Duration must be a positive integer multiple of dt")
    lo = pop.wall_margin * pop.arena_size
    hi = pop.arena_size - lo
    errors = {"fixed": [], "learned": []}
    for evaluation_seed in range(seed, seed + runs):
        rng = np.random.RandomState(evaluation_seed + 1)
        x = np.full(2, pop.arena_size / 2, dtype=np.float32)
        v = rng.randn(2).astype(np.float32) * 0.25
        states = {name: pop.encode(torch.tensor(x[None], device=device)) for name in errors}
        history = {name: [h] for name, h in states.items()}
        truth = [x.copy()]
        for _ in range(steps):
            v = 0.95 * v + 0.025 * rng.randn(2).astype(np.float32)
            for j in range(2):
                if x[j] + v[j] * dt < lo:
                    v[j] = abs(v[j])
                elif x[j] + v[j] * dt > hi:
                    v[j] = -abs(v[j])
            x = np.clip(x + dt * v, lo, hi)
            vt = torch.tensor(v[None], device=device)
            states["fixed"] = pop.rk4_step(states["fixed"], vt, dt)
            states["learned"] = rk4(lambda h: pop.force(h) + shifting(h, vt), states["learned"], dt)
            for name, h in states.items():
                _finite(h)
                history[name].append(h)
            truth.append(x.copy())
        for name, values in history.items():
            position = pop.decode(torch.cat(values)).cpu().numpy()
            position[0] = truth[0]
            error = np.linalg.norm(position - np.asarray(truth), axis=-1)
            errors[name].append(float(error.mean()))
        print(f"Place tracking: {evaluation_seed-seed+1}/{runs} trajectories", flush=True)
    return [{"population": "place", "velocity_limit": "random_walk", "method": method,
             "unit": "metre", **_summary(values)} for method, values in errors.items()]
