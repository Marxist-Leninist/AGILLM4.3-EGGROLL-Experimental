"""Fused low-rank antithetic evolution utilities for AGILLM 4.3 routers.

This module is deliberately trainer-agnostic.  It provides the GPU-friendly
mathematics needed by the v25 EGGROLL redesign without modifying production
training code.  A caller captures router inputs once, evaluates a population
of low-rank perturbations in batched tensor operations, obtains one scalar
objective per perturbation, and applies the resulting dense router update.

The implementation follows the antithetic Gaussian ES estimator.  It does
not claim that EGGROLL improves next-token pretraining; promotion still
requires a fixed-token, wall-clock controlled A/B test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PopulationFactors:
    """Rank-r factors for one perturbation from every antithetic pair.

    ``a`` has shape ``[pairs, out_features, rank]``.
    ``b`` has shape ``[pairs, in_features, rank]``.
    The positive perturbation is ``A @ B.T / sqrt(rank)`` and the negative
    member uses its exact negation.
    """

    a: Tensor
    b: Tensor

    @property
    def pairs(self) -> int:
        return int(self.a.shape[0])

    @property
    def rank(self) -> int:
        return int(self.a.shape[-1])

    def validate(self, *, in_features: int, out_features: int) -> None:
        if self.a.ndim != 3 or self.b.ndim != 3:
            raise ValueError("a and b must both be rank-3 tensors")
        if self.a.shape[0] != self.b.shape[0]:
            raise ValueError("a and b must contain the same number of pairs")
        if self.a.shape[1] != out_features:
            raise ValueError(
                f"a out_features mismatch: {self.a.shape[1]} != {out_features}"
            )
        if self.b.shape[1] != in_features:
            raise ValueError(
                f"b in_features mismatch: {self.b.shape[1]} != {in_features}"
            )
        if self.a.shape[2] != self.b.shape[2]:
            raise ValueError("a and b must use the same rank")
        if self.pairs < 1 or self.rank < 1:
            raise ValueError("at least one pair and rank >= 1 are required")
        if self.a.device != self.b.device:
            raise ValueError("a and b must be on the same device")
        if self.a.dtype != self.b.dtype:
            raise ValueError("a and b must have the same dtype")


@dataclass(frozen=True)
class UpdateStats:
    pairs: int
    sigma: float
    pair_signal_mean: float
    pair_signal_std: float
    raw_update_rms: float
    applied_update_rms: float
    weight_rms: float
    clip_scale: float


@dataclass(frozen=True)
class SigmaController:
    """Multiplicatively adapt sigma toward a target hard-route flip rate."""

    target_flip_fraction: float = 0.05
    gain: float = 0.25
    min_sigma: float = 1e-5
    max_sigma: float = 0.25
    max_change_factor: float = 2.0
    deadband_fraction: float = 0.20
    eps: float = 1e-8

    def update(self, sigma: float, observed_flip_fraction: float) -> float:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not math.isfinite(observed_flip_fraction):
            raise ValueError("observed_flip_fraction must be finite")
        if not 0 < self.target_flip_fraction < 1:
            raise ValueError("target_flip_fraction must be in (0, 1)")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.min_sigma <= 0 or self.max_sigma < self.min_sigma:
            raise ValueError("invalid sigma bounds")
        if self.max_change_factor < 1:
            raise ValueError("max_change_factor must be >= 1")

        observed = min(max(float(observed_flip_fraction), 0.0), 1.0)
        relative_error = (
            abs(observed - self.target_flip_fraction)
            / self.target_flip_fraction
        )
        if relative_error <= self.deadband_fraction:
            return min(max(float(sigma), self.min_sigma), self.max_sigma)

        log_step = self.gain * math.log(
            self.target_flip_fraction / max(observed, self.eps)
        )
        max_log_step = math.log(self.max_change_factor)
        log_step = min(max(log_step, -max_log_step), max_log_step)
        new_sigma = float(sigma) * math.exp(log_step)
        return min(max(new_sigma, self.min_sigma), self.max_sigma)


def sample_population_factors(
    *,
    pairs: int,
    in_features: int,
    out_features: int,
    rank: int,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> PopulationFactors:
    """Sample Gaussian low-rank factors for antithetic perturbation pairs."""

    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    if in_features < 1 or out_features < 1 or rank < 1:
        raise ValueError("feature sizes and rank must be >= 1")
    a = torch.randn(
        pairs,
        out_features,
        rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    b = torch.randn(
        pairs,
        in_features,
        rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return PopulationFactors(a=a, b=b)


def _validate_inputs(
    activations: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    factors: PopulationFactors,
    sigma: float,
) -> None:
    if activations.ndim < 2:
        raise ValueError("activations must have shape [..., in_features]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    out_features, in_features = map(int, weight.shape)
    if activations.shape[-1] != in_features:
        raise ValueError(
            "activation width does not match router weight input width"
        )
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("bias must have shape [out_features]")
    if activations.device != weight.device:
        raise ValueError("activations and weight must be on the same device")
    if activations.dtype != weight.dtype:
        raise ValueError("activations and weight must have the same dtype")
    if bias is not None and (bias.device != weight.device or bias.dtype != weight.dtype):
        raise ValueError("bias must match weight device and dtype")
    factors.validate(in_features=in_features, out_features=out_features)
    if factors.a.device != weight.device or factors.a.dtype != weight.dtype:
        raise ValueError("population factors must match weight device and dtype")
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")


def fused_antithetic_router_logits(
    activations: Tensor,
    weight: Tensor,
    factors: PopulationFactors,
    *,
    sigma: float,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate all positive/negative low-rank router perturbations.

    Returns ``(positive, negative)`` with shape
    ``[pairs, *activations.shape[:-1], out_features]``.

    The base router projection is computed once.  Candidate deltas are
    produced with two batched contractions instead of materialising one
    dense perturbed weight matrix per population member.
    """

    _validate_inputs(activations, weight, bias, factors, sigma)
    base = F.linear(activations, weight, bias)
    # [..., in] x [pairs, in, rank] -> [pairs, ..., rank]
    projected = torch.einsum("...i,pir->p...r", activations, factors.b)
    # [pairs, ..., rank] x [pairs, out, rank] -> [pairs, ..., out]
    delta = torch.einsum("p...r,por->p...o", projected, factors.a)
    delta = delta * (float(sigma) / math.sqrt(factors.rank))
    base = base.unsqueeze(0)
    return base + delta, base - delta


def iter_fused_antithetic_router_logits(
    activations: Tensor,
    weight: Tensor,
    factors: PopulationFactors,
    *,
    sigma: float,
    bias: Tensor | None = None,
    pair_chunk: int,
) -> Iterator[tuple[slice, Tensor, Tensor]]:
    """Yield population chunks while computing the base projection once."""

    _validate_inputs(activations, weight, bias, factors, sigma)
    if pair_chunk < 1:
        raise ValueError("pair_chunk must be >= 1")

    base = F.linear(activations, weight, bias).unsqueeze(0)
    scale = float(sigma) / math.sqrt(factors.rank)
    for start in range(0, factors.pairs, pair_chunk):
        stop = min(start + pair_chunk, factors.pairs)
        a = factors.a[start:stop]
        b = factors.b[start:stop]
        projected = torch.einsum("...i,pir->p...r", activations, b)
        delta = torch.einsum("p...r,por->p...o", projected, a) * scale
        yield slice(start, stop), base + delta, base - delta


@torch.no_grad()
def evaluate_antithetic_fitness(
    activations: Tensor,
    weight: Tensor,
    factors: PopulationFactors,
    *,
    sigma: float,
    fitness_fn: Callable[[Tensor], Tensor],
    bias: Tensor | None = None,
    pair_chunk: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate scalar fitness for every antithetic population member.

    ``fitness_fn`` receives logits shaped
    ``[population_chunk, *token_shape, out_features]`` and must return one
    scalar per first-dimension member.  Higher values should mean better.
    """

    if pair_chunk is None:
        pair_chunk = factors.pairs
    positive_scores: list[Tensor] = []
    negative_scores: list[Tensor] = []
    with torch.inference_mode():
        for _, positive, negative in iter_fused_antithetic_router_logits(
            activations,
            weight,
            factors,
            sigma=sigma,
            bias=bias,
            pair_chunk=pair_chunk,
        ):
            pos = fitness_fn(positive)
            neg = fitness_fn(negative)
            if pos.ndim != 1 or pos.shape[0] != positive.shape[0]:
                raise ValueError(
                    "fitness_fn must return shape [population_chunk]"
                )
            if neg.shape != pos.shape:
                raise ValueError("positive and negative score shapes differ")
            positive_scores.append(pos)
            negative_scores.append(neg)
    return torch.cat(positive_scores), torch.cat(negative_scores)


def route_flip_fraction(
    activations: Tensor,
    weight: Tensor,
    factors: PopulationFactors,
    *,
    sigma: float,
    bias: Tensor | None = None,
    pair_chunk: int | None = None,
) -> float:
    """Return the mean hard top-1 route change across both pair signs."""

    if pair_chunk is None:
        pair_chunk = factors.pairs
    base_route = F.linear(activations, weight, bias).argmax(dim=-1)
    changed = 0
    total = 0
    with torch.inference_mode():
        for _, positive, negative in iter_fused_antithetic_router_logits(
            activations,
            weight,
            factors,
            sigma=sigma,
            bias=bias,
            pair_chunk=pair_chunk,
        ):
            target = base_route.unsqueeze(0)
            changed += int((positive.argmax(dim=-1) != target).sum().item())
            changed += int((negative.argmax(dim=-1) != target).sum().item())
            total += 2 * positive.shape[0] * base_route.numel()
    return float(changed / total) if total else 0.0


def counterfactual_top1_fitness(
    logits: Tensor,
    expert_costs: Tensor,
    *,
    load_balance_coefficient: float = 0.0,
) -> Tensor:
    """Score hard router choices using precomputed per-expert local costs.

    ``logits`` must have shape ``[population, *token_shape, experts]``.
    ``expert_costs`` must have shape ``[*token_shape, experts]`` and contain
    a smaller-is-better local cost for assigning each token to each expert.

    This makes the candidate objective genuinely non-differentiable while
    avoiding a full model forward per perturbation.  The optional balance
    term penalises population members that collapse tokens onto one expert.
    """

    if logits.ndim < 3:
        raise ValueError(
            "logits must have shape [population, *token_shape, experts]"
        )
    if expert_costs.shape != logits.shape[1:]:
        raise ValueError(
            "expert_costs must match logits without the population dimension"
        )
    if logits.device != expert_costs.device:
        raise ValueError("logits and expert_costs must be on the same device")
    if not math.isfinite(float(load_balance_coefficient)):
        raise ValueError("load_balance_coefficient must be finite")
    if load_balance_coefficient < 0:
        raise ValueError("load_balance_coefficient must be non-negative")

    population = int(logits.shape[0])
    experts = int(logits.shape[-1])
    token_dims = tuple(range(1, logits.ndim - 1))
    route = logits.argmax(dim=-1)
    costs = expert_costs.unsqueeze(0).expand(population, *expert_costs.shape)
    selected = torch.gather(costs, dim=-1, index=route.unsqueeze(-1)).squeeze(-1)
    fitness = -selected.mean(dim=token_dims)

    if load_balance_coefficient:
        one_hot = F.one_hot(route, num_classes=experts).to(logits.dtype)
        frequency = one_hot.mean(dim=token_dims)
        target = 1.0 / experts
        balance_penalty = (frequency - target).square().sum(dim=-1)
        fitness = fitness - float(load_balance_coefficient) * balance_penalty
    return fitness


def antithetic_es_update(
    weight: Tensor,
    factors: PopulationFactors,
    positive_fitness: Tensor,
    negative_fitness: Tensor,
    *,
    sigma: float,
    learning_rate: float,
    max_update_rms_ratio: float | None = None,
    standardize_pair_signal: bool = False,
    eps: float = 1e-12,
) -> tuple[Tensor, UpdateStats]:
    """Construct a dense ascent update without materialising perturbations.

    This implements

        lr / (2 * pairs * sigma) * sum_i (f_i+ - f_i-) E_i,

    where ``E_i = A_i B_i.T / sqrt(rank)``.  Fitness must be larger-is-better.
    To minimise a loss, pass its negation as fitness.
    """

    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    factors.validate(
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
    )
    if positive_fitness.shape != (factors.pairs,):
        raise ValueError("positive_fitness must have shape [pairs]")
    if negative_fitness.shape != positive_fitness.shape:
        raise ValueError("negative_fitness shape mismatch")
    if positive_fitness.device != weight.device:
        raise ValueError("fitness tensors must be on the weight device")
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(float(learning_rate)) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and non-negative")
    if max_update_rms_ratio is not None and max_update_rms_ratio <= 0:
        raise ValueError("max_update_rms_ratio must be positive")

    accumulation_dtype = (
        torch.float64 if weight.dtype == torch.float64 else torch.float32
    )
    signal = (positive_fitness - negative_fitness).to(
        device=weight.device,
        dtype=accumulation_dtype,
    )
    if not torch.isfinite(signal).all():
        raise ValueError("fitness produced non-finite pair signals")
    if standardize_pair_signal:
        signal = signal - signal.mean()
        scale = signal.square().mean().sqrt()
        signal = signal / scale.clamp_min(eps)

    # Accumulate low-precision routers in fp32; preserve fp64 for tests.
    a = factors.a.to(dtype=accumulation_dtype)
    b = factors.b.to(dtype=accumulation_dtype)
    direction = torch.einsum("p,por,pir->oi", signal, a, b)
    direction = direction / math.sqrt(factors.rank)
    direction = direction / (2.0 * factors.pairs * float(sigma))
    raw_update = direction * float(learning_rate)

    weight_rms = float(weight.float().square().mean().sqrt().item())
    raw_update_rms = float(raw_update.square().mean().sqrt().item())
    clip_scale = 1.0
    if max_update_rms_ratio is not None:
        limit = float(max_update_rms_ratio) * max(weight_rms, eps)
        if raw_update_rms > limit:
            clip_scale = limit / max(raw_update_rms, eps)

    applied = (raw_update * clip_scale).to(dtype=weight.dtype)
    applied_rms = float(applied.float().square().mean().sqrt().item())
    stats = UpdateStats(
        pairs=factors.pairs,
        sigma=float(sigma),
        pair_signal_mean=float(signal.mean().item()),
        pair_signal_std=float(signal.std(unbiased=False).item()),
        raw_update_rms=raw_update_rms,
        applied_update_rms=applied_rms,
        weight_rms=weight_rms,
        clip_scale=float(clip_scale),
    )
    return applied, stats


@torch.no_grad()
def apply_update_(weight: Tensor, update: Tensor) -> None:
    """Apply a previously audited update in place."""

    if weight.shape != update.shape:
        raise ValueError("weight and update shapes differ")
    if weight.device != update.device:
        raise ValueError("weight and update devices differ")
    weight.add_(update)
