from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fused_router_kernel import (
    PopulationFactors,
    SigmaController,
    antithetic_es_update,
    counterfactual_top1_fitness,
    evaluate_antithetic_fitness,
    fused_antithetic_router_logits,
    route_flip_fraction,
)


def _fixture(dtype: torch.dtype = torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    activations = torch.randn(2, 3, 5, generator=generator, dtype=dtype)
    weight = torch.randn(4, 5, generator=generator, dtype=dtype)
    bias = torch.randn(4, generator=generator, dtype=dtype)
    factors = PopulationFactors(
        a=torch.randn(7, 4, 2, generator=generator, dtype=dtype),
        b=torch.randn(7, 5, 2, generator=generator, dtype=dtype),
    )
    return activations, weight, bias, factors


def test_fused_logits_match_dense_naive_reference():
    activations, weight, bias, factors = _fixture()
    sigma = 0.037
    positive, negative = fused_antithetic_router_logits(
        activations,
        weight,
        factors,
        sigma=sigma,
        bias=bias,
    )

    perturbations = torch.einsum(
        "por,pir->poi", factors.a, factors.b
    ) / math.sqrt(factors.rank)
    positive_reference = torch.stack(
        [
            F.linear(activations, weight + sigma * perturbation, bias)
            for perturbation in perturbations
        ]
    )
    negative_reference = torch.stack(
        [
            F.linear(activations, weight - sigma * perturbation, bias)
            for perturbation in perturbations
        ]
    )

    torch.testing.assert_close(positive, positive_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(negative, negative_reference, rtol=1e-12, atol=1e-12)


def test_chunked_fitness_matches_full_population():
    activations, weight, bias, factors = _fixture()
    target = torch.randn(2, 3, 4, dtype=activations.dtype)

    def fitness(logits: torch.Tensor) -> torch.Tensor:
        return -(logits - target.unsqueeze(0)).square().mean(dim=(1, 2, 3))

    full = evaluate_antithetic_fitness(
        activations,
        weight,
        factors,
        sigma=0.02,
        bias=bias,
        fitness_fn=fitness,
        pair_chunk=factors.pairs,
    )
    chunked = evaluate_antithetic_fitness(
        activations,
        weight,
        factors,
        sigma=0.02,
        bias=bias,
        fitness_fn=fitness,
        pair_chunk=3,
    )
    torch.testing.assert_close(chunked[0], full[0])
    torch.testing.assert_close(chunked[1], full[1])


def test_update_matches_materialised_antithetic_estimator():
    _, weight, _, factors = _fixture()
    positive = torch.tensor(
        [0.1, -0.3, 0.7, 0.2, 0.5, -0.4, 0.9], dtype=weight.dtype
    )
    negative = torch.tensor(
        [-0.2, -0.1, 0.4, 0.0, 0.6, -0.8, 0.3], dtype=weight.dtype
    )
    sigma = 0.05
    learning_rate = 0.007

    update, stats = antithetic_es_update(
        weight,
        factors,
        positive,
        negative,
        sigma=sigma,
        learning_rate=learning_rate,
    )

    perturbations = torch.einsum(
        "por,pir->poi", factors.a, factors.b
    ) / math.sqrt(factors.rank)
    expected = (
        learning_rate
        * torch.einsum("p,poi->oi", positive - negative, perturbations)
        / (2 * factors.pairs * sigma)
    )
    torch.testing.assert_close(update, expected, rtol=1e-12, atol=1e-12)
    assert stats.clip_scale == pytest.approx(1.0)


def test_update_rms_cap_is_enforced():
    _, weight, _, factors = _fixture(dtype=torch.float32)
    positive = torch.full((factors.pairs,), 1000.0)
    negative = -positive
    cap = 0.002

    update, stats = antithetic_es_update(
        weight,
        factors,
        positive,
        negative,
        sigma=0.01,
        learning_rate=1.0,
        max_update_rms_ratio=cap,
    )

    expected_limit = cap * weight.square().mean().sqrt()
    actual = update.square().mean().sqrt()
    assert actual <= expected_limit * (1 + 1e-5)
    assert stats.clip_scale < 1.0


def test_sigma_controller_moves_toward_target_flip_rate():
    controller = SigmaController(target_flip_fraction=0.05)
    assert controller.update(0.01, 0.001) > 0.01
    assert controller.update(0.01, 0.40) < 0.01
    assert controller.update(0.01, 0.052) == pytest.approx(0.01)


def test_route_flip_fraction_has_valid_bounds_and_reacts_to_sigma():
    activations, weight, bias, factors = _fixture(dtype=torch.float32)
    small = route_flip_fraction(
        activations, weight, factors, sigma=1e-6, bias=bias, pair_chunk=3
    )
    large = route_flip_fraction(
        activations, weight, factors, sigma=2.0, bias=bias, pair_chunk=3
    )
    assert 0.0 <= small <= 1.0
    assert 0.0 <= large <= 1.0
    assert large >= small


def test_counterfactual_top1_fitness_uses_precomputed_expert_costs():
    logits = torch.tensor(
        [
            [[[4.0, 1.0], [0.0, 3.0]]],
            [[[1.0, 4.0], [3.0, 0.0]]],
        ]
    )
    # Candidate zero selects expert 0 then 1, both with cost 1.
    # Candidate one selects expert 1 then 0, both with cost 5.
    costs = torch.tensor([[[1.0, 5.0], [5.0, 1.0]]])
    fitness = counterfactual_top1_fitness(logits, costs)
    torch.testing.assert_close(fitness, torch.tensor([-1.0, -5.0]))
