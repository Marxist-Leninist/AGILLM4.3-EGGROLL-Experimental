#!/usr/bin/env python3
"""Correctness and microbenchmark harness for the v25 fused router kernel."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from fused_router_kernel import (
    PopulationFactors,
    fused_antithetic_router_logits,
    route_flip_fraction,
    sample_population_factors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, *, iterations: int, device: torch.device) -> float:
    for _ in range(5):
        fn()
    synchronise(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronise(device)
    return (time.perf_counter() - start) / iterations


def dense_reference(
    activations: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    factors: PopulationFactors,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    perturbations = torch.einsum(
        "por,pir->poi", factors.a, factors.b
    ) / math.sqrt(factors.rank)
    positive = torch.stack(
        [
            F.linear(activations, weight + sigma * perturbation, bias)
            for perturbation in perturbations
        ]
    )
    negative = torch.stack(
        [
            F.linear(activations, weight - sigma * perturbation, bias)
            for perturbation in perturbations
        ]
    )
    return positive, negative


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit("float16 CPU linear algebra is not a useful benchmark")

    generator = torch.Generator(device=device).manual_seed(4317)
    activations = torch.randn(
        args.tokens,
        args.in_features,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        args.out_features,
        args.in_features,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    bias = torch.randn(
        args.out_features,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    factors = sample_population_factors(
        pairs=args.pairs,
        in_features=args.in_features,
        out_features=args.out_features,
        rank=args.rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    with torch.inference_mode():
        fused = fused_antithetic_router_logits(
            activations, weight, factors, sigma=args.sigma, bias=bias
        )
        # Keep the correctness reference bounded so a careless command does
        # not manufacture a VRAM crisis for sport.
        reference_pairs = min(args.pairs, 8)
        reference_factors = PopulationFactors(
            a=factors.a[:reference_pairs],
            b=factors.b[:reference_pairs],
        )
        reference = dense_reference(
            activations, weight, bias, reference_factors, args.sigma
        )
        torch.testing.assert_close(
            fused[0][:reference_pairs],
            reference[0],
            rtol=2e-3 if dtype != torch.float32 else 1e-5,
            atol=2e-3 if dtype != torch.float32 else 1e-5,
        )
        torch.testing.assert_close(
            fused[1][:reference_pairs],
            reference[1],
            rtol=2e-3 if dtype != torch.float32 else 1e-5,
            atol=2e-3 if dtype != torch.float32 else 1e-5,
        )

        fused_seconds = timed(
            lambda: fused_antithetic_router_logits(
                activations, weight, factors, sigma=args.sigma, bias=bias
            ),
            iterations=args.iterations,
            device=device,
        )
        dense_seconds = timed(
            lambda: dense_reference(
                activations, weight, bias, factors, args.sigma
            ),
            iterations=args.iterations,
            device=device,
        )
        flip = route_flip_fraction(
            activations,
            weight,
            factors,
            sigma=args.sigma,
            bias=bias,
            pair_chunk=min(32, args.pairs),
        )

    report = {
        "schema": "agillm43.eggroll.v25.fused_router_microbenchmark.v1",
        "device": str(device),
        "dtype": args.dtype,
        "pairs": args.pairs,
        "population": args.pairs * 2,
        "tokens": args.tokens,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "sigma": args.sigma,
        "fused_seconds_per_call": fused_seconds,
        "dense_seconds_per_call": dense_seconds,
        "kernel_speedup_over_dense_reference": dense_seconds / fused_seconds,
        "route_flip_fraction": flip,
        "correctness_reference_pairs": reference_pairs,
        "warning": (
            "Kernel-only microbenchmark. It is not evidence of end-to-end "
            "pretraining acceleration."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
