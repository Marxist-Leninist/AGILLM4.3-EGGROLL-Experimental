# v25 fused-router EGGROLL experiment

This directory contains the first code stage of the AGILLM 4.3 EGGROLL
redesign. It is intentionally independent from the production single-file
trainer.

## Contents

- `fused_router_kernel.py`: shared-base antithetic low-rank router kernel,
  counterfactual hard-routing fitness, adaptive sigma controller and dense
  update synthesis.
- `test_fused_router_kernel.py`: deterministic comparisons against dense
  reference calculations.
- `benchmark_fused_router.py`: kernel-only correctness and timing harness.

The scientific and promotion contract is in
[`../../V25_FUSED_ROUTER_DESIGN.md`](../../V25_FUSED_ROUTER_DESIGN.md).

## Test

```bash
cd experiments/v25-fused-router
python -m pytest -q test_fused_router_kernel.py
```

## Microbenchmark

```bash
cd experiments/v25-fused-router
python benchmark_fused_router.py \
  --device cuda \
  --pairs 64 \
  --tokens 512 \
  --in-features 512 \
  --out-features 2 \
  --rank 1 \
  --sigma 0.01
```

The benchmark compares the fused router algebra with a deliberately simple
dense reference. It does **not** establish end-to-end training speedup.

## Integration boundary

The next trainer patch must provide a per-token, per-expert local cost table
for one selected MoE router:

```python
expert_costs.shape == (*token_shape, number_of_experts)
```

Then candidate fitness can be evaluated without one full model forward per
perturbation:

```python
positive_fitness, negative_fitness = evaluate_antithetic_fitness(
    activations,
    router.weight,
    factors,
    sigma=sigma,
    fitness_fn=lambda logits: counterfactual_top1_fitness(
        logits,
        expert_costs,
        load_balance_coefficient=load_balance_coefficient,
    ),
    pair_chunk=pair_chunk,
)
```

A full-model fixed-token guard remains mandatory before any update is
accepted. The module being fast is useful engineering; it is not permission
to promote the method by wishful thinking.
