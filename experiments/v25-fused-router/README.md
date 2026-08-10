# v25 fused-router EGGROLL experiment

This directory contains the first two code stages of the AGILLM 4.3 EGGROLL
redesign. It remains independent from the production single-file trainer and
cannot mutate an active checkpoint unless a caller explicitly supplies
passing fixed-token guard measurements.

## Contents

- `fused_router_kernel.py`: shared-base antithetic low-rank router kernel,
  counterfactual hard-routing fitness, adaptive sigma controller and dense
  update synthesis.
- `trainer_adapter.py`: non-mutating proposal path, aggregate line search,
  event scheduling, round-robin router selection, measured-noise acceptance
  gate and atomic application of accepted updates.
- `test_fused_router_kernel.py`: deterministic comparisons against dense
  reference calculations.
- `test_trainer_adapter.py`: synthetic hard-routing correction and rejection
  tests for noise-only gains, guard regression, hash mismatch and no-signal
  proposals.
- `benchmark_fused_router.py`: kernel-only correctness and timing harness.

The scientific and promotion contract is in
[`../../V25_FUSED_ROUTER_DESIGN.md`](../../V25_FUSED_ROUTER_DESIGN.md).

## Test

```bash
cd experiments/v25-fused-router
python -m pytest -q
```

Current local result: **13 passed**.

The synthetic integration test begins with all eight tokens deliberately
routed to the wrong expert. With a fixed random seed, the fused population
proposal improves the counterfactual objective and moves all eight to the
known correct expert without mutating the source router before acceptance.

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

## Trainer integration boundary

The production trainer must capture a per-token, per-expert local cost table
for one selected MoE router:

```python
expert_costs.shape == (*token_shape, number_of_experts)
```

It then creates a non-mutating proposal:

```python
proposal = propose_router_update(
    RouterEventBatch(
        router_id=router_id,
        committed_step=committed_step,
        activations=router_activations,
        expert_costs=expert_costs,
        router_weight=router.weight,
        router_bias=router.bias,
        fixed_tokens_sha256=fixed_tokens_sha256,
        objective_sha256=objective_sha256,
    ),
    config,
    generator=generator,
)
```

The trainer temporarily evaluates the aggregate proposal on the paired
full-model search crop and independent guard crops, then supplies the losses
to `decide_acceptance`. An update is applicable only when:

- counterfactual router fitness improved;
- full-model search loss improved by more than twice measured baseline-repeat
  noise;
- no independent guard crop regressed beyond its noise allowance;
- fixed-token and objective hashes match;
- route load remains inside the configured skew limit.

`apply_accepted_update_` refuses every label other than
`accepted_measurable`. Thus a fast local kernel, a prettier routing metric or
an update that merely did not crash cannot quietly promote itself into
production, a standard that regrettably excludes much of software history.
