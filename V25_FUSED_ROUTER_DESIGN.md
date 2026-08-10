# AGILLM 4.3 EGGROLL v25: fused discrete-router redesign

**Status:** experimental, separate branch, disabled by default  
**Production checkpoint policy:** no fork until the wall-clock acceptance test passes

## 1. Why v24 did not earn promotion

The corrected fixed-token A100 probe was useful because it ruled out the easy
answer.  The active branch used a population of 16 on every committed step,
but ran at only 70.6% of baseline throughput and produced no statistically
resolved next-token-loss gain.  The update cap also limited accepted router
updates to 0.1% of router-weight RMS; all 14 guard-loss changes rounded to
zero at the recorded precision.

That result is not a general rejection of EGGROLL.  It is evidence against
the particular implementation and objective:

1. Candidate evaluation was attached to ordinary backprop training rather
   than implemented as a true fused population operation.
2. It spent zeroth-order samples on differentiable cross-entropy, where
   backprop already provides a much denser signal per token.
3. A fixed sigma of 0.1 was needed to obtain frequent pair signal, but it
   caused discontinuous top-1 route changes.
4. Tiny clipped updates could pass a non-regression gate without producing a
   measurable held-out effect.
5. Fourteen events were sufficient to reject a large throughput regression,
   but not sufficient to establish a small optimisation advantage.

The redesign therefore changes the computational path and the objective.
Simply sweeping the old knobs would be cheaper theatre, but still theatre.

## 2. Design goal

Use EGGROLL only for the part of the MoE system that is genuinely awkward
for gradients: **hard, discrete top-1 expert assignment**.

For one selected router event:

1. Capture the router input activations once from a normal training batch.
2. Compute each expert's local counterfactual cost for each captured token
   once.
3. Evaluate many low-rank router perturbations from the captured activations
   with fused tensor contractions.
4. Score each hard routing pattern by gathering from the precomputed
   per-expert cost table.
5. Construct the dense antithetic ES update without materialising dense
   candidate weight matrices.
6. Run full-model search and guard checks only for the proposed aggregate
   update, not once per population member.

This turns population cost from approximately "one model forward per
candidate" into cheap router algebra plus a small, fixed number of guard
forwards.

## 3. Fused kernel

`experiments/v25-fused-router/fused_router_kernel.py` implements the core
operation for a router weight `W`, activations `x`, and rank-r factors
`A_i, B_i`:

```text
base_i = x W^T
delta_i = sigma / sqrt(r) * (x B_i) A_i^T
positive_i = base + delta_i
negative_i = base - delta_i
```

The base projection is evaluated once.  The population dimension is handled
by batched `einsum` contractions, and candidates can be chunked to respect
VRAM limits.

The dense update is accumulated directly as:

```text
update = lr / (2 P sigma)
         * sum_i (fitness_i+ - fitness_i-)
         * A_i B_i^T / sqrt(r)
```

No `[population, out_features, in_features]` tensor is required.

The module also supplies:

- exact antithetic positive/negative evaluation;
- chunked population fitness evaluation;
- hard-route flip measurement;
- a multiplicative sigma controller;
- counterfactual top-1 fitness from precomputed expert costs;
- fp32 update accumulation for fp16/bf16 routers;
- an optional router-update RMS cap;
- deterministic unit tests against dense reference calculations.

## 4. Counterfactual router objective

For two experts, compute a local cost table:

```text
expert_cost[token, expert]
```

The exact definition must be wired to the current diffusion-block local
objective.  Candidate fitness is then:

```text
route = argmax(candidate_router_logits)
fitness = -mean(expert_cost[token, route[token]])
          - load_balance_penalty
```

This objective is discontinuous at route boundaries, so zeroth-order search
has a legitimate role.  It also prevents EGGROLL from redundantly estimating
the gradient of ordinary cross-entropy.

The first integration should support two modes:

- `counterfactual_local`: the promotion candidate, using actual per-expert
  local costs.
- `router_margin_diagnostic`: a debugging-only signal used to calibrate
  sigma, never accepted as evidence of language-model improvement.

An entropy-only or balance-only objective is not acceptable because it can
make routing statistics look tidy while degrading expert choice.

## 5. Adaptive perturbation scale

The v24 sweep showed almost no pair signal at very small sigma and frequent
route discontinuities at sigma 0.1.  v25 therefore controls sigma from the
observed hard-route flip rate.

Initial contract:

- target mean antithetic route-flip fraction: 5%;
- deadband: ±20% relative to target;
- multiplicative update gain: 0.25;
- maximum sigma change per event: 2x;
- hard sigma bounds: `1e-5` to `0.25`;
- controller state persisted per router.

The target is a starting hypothesis, not a magic constant.  The experiment
must log sigma, pair-signal RMS, route flips, update RMS and guard changes for
every event.

## 6. Event scheduling

Do not run an ES event every training step.

Initial experimental schedule:

- one target router per event;
- round-robin router selection;
- one event every 32 committed diffusion-block updates;
- 64 antithetic pairs (population 128), chunked as VRAM permits;
- rank 1 initially;
- four captured search crops;
- two independent full-model guard crops;
- no checkpoint fork from an accepted event during the probe.

A fused population needs enough samples to provide a useful estimator.  The
paper's pretraining results explicitly rely on large populations, so reducing
the population to two or four merely to make the clock look pleasant would
destroy the method being tested.

## 7. Update acceptance

"Accepted" must mean measurable improvement, not merely "did not trip a
zero-tolerance guard".

A proposed router update is eligible only when all conditions hold:

1. Counterfactual search fitness improves.
2. The full-model search loss improves by more than two times the measured
   baseline-repeat noise scale.
3. Neither independent guard crop regresses beyond its baseline noise bound.
4. Router update RMS is within the configured cap.
5. Route load remains inside the existing MoE skew limit.
6. All values are finite and exact token/hash contracts match.

Rejected updates are rolled back atomically.  The log must distinguish
`no_signal`, `counterfactual_only`, `guard_rejected`, `accepted_measurable`
and `invalid`.

## 8. Controlled A/B protocol

The first paid v25 test should remain small enough to stop cheaply but large
enough to answer the question.

### Branches

- baseline A;
- baseline B;
- v25 fused-router active, seed 1;
- v25 fused-router active, seed 2;
- v25 fused-router active, seed 3.

All branches use the same frozen checkpoint, fixed token file, batch hashes,
objective schedule and committed-step window.

### Minimum window

- at least 128 committed updates per branch;
- at least four EGGROLL events per active branch;
- extend to 512 commits only if the predeclared futility check is not met.

### Primary metric

**Improvement per wall-clock second**, reported as both:

- loss-area-under-curve versus elapsed time;
- elapsed time or GPU-seconds to reach the same paired loss target.

Tokens per second is a required secondary metric, not the sole objective.

### Promotion gate

Promotion remains forbidden unless all of the following hold:

- active throughput is at least 95% of baseline;
- the paired loss-per-second effect is favourable in all three active seeds;
- the aggregate 95% confidence interval excludes zero;
- no guard or stability regression is observed;
- projected compute to a fixed loss target is lower than baseline.

If the fused kernel is fast but the optimisation effect remains unresolved,
retain it as experimental infrastructure rather than forcing it into
production through optimism and decorative naming.

## 9. Integration sequence

1. Land and test the standalone fused router kernel.
2. Add a trainer hook that captures selected-router activations and
   counterfactual expert costs.
3. Run a no-update correctness probe comparing fused population logits with
   the old dense candidate path.
4. Run a synthetic objective test proving that the aggregate update follows
   the known optimum.
5. Run the fixed-token controlled A/B described above.
6. Only after a passing audit, create a new optional trainer version and a
   candidate checkpoint fork.

Production v22 and the frozen step-1,777,105 checkpoint remain unchanged.
