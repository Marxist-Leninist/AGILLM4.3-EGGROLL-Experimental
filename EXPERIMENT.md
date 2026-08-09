# Experimental design

## Estimator

For each selected router weight matrix W and antithetic pair i, the sidecar samples low-rank factors A_i and B_i and evaluates

    W_i+ = W + sigma / sqrt(r) * A_i B_i^T
    W_i- = W - sigma / sqrt(r) * A_i B_i^T

without copying W. During the forward pass, the equivalent perturbation is added to router logits as

    x B_i A_i^T.

The pair coefficient is estimated from the deterministic search-crop losses:

    c_i = (L_i+ - L_i-) / (2 sigma).

The reconstructed descent direction is

    G = mean_i c_i A_i B_i^T / sqrt(r).

The implementation can standardise pair coefficients before reconstruction. It then applies a direct, small router-only update bounded by a configurable ratio of update RMS to weight RMS.

## Acceptance boundary

An event is committed only when all of the following hold:

- candidate losses and reconstructed updates are finite;
- antithetic pair signal exceeds `--eggroll_min_pair_signal`;
- the unperturbed search crop does not regress after the update;
- a separate concatenated set of guard crops does not regress beyond `--eggroll_accept_tolerance`.

A rejected event restores every selected router tensor from an exact clone. EGGROLL evaluation runs in model evaluation mode, so dropout does not contaminate antithetic differences. RNG state is restored after the event.

## Production boundary

The production v22 trainer is not modified. EGGROLL is forbidden in the immutable repair profile, disabled by default everywhere else, and not executed during the current completion-only SFT path. A v23 checkpoint stores configuration identity and all EGGROLL counters so an exact v23 resume cannot silently change experimental settings.

## First A/B protocol

1. Run v23 with `--eggroll_dry_run` and population 8 for enough scheduled events to estimate wall-clock overhead and pair signal.
2. Fork the same frozen checkpoint into baseline and EGGROLL branches.
3. Give both branches equal GPU-hours and identical dataset order.
4. Compare held-out AR/SAT/NAT loss, downstream reasoning/code evaluations, accepted-event rate, EGGROLL fitness tokens, and total tokens processed.
5. Promote nothing solely because population evaluations per second look impressive. The denominator remains total GPU-hours to useful model improvement, a detail benchmarks occasionally misplace behind a very attractive graph.

## Future work

- dedicated ES optimiser state or AdamW-moment reconciliation;
- active-router or high-uncertainty router selection;
- SAT/NAT fitness terms;
- outcome rewards for maths, code execution, tools, and pass@k diversity;
- multi-router and two-layer DBlock perturbations;
- multi-GPU candidate evaluation;
- held-out guard batches and sequential statistical acceptance.
