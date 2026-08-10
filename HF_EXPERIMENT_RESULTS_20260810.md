# Hugging Face EGGROLL pretraining experiment, 10 August 2026

## Decision

**Keep EGGROLL optional and disabled by default. Do not fork a new active checkpoint from this result.**

The corrected fixed-token probe found real antithetic router signal, but did **not** demonstrate faster AGILLM 4.3 pretraining. Active EGGROLL took **1.417x** baseline training-window time: **41.7% overhead** and **70.6% of baseline throughput**. Mean local-loss delta was `-0.000785714`, but the median was zero, the 95% t interval was `-0.012267` to `+0.010696`, and the exact two-sided sign-test p-value was `0.726562`. It was better than both baselines on 3/14 commits, worse on 5/14, and tied on 6/14.

## Valid final probe

- Hugging Face Job: `6a7937ae3e1f34a7e32c16e7`
- Hardware: one `a100-large`
- Frozen checkpoint: step `1,777,105`, tokens seen `87,348,264,960`
- Checkpoint manifest SHA-256: `72ee3cf3ae4e3dedb22f16d8c2d21c80c66e0bdcd2001f606f030886cc98e2af`
- Branches: two EGGROLL-disabled repeats and one active branch
- Per branch: 14 commits, batch 4, block 2,048, **114,688 tokens**
- Active setup: population 16, rank 1, sigma 0.1, one router per event, all 14 routers once, four guard crops, update-RMS cap 0.001
- Exact batch, input, target, objective and block hashes matched all 14 commits.
- Fixed token file SHA-256: `381f82fe92dacaf0621d881bdd951e7f5c76d32d403dbe41382cd2d45cd4bbb3`
- Token-stream SHA-256: `bbde62ac4424d40585ae8df66c9369d3606e3dc134b4c75e56ca0493f1b97b42`

## Results

| Metric | Result |
|---|---:|
| Baseline repeat mean absolute loss noise | 0.000285714 |
| Baseline repeat maximum absolute difference | 0.004000000 |
| Active mean loss delta versus mean baseline | -0.000785714 |
| Active median loss delta | +0.000000000 |
| 95% t interval | [-0.012267, +0.010696] |
| Exact two-sided sign-test p | 0.726562 |
| Active better / worse / tie | 3 / 5 / 6 |
| Active/baseline runtime ratio | 1.416807 |
| Accepted / no-signal / rejected | 10 / 3 / 1 |
| Nonzero pair signal | 11 / 14 events |
| Nonzero search CE change | 1 / 14 |
| Nonzero guard CE change | 0 / 14 |

Ten events were labelled `accepted`, meaning their tiny update passed the non-regression gate. It does not mean ten useful improvements. Thirteen of fourteen recorded zero search-CE change at floating-point resolution; all fourteen recorded zero guard-CE change. The one nonzero search change was only `2.86102295e-06`.

## Paired local losses

Negative delta favours EGGROLL.

| Commit | Baseline A | Baseline B | Active | Active minus baseline mean |
|---:|---:|---:|---:|---:|
| 46755 | 10.174000 | 10.174000 | 10.174000 | +0.000000 |
| 46756 | 8.801000 | 8.801000 | 8.801000 | +0.000000 |
| 46757 | 10.876000 | 10.876000 | 10.876000 | +0.000000 |
| 46758 | 9.320000 | 9.320000 | 9.320000 | +0.000000 |
| 46759 | 10.230000 | 10.230000 | 10.230000 | +0.000000 |
| 46760 | 9.958000 | 9.958000 | 9.958000 | +0.000000 |
| 46761 | 8.487000 | 8.487000 | 8.494000 | +0.007000 |
| 46762 | 8.589000 | 8.589000 | 8.620000 | +0.031000 |
| 46763 | 8.299000 | 8.299000 | 8.322000 | +0.023000 |
| 46764 | 8.195000 | 8.195000 | 8.157000 | -0.038000 |
| 46765 | 8.468000 | 8.468000 | 8.481000 | +0.013000 |
| 46766 | 8.576000 | 8.576000 | 8.563000 | -0.013000 |
| 46767 | 8.400000 | 8.400000 | 8.358000 | -0.042000 |
| 46768 | 8.718000 | 8.714000 | 8.724000 | +0.008000 |

## Invalid and diagnostic runs

1. The first population-8/population-16 A/B was invalid because the `agillm4_floor` preset silently restored the ten-source streaming mixture; every matched branch step used a different batch.
2. The sigma sweep remained useful within antithetic events. Signal appeared in 0/14 events at sigma 0.0003, 1/14 at 0.001, 2/14 at 0.003, 2/14 at 0.01, 3/14 at 0.03 and 11/14 at 0.1. Sigma 0.1 caused large discontinuous route changes, so it is diagnostic, not a safe production default.
3. Exact-batch replay over 84 commits and a controlled replay over 56 commits exposed substantial CUDA/8-bit numerical drift. Both harnesses correctly refused active comparisons. The final 14-commit probe used two baselines and measured the residual noise directly.

## Billing

All project A100 jobs cost an estimated **$2.125** using conservative per-job minute rounding, versus **$1.971** from raw running seconds. This excludes only a negligible three-second CPU mount canary. No project job remained active.

## Interpretation

Fine-tuning and pretraining are both additional optimisation, so EGGROLL could accelerate pretraining in principle. The decisive quantity is useful improvement per compute. For differentiable next-token loss, backpropagation extracts dense gradient information from each token. This router-only sidecar added population forwards; here its average loss movement was statistically unresolved while wall-clock cost rose sharply.

Promising future uses remain non-differentiable post-training rewards, continuous router-margin objectives, adaptive per-router sigma, broader low-rank targets, or a specialised batched inference engine. The current code is useful experimental infrastructure, not a production accelerator.
