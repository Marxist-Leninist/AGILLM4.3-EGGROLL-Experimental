# AGILLM 4.3 EGGROLL fixed-token probe

Run `hf-a100-eggroll-fixed-probe-step1777105-20260810` used two EGGROLL-disabled repeats and one active branch from checkpoint step 1777105. Every branch consumed 14 commits, batch 4, and 114,688 tokens.

- Exact token/input/schedule contracts passed: **True**
- Mean baseline repeat loss noise: `0.00028571428571425424`
- Mean active loss delta versus baseline mean: `-0.0007857142857144847`
- Active better than both baselines: `3/14` steps
- Active worse than both baselines: `5/14` steps
- Active/baseline training-window ratio: `1.4168067065511134`
- EGGROLL accepted events: `10/14`
- Mean immediate search improvement: `None`
- Mean guard improvement: `None`

**Classification: `mechanistic_signal_not_speedup`**

This is a short matched pretraining probe, not a claim about long-run convergence. It is designed to decide whether active router EGGROLL earns a larger experiment while keeping production v22 untouched.
