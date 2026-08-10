---
library_name: pytorch
tags:
- language-model
- evolution-strategies
- eggroll
- mixture-of-experts
- experimental
license: other
---

# AGILLM 4.3 EGGROLL Experimental

<!-- V25-UNIFIED-START -->
## v25 unified EGGROLL experimental runtime — 10 August 2026

**Measured verdict:** EGGROLL produced genuine MoE-router fitness signal, but **did not demonstrate faster AGILLM-4.3 pretraining**. In the fixed-token probe the active branch took **41.7% longer** than the mean baseline training window. Production v22 remains unchanged, and no long-running EGGROLL checkpoint fork was started.

v25 merges the two former v24 branches into one coherent file:

- strict fixed-token input plus SHA-256 receipts;
- calibrated per-router perturbation scales;
- measurable-improvement acceptance, so numerical no-change is rejected;
- route-flip, router-margin and perturbation diagnostics;
- exact rollback, clipping and default-off behaviour.

Files:

- [`agillm43_singlefile_intelligence_v25_eggroll_unified_experimental.py`](agillm43_singlefile_intelligence_v25_eggroll_unified_experimental.py)
- [`launch_v25_calibrated_dry_run.sh`](launch_v25_calibrated_dry_run.sh)
- [`RESULTS_20260810.md`](RESULTS_20260810.md)
- [`EXPERIMENT_RESULTS_20260810.json`](EXPERIMENT_RESULTS_20260810.json)
- complete immutable bundle: [`validation/2026-08-10-v25/`](validation/2026-08-10-v25/)

SHA-256: `8b9e62eaa9dde68c624ea9f3d4c9311b4a841ec43a89761453754e0673e25674`
<!-- V25-UNIFIED-END -->

Default-off EGGROLL experiments for AGILLM 4.3's top-1 Mixture-of-Experts routers, inspired by [Evolution Strategies at the Hyperscale](https://eshyperscale.github.io/imgs/paper.pdf). Normal AdamW, AR, SAT, NAT and DiffusionBlock training remain intact.

## Experiment verdict, 10 August 2026

A paid, fixed-token Hugging Face A100 probe from checkpoint step 1,777,105 found **mechanistic antithetic signal but no demonstrated pretraining speedup**.

- Exact tensor hashes matched across two baselines and one active branch.
- Active EGGROLL was **41.7% slower** over the measured training window.
- Mean local-loss delta was `-0.000786`, median zero, 95% interval crossed zero widely, sign-test p `0.727`.
- Better than both baselines on 3/14 commits, worse on 5/14, tied on 6/14.
- Ten updates passed non-regression, but none improved guard CE measurably.

**Decision:** keep it disabled by default and do not fork an active checkpoint from this evidence. See [`HF_EXPERIMENT_RESULTS_20260810.md`](HF_EXPERIMENT_RESULTS_20260810.md) and [`results/2026-08-10-hf-fixed-probe/`](results/2026-08-10-hf-fixed-probe/).

## Locations

- GitHub: `Marxist-Leninist/AGILLM4.3-EGGROLL-Experimental`
- Canonical Hugging Face: `MarxistLeninist/AGILLM-4.3-EGGROLL-Experimental`
- Legacy mirror: `OpenTransformer/AGILLM-4.3-EGGROLL-Experimental`

## Versions

### v23 original optional sidecar

`agillm43_singlefile_intelligence_v23_eggroll_experimental.py`, SHA-256 `f487cf2fbb38b69bf9f19322759176c3e4b66636a6375fc30759434be22aab80`.

Includes antithetic low-rank perturbations, candidate-major evaluation, search and guard crops, clipping, rollback, dry-run mode and JSONL receipts. Disabled when `--eggroll_every_steps 0`.

### v24 strict fixed-token runtime

`agillm43_singlefile_intelligence_v24_eggroll_fixed_tokens.py`, SHA-256 `e3c89682ed04e8d5419cf30e814577b50def3346d3546a4eb27f37acd78f19cf`.

Adds `--fixed_tokens_file`, `--fixed_tokens_sha256`, `--[no-]fixed_tokens_cycle`, bypasses presets and hot dataset reload, and prints file/token-stream SHA-256 receipts. Test-only deterministic CUDA settings use `AGILLM43_EXPERIMENTAL_DETERMINISTIC=1`.

## Frozen base

- Step `1,777,105`
- Tokens `87,348,264,960`
- Checkpoint SHA-256 `72ee3cf3ae4e3dedb22f16d8c2d21c80c66e0bdcd2001f606f030886cc98e2af`

Production v22 was not modified.

## Files

- `agillm43_singlefile_intelligence_v23_eggroll_experimental.py`
- `agillm43_singlefile_intelligence_v24_eggroll_fixed_tokens.py`
- `experiments/hf-fixed-probe/fixed_active_probe.py`
- `HF_EXPERIMENT_RESULTS_20260810.md`
- `results/2026-08-10-hf-fixed-probe/`
- `EXPERIMENT.md`, `EXPERIMENT_MANIFEST.json`, and canonical HF `base_checkpoint/`

## Limits

The valid probe covered 14 commits and 114,688 tokens per branch, population 16, rank 1, sigma 0.1, router-only. It argues against enabling this configuration as a pretraining accelerator; it does not rule out every EGGROLL target, engine or non-differentiable objective.
