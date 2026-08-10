# v25 unified calibrated EGGROLL experimental runtime

Date: 10 August 2026

## Combined

- v24 strict fixed-token input and reproducibility receipts.
- Calibrated per-router EGGROLL perturbation controls.

## Added

- `--eggroll_router_sigma_map`.
- `--eggroll_min_accept_improvement`.
- Target-specific sigma and acceptance-reason receipts.
- Route-flip and router-margin diagnostics.
- Fixed-token cycling checks in the permanent smoke test.

## Fixed

- One global sigma for routers with different margins.
- Acceptance of numerically unchanged updates.
- Split feature branches both using the v24 label.

## Safety and compatibility

- EGGROLL remains disabled by default.
- The recommended launcher always appends `--eggroll_dry_run`.
- Production v22 is untouched.
- Exact EGGROLL continuation state is required only for the v25 profile itself.

SHA-256: `8b9e62eaa9dde68c624ea9f3d4c9311b4a841ec43a89761453754e0673e25674`
