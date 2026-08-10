#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${AGILLM43_V25_TRAINER:-$SCRIPT_DIR/agillm43_singlefile_intelligence_v25_eggroll_unified_experimental.py}"
exec python3 "$TRAINER" train "$@" \
  --eggroll_every_steps "${EGGROLL_EVERY_STEPS:-5000}" \
  --eggroll_population "${EGGROLL_POPULATION:-16}" \
  --eggroll_population_chunk "${EGGROLL_POPULATION_CHUNK:-16}" \
  --eggroll_rank "${EGGROLL_RANK:-1}" \
  --eggroll_sigma "${EGGROLL_FALLBACK_SIGMA:-0.0003}" \
  --eggroll_router_sigma_map "${EGGROLL_ROUTER_SIGMA_MAP:-0=0.03,1=0.03,13=0.001}" \
  --eggroll_lr "${EGGROLL_LR:-0.0001}" \
  --eggroll_min_pair_signal "${EGGROLL_MIN_PAIR_SIGNAL:-0.0000001}" \
  --eggroll_min_accept_improvement "${EGGROLL_MIN_ACCEPT_IMPROVEMENT:-0.000001}" \
  --eggroll_guard_crops "${EGGROLL_GUARD_CROPS:-2}" \
  --eggroll_max_update_rms_ratio "${EGGROLL_MAX_UPDATE_RMS_RATIO:-0.005}" \
  --eggroll_dry_run
