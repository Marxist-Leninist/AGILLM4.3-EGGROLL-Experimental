#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hfhome
export HF_DATASETS_CACHE=/tmp/hfhome/datasets
export TRANSFORMERS_CACHE=/tmp/hfhome/transformers
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HF_HOME" /results
python -m pip install -q --no-cache-dir \
  'numpy==1.26.3' \
  'transformers==4.48.0' \
  'tokenizers==0.21.0' \
  'datasets==2.21.0' \
  'bitsandbytes==0.43.3' \
  'zstandard==0.23.0' \
  'huggingface_hub==0.24.7' \
  'tqdm==4.70.0' \
  'safetensors==0.4.5' \
  'sentencepiece>=0.2,<0.3'
python -u /job/fixed_active_probe.py
