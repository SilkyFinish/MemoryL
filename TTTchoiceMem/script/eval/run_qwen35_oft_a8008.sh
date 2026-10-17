#!/bin/bash
set -euo pipefail

if [ ! -d "/mnt/data/dylan/d2/memory/TTTchoiceMem" ] || [ ! -d "/mnt/data/dylan/d2/memory/RMBench" ]; then
    echo "Error: the shared project or RMBench mount is unavailable." >&2
    exit 1
fi
if [ ! -x "/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python" ] || \
   [ ! -x "/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python" ]; then
    echo "Error: the two project Python environments are unavailable." >&2
    exit 1
fi
cd "/mnt/data/dylan/d2/memory/TTTchoiceMem"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The configurator derives the same independent paths and forwards only validated arguments.
exec /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python \
    /mnt/data/dylan/d2/memory/TTTchoiceMem/configure_training_job.py --evaluate-only "$@"
