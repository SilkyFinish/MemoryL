#!/bin/bash
set -euo pipefail

configuration_arguments=()
resume_arguments=()
dry_run=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action-head|--workers-per-gpu)
            if [ "$#" -lt 2 ] || [[ "$2" = --* ]]; then
                echo "Error: $1 requires a value." >&2
                exit 2
            fi
            configuration_arguments+=("$1" "$2")
            shift 2
            ;;
        --cameras)
            configuration_arguments+=("$1")
            shift
            camera_count=0
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                configuration_arguments+=("$1")
                camera_count=$((camera_count + 1))
                shift
            done
            if [ "$camera_count" -eq 0 ]; then
                echo "Error: --cameras requires an ordered camera list." >&2
                exit 2
            fi
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --resume)
            if [ "$#" -lt 2 ] || [[ "$2" = --* ]]; then
                echo "Error: --resume requires an explicit checkpoint path." >&2
                exit 2
            fi
            resume_arguments=(--resume "$2")
            shift 2
            ;;
        *)
            echo "Usage: bash run_qwen35_oft_a8008.sh [--action-head oft|diffusion] [--cameras CAMERA ...] [--workers-per-gpu 1|2|3] [--dry-run] [--resume CHECKPOINT]" >&2
            exit 2
            ;;
    esac
done

if [ ! -d "/mnt/data/dylan/d2/memory/TTTchoiceMem" ] || [ ! -d "/mnt/data/dylan/d2/memory/RMBench/data" ]; then
    echo "Error: the shared project or RMBench data mount is unavailable." >&2
    exit 1
fi
if [ ! -x "/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python" ] || \
   [ ! -x "/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python" ]; then
    echo "Error: the two project Python environments are not accessible on this worker." >&2
    exit 1
fi

cd "/mnt/data/dylan/d2/memory/TTTchoiceMem"

# Keep both model and simulator interpreters explicit; do not activate Lingbot's va environment.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ "$dry_run" = true ]; then
    exec /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -u \
        /mnt/data/dylan/d2/memory/TTTchoiceMem/configure_training_job.py \
        "${configuration_arguments[@]}" --dry-run
fi

plan_path="$(/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python \
    /mnt/data/dylan/d2/memory/TTTchoiceMem/configure_training_job.py "${configuration_arguments[@]}")"

/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python - "$plan_path" <<'PYTHON'
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/dylan/d2/memory/TTTchoiceMem")
from run_training_job import TrainingJobPlan, check_training_gpu_memory
from eval_multitask import check_gpu_utilization, check_port_free
from configure_training_job import validate_cluster_gpu_models

plan = TrainingJobPlan.load(Path(sys.argv[1]))
check_gpu_utilization(plan.allowed_gpus)
check_training_gpu_memory(plan)
for port in (plan.training_port(), *plan.evaluation_config.ports):
    check_port_free("127.0.0.1", port)

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5
import causal_conv1d
import fla

if torch.__version__ != "2.6.0+cu124":
    raise RuntimeError(f"Expected the verified Qwen torch2.6.0+cu124 environment, got {torch.__version__}")
required_assets = (
    "/mnt/data/dylan/d2/memory/TTTchoiceMem/assets/Qwen3.5-0.8B/config.json",
    "/mnt/data/dylan/d2/memory/TTTchoiceMem/assets/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors",
    "/mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/rmbench10_h16_p4_g16_host64_20261013.json",
)
for asset_path in required_assets:
    if not Path(asset_path).is_file():
        raise FileNotFoundError(asset_path)
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
)
gpu_names = {}
for line in result.stdout.splitlines():
    gpu_index, gpu_name = line.split(",", 1)
    gpu_names[int(gpu_index.strip())] = gpu_name.strip()
selected_gpu_names = validate_cluster_gpu_models(plan.allowed_gpus, gpu_names)
print(json.dumps({"event": "cluster_gpu_inventory", "gpu_names": selected_gpu_names}), flush=True)
print("Qwen environment, local assets and eight A100/A800 devices are visible; full job safety checks follow.")
PYTHON

CUDA_VISIBLE_DEVICES=0 RMBENCH_RENDER_DEVICE=cuda:0 \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python - <<'PYTHON'
import sys
sys.path.insert(0, "/mnt/data/dylan/d2/memory/RMBench/XPolicyLab")
sys.path.insert(0, "/mnt/data/dylan/d2/memory/RMBench/scripts")
import sapien
from client_server.ws.model_server import PolicyServer
from test_render import Sapien_TEST
Sapien_TEST()
print("Native renderer initialization and shared RPC imports passed on assigned GPU0.")
PYTHON

echo "Starting the selected global128/55k variant; final validation and 500 native episodes are automatic."
echo "Frozen plan: $plan_path"
exec /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -u \
    /mnt/data/dylan/d2/memory/TTTchoiceMem/run_training_job.py \
    --plan "$plan_path" "${resume_arguments[@]}"
