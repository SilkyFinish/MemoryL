#!/usr/bin/env bash
# Formal Turbo VL-only training on four allocated A100 GPUs.
set -euo pipefail

STAGE="startup"
trap 'rc=$?; if ((rc != 0)); then printf "[FAILED] stage=%s exit=%s run=%s\n" "$STAGE" "$rc" "${RUN_ROOT:-not-created}" >&2; fi' EXIT
printf '[%s] Turbo launcher entered (pid=%s)\n' "$(date -Is)" "$$"

ROOT="/mnt/data/dylan/d2/memory_l/TTTchoiceMem"
PY="/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-memoryvla/bin/python"
DATA_ROOT="/mnt/data/dylan/d2/memory/RMBench/data"
TURBO_ASSETS="$ROOT/runs/turbo_vl_core_20261017/turbo_formal_assets.json"

TASKS=(put_back_block)
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-5000}"

for value in "$MAX_STEPS" "$SAVE_EVERY" "$SNAPSHOT_EVERY"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        printf 'Step budgets and checkpoint intervals must be positive integers.\n' >&2
        exit 1
    }
done
[[ -x "$PY" ]] || { printf 'Python not found: %s\n' "$PY" >&2; exit 1; }
[[ -f "$TURBO_ASSETS" ]] || {
    printf 'Asset manifest not found: %s\n' "$TURBO_ASSETS" >&2
    exit 1
}

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export RMBENCH_CODEC_PATH="$DATA_ROOT/decode_image_bit.py"

RUN_ROOT="$ROOT/runs/turbo4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/runs"
mkdir "$RUN_ROOT"
MANIFEST="$RUN_ROOT/data_manifest.json"

export HOME="$RUN_ROOT/cache/home"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export HF_HOME="$RUN_ROOT/cache/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export CUDA_CACHE_PATH="$RUN_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/cache/torch_extensions"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" \
    "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

exec > >(tee "$RUN_ROOT/launcher.log") 2>&1
printf 'Run directory: %s\nPython: %s\nTotal optimizer steps: %s\n' "$RUN_ROOT" "$PY" "$MAX_STEPS"
STAGE="gpu-preflight"
printf '[%s] Checking PyTorch and four allocated A100 GPUs (120s timeout)\n' "$(date -Is)"

# Preserve scheduler/container GPU visibility; never select host GPUs implicitly.
timeout --kill-after=10s 120s "$PY" -I -B - <<'PY'
print("[preflight] Importing PyTorch...", flush=True)
import torch

print(f"[preflight] PyTorch {torch.__version__}; querying CUDA device count...", flush=True)
count = torch.cuda.device_count()
print(f"[preflight] Visible CUDA devices: {count}", flush=True)
if count != 4:
    raise SystemExit(
        f"Expected exactly four allocated A100 GPUs, but {count} GPUs are visible. "
        "Run inside the allocated four-GPU environment, or set CUDA_VISIBLE_DEVICES "
        "to the four GPU IDs assigned to this job."
    )
for index in range(count):
    print(f"[preflight] Initializing visible GPU{index}...", flush=True)
    info = torch.cuda.get_device_properties(index)
    if "A100" not in info.name:
        raise SystemExit(f"Visible GPU{index} is {info.name}, not an A100; refusing to launch.")
    print(f"GPU{index}: {info.name}, {info.total_memory / 1024**3:.1f} GiB", flush=True)
PY

# Hide GPUs from TensorFlow in every rank before importing the training entrypoint.
ENTRY='
import os
import pathlib
import runpy
import sys

entry = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), *sys.argv[2:]]
rank = os.environ.get("RANK", "prepare")
print(f"[entry rank={rank}] Importing TensorFlow for CPU-only use...", flush=True)

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

print(f"[entry rank={rank}] Starting {entry.name}", flush=True)
runpy.run_path(str(entry), run_name="__main__")
'

STAGE="prepare-data"
printf '[%s] Preparing data manifest (CPU stage; no training GPU memory expected yet)\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES="" "$PY" -I -B -c "$ENTRY" \
    "$ROOT/train_rmbench.py" \
    --prepare-only \
    --data-root "$DATA_ROOT" \
    --tasks "${TASKS[@]}" \
    --manifest "$MANIFEST" \
    --cameras head_camera \
    --horizon 16 --execute-prefix 4 --group-size 16 \
    --history-sampling random --image-aug \
    --seed 42 --validation-fraction 0.1 \
    2>&1 | tee "$RUN_ROOT/prepare.log"

# DDP: four ranks x group16 x accumulation1 = global batch64.
STAGE="distributed-training"
printf '[%s] Starting four-rank training\n' "$(date -Is)"
"$PY" -I -B -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --max-restarts=0 \
    --no-python "$PY" -I -B -c "$ENTRY" \
    "$ROOT/train_multitask.py" \
    --backbone turbo_interaction \
    --backbone-initialization pretrained \
    --backbone-initialization-seed 42 \
    --base-vlm-id prism-dinosiglip-224px+7b \
    --model-path "$TURBO_ASSETS" \
    --unfreeze-backbone \
    --action-head diffusion --diffusion-model-type DiT-L \
    --dtype bfloat16 --device rankauto --cpu-threads 2 \
    --gradient-checkpointing --find-unused-parameters \
    --data-root "$DATA_ROOT" \
    --tasks "${TASKS[@]}" \
    --manifest "$MANIFEST" \
    --run-dir "$RUN_ROOT/train" \
    --cameras head_camera \
    --horizon 16 --execute-prefix 4 \
    --group-size 16 --per-device-batch-size 16 \
    --global-batch-size 64 --grad-accumulation 1 \
    --history-sampling random --image-aug \
    --sampling-strategy episode \
    --mem-length 16 --retrieval-layers 2 --per-token-size 256 \
    --repeated-diffusion-steps 4 --no-mask-padding-loss \
    --inference-timestep-mode query \
    --optimization-recipe memoryvla \
    --learning-rate 2e-5 --weight-decay 0 --gradient-clip-norm 1 \
    --seed 42 --validation-fraction 0.1 \
    --max-steps "$MAX_STEPS" \
    --save-every "$SAVE_EVERY" --snapshot-every "$SNAPSHOT_EVERY" \
    2>&1 | tee "$RUN_ROOT/train.log"

printf '\nTraining finished. Run directory: %s\n' "$RUN_ROOT"
