# Qwen + MemoryVLA：全量训练与评测命令

更新日期：2026-10-14。十任务联合训练一套模型，统一预算55,000次参数更新。

- **公司集群**：8张A100/A800，每卡16、累积1、全局128，OFT、单主相机，约8.83亿可训练参数。
- **本机DiT-L**：4卡、全局64、三相机，约12.88亿参数；20261013这轮已完成55k，实际训练20小时2分36秒。
- 两者均为FP32主权重、BF16前向、全参数DDP和梯度检查点；预测16个14维动作、执行前4个。

> **不要重复启动已有任务。** 同配置的新一轮必须更换计划及管理/训练/评测输出路径，见第3节；续训才用`--resume`。本文更新不启动任何任务。

快速查找：第2节是集群启动入口，第3节是新一轮自动训练/评测，第4节保留两套完整训练参数，第5节恢复，第6节评测，第7节查看结果。

## 1. 环境与启动前检查

直接使用完整解释器路径，无需`conda activate`或`conda run`，不要用其他项目的`mem0_rm`环境代替：

```text
训练/模型服务：/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python
模拟器：      /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python
```

前者继承conda`turbomem2`，后者继承`RMBench2`；`host512`只是历史环境名。依赖已安装，不重新执行旧`pip install -e .`，不在运行中升级Python/Triton。

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

只启动在已分配且利用率低于50%、每卡空闲至少35GiB的GPU上；检查训练端口18844和评测端口，无权停止他人任务。直接训练命令不代替这些检查。

## 2. 公司集群启动：训练完成后自动评测

在平台启动命令框粘贴以下内容。外层兼容`/bin/sh`，**不写`set -euo pipefail`**，显式用Bash运行脚本；内部已固定八卡和完整训练参数：

```bash
export PATH=/mnt/data/WORKSPACE/anaconda3/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /mnt/data/dylan/d2/memory/TTTchoiceMem && \
exec /bin/bash /mnt/data/dylan/d2/memory/TTTchoiceMem/script/train/run_qwen35_oft_a8008.sh \
  --action-head oft \
  --cameras head_camera \
  --workers-per-gpu 3
```

**上面命令对应的20261013输出路径已被使用，不能再次作为新实验启动。** 新一轮按第3节生成独立计划；脚本名中的`a8008`不限制A100。

- `--action-head oft|diffusion`：切换动作头；diffusion使用DiT-L，OFT不要求DiT/DDIM/CFG参数。
- `--cameras head_camera`：单主视角；两路用`head_camera right_camera`或`head_camera left_camera`，三路用`head_camera left_camera right_camera`。相机列表按顺序保存，改相机使用独立数据清单并从头训练。
- `--workers-per-gpu 1|2|3`：只控制后续评测，八卡最多8/16/24回合并发；每worker独立模型和Bank，不保证线性加速。
- 末尾加`--dry-run`：只检查并打印配置，不生成计划、不占用模型显存。

脚本检查挂载、八卡型号/资源、依赖和基础渲染后进入管理进程。训练退出0、完成55k且最终权重校验通过才自动进行十任务各50回合评测。控制台不持续刷步数时，查看第7节的`training.log`，不要重复提交。

## 3. 同配置再训一轮：生成新计划

下面以公司集群OFT/单相机/每卡3个评测worker的`repeat1_20261014`为例。**每轮修改三处输出路径和新计划文件名，后面的启动命令同步改为该新计划；原始模板及既有计划不覆盖。** 本机四卡新实验可改为读取四卡原计划，保留其训练配置和资源分配。

```bash
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python - <<'PYTHON'
import json
from pathlib import Path
import sys

sys.path.insert(0, "/mnt/data/dylan/d2/memory/TTTchoiceMem")
from configure_training_job import CLUSTER_TEMPLATE, build_cluster_variant, validate_variant
from choice_mem.data import write_manifest_once

plan = build_cluster_variant(json.loads(CLUSTER_TEMPLATE.read_text()), "oft", ["head_camera"], 3)
plan["job_root"] = "/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/job_qwen35_oft_rmbench10_global128_repeat1_20261014"
plan["run_dir"] = "/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_global128_repeat1_20261014"
plan["eval_root"] = "/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/eval_qwen35_oft_rmbench10_global128_repeat1_20261014"
plan["training_command"][plan["training_command"].index("--run-dir") + 1] = plan["run_dir"]
for field_name in ("job_root", "run_dir", "eval_root"):
    if Path(plan[field_name]).exists():
        raise FileExistsError(plan[field_name])
validate_variant(plan)
write_manifest_once(Path("/mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/qwen35_oft_global128_repeat1_20261014_plan.json"), plan)
PYTHON
```

确认第1节资源和原生渲染可用后，在获得八卡分配的节点执行；这是管理入口，仍会自动评测，不要与第2节同时运行：

```bash
export PATH=/mnt/data/WORKSPACE/anaconda3/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd /mnt/data/dylan/d2/memory/TTTchoiceMem && \
exec /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -u \
  /mnt/data/dylan/d2/memory/TTTchoiceMem/run_training_job.py \
  --plan /mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/qwen35_oft_global128_repeat1_20261014_plan.json
```

同一命令加`--dry-run`可先检查计划，不训练；该管理入口不重复第2节Shell里的依赖导入和渲染预检，换节点需先确认环境。平台保持前台运行，不加`nohup`、`--background`或`sleep`。交互节点可在独立tmux中运行，`Ctrl-b`再按`d`脱离；tmux不能延长集群分配时间。

## 4. 完整torchrun训练命令（不自动评测）

以下是直接训练器的完整参数，**与管理入口二选一**。示例使用新的`manual1_20261014`目录；路径用过后必须更名。直接训练结束后按第6节单独评测。

### 4.1 八卡OFT、单主相机、全局128

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m torch.distributed.run \
  --nproc-per-node 8 --master-addr 127.0.0.1 --master-port 18844 \
  /mnt/data/dylan/d2/memory/TTTchoiceMem/train_multitask.py \
  --backbone qwen35 --action-head oft --oft-hidden-dim 256 \
  --unfreeze-backbone --gradient-checkpointing \
  --model-path /mnt/data/dylan/d2/memory/TTTchoiceMem/assets/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --data-root /mnt/data/dylan/d2/memory/RMBench/data \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --cameras head_camera \
  --manifest /mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/rmbench10_h16_p4_g16_head_20261013.json \
  --run-dir /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_global128_manual1_20261014 \
  --group-size 16 --per-device-batch-size 16 --global-batch-size 128 --grad-accumulation 1 \
  --max-steps 55000 --horizon 16 --execute-prefix 4 \
  --mem-length 16 --retrieval-layers 2 --per-token-size 256 \
  --backbone-learning-rate 0.00001 --learning-rate 0.0001 \
  --minimum-backbone-learning-rate 0.000001 --minimum-head-learning-rate 0.000005 \
  --weight-decay 0.005 --warmup-ratio 0.05 --gradient-clip-norm 2.5 \
  --save-every 5000 --snapshot-every 5000 --validation-fraction 0.1 --seed 42
```

### 4.2 四卡DiT-L、三相机、全局64

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m torch.distributed.run \
  --nproc-per-node 4 --master-addr 127.0.0.1 --master-port 18844 \
  /mnt/data/dylan/d2/memory/TTTchoiceMem/train_multitask.py \
  --backbone qwen35 --action-head diffusion --unfreeze-backbone --gradient-checkpointing \
  --model-path /mnt/data/dylan/d2/memory/TTTchoiceMem/assets/Qwen3.5-0.8B \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --data-root /mnt/data/dylan/d2/memory/RMBench/data \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --cameras head_camera left_camera right_camera \
  --manifest /mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/rmbench10_h16_p4_g16_host64_20261013.json \
  --run-dir /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_ditL_rmbench10_global64_manual1_20261014 \
  --group-size 16 --per-device-batch-size 16 --global-batch-size 64 --grad-accumulation 1 \
  --max-steps 55000 --diffusion-model-type DiT-L --horizon 16 --execute-prefix 4 \
  --mem-length 16 --retrieval-layers 2 --per-token-size 256 \
  --backbone-learning-rate 0.00001 --learning-rate 0.0001 \
  --minimum-backbone-learning-rate 0.000001 --minimum-head-learning-rate 0.000005 \
  --weight-decay 0.005 --warmup-ratio 0.05 --gradient-clip-norm 2.5 \
  --save-every 5000 --snapshot-every 5000 --validation-fraction 0.1 --seed 42
```

更换相机时同时更换`--manifest`，不覆盖旧清单；数据、模型、学习率等恢复参数必须一致。八卡和四卡命令共用18844端口，不可在同一节点原样并行运行。

## 5. 断点恢复

仅在原任务和所有worker退出、评测尚未开始时使用；恢复到总55k，不追加55k，不改变卡数、动作头、相机或优化器配置。

```bash
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m json.tool /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_a8008_global128_head_w3_20261013/latest_checkpoint.json
```

以下是原集群作业的恢复示例，**第10000步仅为示例，必须换成上面实际存在的路径**；新一轮作业应使用其自己的管理目录`plan.json`：

```bash
export PATH=/mnt/data/WORKSPACE/anaconda3/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /mnt/data/dylan/d2/memory/TTTchoiceMem && \
exec /bin/bash /mnt/data/dylan/d2/memory/TTTchoiceMem/script/train/run_qwen35_oft_a8008.sh \
  --action-head oft --cameras head_camera --workers-per-gpu 3 \
  --resume /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_a8008_global128_head_w3_20261013/checkpoints/checkpoint-step010000.pt
```

通用管理入口是在原`run_training_job.py --plan /完整路径/plan.json`后加`--resume /完整路径/checkpoint-stepXXXXXX.pt`；直接torchrun训练则在第4节**原完整命令**后加`--resume latest`。`snapshots/`不含AdamW/RNG，不能用于续训。

## 6. 独立评测命令

自动评测已在运行时不要另起。下面对应第4节直接训练的新目录；用于其他实验时明确替换checkpoint及全新的评测根目录，不能覆盖旧结果。评测必须使用完成55k且契约校验通过的权重。

### 6.1 八卡OFT、单主相机、每卡3个worker

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python /mnt/data/dylan/d2/memory/TTTchoiceMem/eval_multitask.py \
  --checkpoint /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_global128_manual1_20261014/snapshots/final.pt \
  --eval-root /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/eval_qwen35_oft_rmbench10_global128_manual1_20261014 \
  --rmbench-root /mnt/data/dylan/d2/memory/RMBench \
  --qwen-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python \
  --env-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --gpu-ids 0 1 2 3 4 5 6 7 --workers-per-gpu 3 \
  --expected-training-steps 55000 --expected-action-head oft --expected-cameras head_camera \
  --episodes 50 --seed 0 --instruction-type unseen --task-config demo_clean --max-seed-attempts 2500 --server-timeout 120
```

### 6.2 四卡DiT-L、三相机、每卡1个worker

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python /mnt/data/dylan/d2/memory/TTTchoiceMem/eval_multitask.py \
  --checkpoint /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_ditL_rmbench10_global64_manual1_20261014/snapshots/final.pt \
  --eval-root /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/eval_qwen35_ditL_rmbench10_global64_manual1_20261014 \
  --rmbench-root /mnt/data/dylan/d2/memory/RMBench \
  --qwen-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python \
  --env-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --gpu-ids 4 5 6 7 --workers-per-gpu 1 --ports 18824 18825 18826 18827 \
  --expected-training-steps 55000 --expected-action-head diffusion --expected-cameras head_camera left_camera right_camera \
  --episodes 50 --seed 0 --instruction-type unseen --task-config demo_clean --max-seed-attempts 2500 --server-timeout 120
```

省略`--ports`时新分配默认从18861开始，24路使用18861～18884。多worker需要每卡多份模型/模拟器，默认每worker至少8GiB空闲显存只是安全下限，不是实测峰值保证。十任务各50个有效策略回合，候选seed100000～102499，保留原生expert/动作/成功规则，成功失败都计数，错误不补跑；全500回合覆盖完整才算评测完成。

原集群作业也有快捷入口，仅在原自动评测未运行、目标目录尚未创建时使用；其他新实验使用上面的完整命令并替换实际路径：

```bash
exec /bin/bash /mnt/data/dylan/d2/memory/TTTchoiceMem/script/eval/run_qwen35_oft_a8008.sh \
  --action-head oft --cameras head_camera --workers-per-gpu 3
```

## 7. 查看进度与保存位置

查看已启动的原集群作业，不会重复启动：

```bash
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m json.tool /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_a8008_global128_head_w3_20261013/progress.json
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m json.tool /mnt/data/dylan/d2/memory/TTTchoiceMem/runs/job_qwen35_oft_rmbench10_a8008_global128_head_w3_20261013/status.json
```

第3节新作业的管理、训练、评测根目录分别为：

```text
/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/job_qwen35_oft_rmbench10_global128_repeat1_20261014/
/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/qwen35_oft_rmbench10_global128_repeat1_20261014/
/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/eval_qwen35_oft_rmbench10_global128_repeat1_20261014/
```

- 管理目录：冻结`plan.json`、`status.json`、`training.log`、`evaluation.log`。控制台停止刷日志不代表训练卡住。
- 训练目录：`progress.json`、`metrics.jsonl`；每5000步保存`checkpoints/`完整状态（最近两份）和`snapshots/`推理快照，55k后保存`snapshots/final.pt`。
- 评测目录：整体`summary.json`及各任务记录；多worker的逐回合日志和视频位于`任务名/seed100000/`等子目录。

训练读取已有HDF5图片，不依赖在线渲染；评测需要SAPIEN/Vulkan。ICD警告后出现`Render Well`只说明基础初始化成功，不代替全任务/多worker验收。不要仅因告警升级共享环境，也不能把异常回合算作正常策略失败。

模型和数据语义见[README_lightweight.md](README_lightweight.md)，历史测速与清理记录见[实验日记](../document/experimental%20diary/2026-10-13_TTTchoiceMem_Qwen全量微调_DiTL_全局64_55k.md)。
