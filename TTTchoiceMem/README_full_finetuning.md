# 第一阶段操作手册：MemoryVLA + RMBench

本手册面向开发副本 `/mnt/data/dylan/d2/memory_l/TTTchoiceMem`。旧 Qwen 55k 计划和原工程的 runs 不属于本阶段。下面给出准备与启动入口，**本轮代码修改没有执行完整 GPU 训练或模拟器 rollout**。

## 1. 环境、资产与资源边界

现存环境候选：

| 用途 | 实际解释器 | 已检查范围 |
| --- | --- | --- |
| 原 VLM | `/mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python` | torch 2.4.1、transformers 4.40.1、timm 0.9.10、TF 2.15.0；本副本 Prismatic factory 与 TF 的 CPU import 通过，未加载模型 |
| CPU 回归 | `/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35/bin/python` | 用于 pytest；不代表兼容原 VLM 全部依赖，该环境缺 TensorFlow |
| 原生模拟器 | `/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python` | 路径存在；本轮未运行 SAPIEN/reset/render/rollout |

不自动安装、升级或卸载共享环境依赖。不要通过关闭 protobuf 等兼容性保护来迁就旧环境；遇到依赖错误应单独核对版本。

本地原资产候选：

```text
/mnt/data/dylan/d2/memory/RMBench/attention-test-memoryvla/assets/manifest.json
```

该清单指向原 LIBERO-100 微调 checkpoint、Llama metadata 和本地视觉权重。加载器核对哈希，仅恢复 VLM 模块；不会复用 LIBERO 的 7D 动作头、旧 Bank 或动作统计。此初始化来源须在实验记录中明确；如要匹配原 OpenVLA 预训练初始化，应提供相应原 VLM run，而不是把 LIBERO 微调资产当作无差异替代。

**显存限制：** 当前训练器是 DDP，不是上游 FSDP。每 rank 都复制完整 FP32 主权重、梯度、AdamW 两份 moment；7B 级模型仅这些状态就可能超过 80GB，尚未加激活、记忆与 DiT-L。8 卡 DDP 不等于把模型切成八份。本轮没有新增分片/卸载训练、没有测峰值显存；下面的 DDP 命令是接口示例，不是已验证可在现有卡上运行的资源承诺。不能以冻结主干、换小头等方法变化偷偷规避这个限制后仍称全量原版基线。

实际启动前检查已分配 GPU 的进程、显存和端口；不停止他人进程，不沿用旧 Qwen 的资源阈值。未经资源确认，先使用 prepare-only / dry-run。

## 2. 只准备新数据清单，不加载模型

下面使用所有十个原生任务；也支持显式非空任务子集。新 manifest 使用独立路径，不覆盖旧 sequential/noaug 清单。prepare-only 不导入 torch，不需要 model-path，也不启动训练。

```bash
CUDA_VISIBLE_DEVICES='' /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/train_multitask.py \
  --data-root /mnt/data/dylan/d2/memory/RMBench/data \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --manifest /mnt/data/dylan/d2/memory_l/TTTchoiceMem/protocols/memoryvla_stage1_h16_p4_g16_head_random_20261017.json \
  --prepare-only
```

新默认为 head_camera、224×224、horizon16、prefix4、group16、随机 episode 采样、train-only 增强。清单创建会读取/统计原数据并校验 codec；对大数据可能耗时，但不执行模型前向。再次使用同路径只接受完全相同的数据与契约，不重新随机划分、不重写。

codec 自动定位只读的 `/mnt/data/dylan/d2/memory/RMBench/data/decode_image_bit.py`。其他挂载可通过 `RMBENCH_CODEC_PATH` 显式指定实际文件；指定但不存在会报错，不会偷偷换 codec。

## 3. 先检查训练计划，不启动

以下是八 rank、每 rank 一组 16、累积 1 的预算示例；30000 是本示例选择，不是新阶段固定要求。run-name 必须换成未使用名称。dry-run 只打印计划，不写计划、不加载 torch/模型、不启动子进程：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/configure_training_job.py \
  --run-name memoryvla_stage1_example_20261017 \
  --data-root /mnt/data/dylan/d2/memory/RMBench/data \
  --manifest /mnt/data/dylan/d2/memory_l/TTTchoiceMem/protocols/memoryvla_stage1_h16_p4_g16_head_random_20261017.json \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --model-path /mnt/data/dylan/d2/memory/RMBench/attention-test-memoryvla/assets/manifest.json \
  --gpu-ids 0 1 2 3 4 5 6 7 --max-steps 30000 --gradient-checkpointing --dry-run
```

确认计划及资源后，去掉 `--dry-run` 才会冻结新计划；这一步仍不训练。真正启动是另一次显式调用：

```text
/mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B /mnt/data/dylan/d2/memory_l/TTTchoiceMem/run_training_job.py --plan <生成器实际打印的绝对计划路径>
```

`<...>` 是需要替换的输出路径，不是可原样执行的 shell 内容。新阶段默认 training-only plan，不因训练退出而自动套用旧 Qwen 的评测计划。原历史计划若指向原工程的训练入口或 runs，会被开发副本的管理器拒绝，避免误操作原实验。

## 4. 直接训练器示例

与管理入口二选一，不要重复启动。下例仅在**已解决第 1 节显存与资源条件**后使用；当前没有验证这些卡可以承载全量模型。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 \
/mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B -m torch.distributed.run \
  --nproc-per-node 8 --master-addr 127.0.0.1 --master-port 18844 \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/train_multitask.py \
  --backbone memoryvla --action-head diffusion --diffusion-model-type DiT-L \
  --unfreeze-backbone --gradient-checkpointing \
  --model-path /mnt/data/dylan/d2/memory/RMBench/attention-test-memoryvla/assets/manifest.json \
  --data-root /mnt/data/dylan/d2/memory/RMBench/data \
  --tasks battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button put_back_block rearrange_blocks swap_T swap_blocks \
  --cameras head_camera \
  --manifest /mnt/data/dylan/d2/memory_l/TTTchoiceMem/protocols/memoryvla_stage1_h16_p4_g16_head_random_20261017.json \
  --run-dir /mnt/data/dylan/d2/memory_l/TTTchoiceMem/runs/memoryvla_stage1_manual_20261017 \
  --group-size 16 --per-device-batch-size 16 --global-batch-size 128 --grad-accumulation 1 \
  --max-steps 30000 --horizon 16 --execute-prefix 4 \
  --mem-length 16 --retrieval-layers 2 --per-token-size 256 \
  --learning-rate 0.00002 --weight-decay 0 --gradient-clip-norm 1 \
  --save-every 5000 --snapshot-every 5000 --validation-fraction 0.1 --seed 42
```

默认随机历史、图像增强、repeat4、unmasked 原扩散目标和 query-index PE 进入 PolicyConfig、协议与 checkpoint。不默认采用旧 Qwen 分层学习率、warmup/cosine、任务均衡和固定步长 query。

**恢复：** 仅在原任务完全退出后，在完全相同的训练命令后加 `--resume latest`，或在管理入口后加 `--resume <完整 checkpoint 绝对路径>`。完整 checkpoint 才能恢复 AdamW/RNG/采样位置，`snapshots/` 不能续训。世界大小、数据 manifest、方法、有效 batch 和预算不得静默改变。

新实验必须使用新 run-dir；只有续训复用原路径。不自动删除旧权重或覆盖已有 checkpoint。

## 5. 模型服务与原生 rollout

训练完成后选用该实验自己的 snapshot；下面的路径对应第 4 节示例，当前尚未生成该文件。先单独完成真实模型和模拟器预检，再跑正式套件。

单服务示例：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/serve_rmbench.py \
  --checkpoint /mnt/data/dylan/d2/memory_l/TTTchoiceMem/runs/memoryvla_stage1_manual_20261017/snapshots/final.pt \
  --rmbench-root /mnt/data/dylan/d2/memory/RMBench --device cuda:0 --port 18820
```

GPU 号仅为示例，必须换成已分配的卡。服务端按 checkpoint 恢复 horizon、prefix、相机和归一化；CLI 参数只能断言，不能覆盖训练契约。客户端使用 `eval_rmbench.py`，真实动作时间来自原 `take_action_cnt`，不改 benchmark 的 success/动作/终止规则。

可先只打印原生客户端命令（本轮已检查该 dry-run，不导入/运行模拟器，也不启动服务）：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/eval_rmbench.py \
  --task-name battery_try \
  --env-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python \
  --sim-gpu 0 --dry-run
```

正式评测入口示例（不要与上面的手动 server 同时启动，suite 自己管理服务）：

```bash
OMP_NUM_THREADS=2 /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python -B \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/eval_multitask.py \
  --checkpoint /mnt/data/dylan/d2/memory_l/TTTchoiceMem/runs/memoryvla_stage1_manual_20261017/snapshots/final.pt \
  --eval-root /mnt/data/dylan/d2/memory_l/TTTchoiceMem/runs/eval_memoryvla_stage1_manual_20261017 \
  --rmbench-root /mnt/data/dylan/d2/memory/RMBench \
  --policy-python /mnt/data/WORKSPACE/anaconda3/envs/mvla_rm/bin/python \
  --env-python /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python \
  --gpu-ids 0 --workers-per-gpu 1 \
  --expected-training-steps 30000 --expected-action-head diffusion --expected-cameras head_camera \
  --episodes 50 --seed 0 --instruction-type unseen --task-config demo_clean --max-seed-attempts 2500
```

`--policy-python` 是旧 `--qwen-python` 的通用别名。未指定 tasks 时选官方全部任务；显式子集必须来自官方任务表，完成率分母按所选任务计算。预期训练步数需与 snapshot 中预算和实际完成步数一致。当前 native suite 沿用 horizon16/prefix4 的执行协议并提前断言；其他 horizon/prefix 可训练和通过通用完成检查，但不能不经独立评测协议核对就交给该 suite。

正式套件保留每任务 20 或 50 个有效策略回合及原生 50 倍候选池。expert setup 失败记为跳过候选，不冒充策略失败；策略成功失败都计数，不按结果重跑。模型/模拟器资源错误不能算正常 benchmark 失败；覆盖不足明确标为 incomplete。

## 6. CPU 回归与尚未验收项

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
/mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35/bin/python -I -B -m pytest -p no:cacheprovider \
  /mnt/data/dylan/d2/memory_l/TTTchoiceMem/tests -q
```

`-I` 隔离 Python 路径，`-B` 禁止写 pyc；不生成 pytest cache、不使用 GPU。可选真实网络 smoke 默认跳过。

本轮验证边界：CPU fixture、损失/Bank 原源码对照、mock 资产和训练/评测协议；已有原 VLM 环境的 factory import 另作检查。没有加载 33GB 原 checkpoint、没有完整训练/恢复大模型、没有验证峰值显存、没有真实模拟器 rollout，也没有本阶段成功率。模型/数据边界详见[说明文档](README_lightweight.md)。
