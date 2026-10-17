# 第一阶段：MemoryVLA + RMBench 原生适配

本开发副本的主路线是 **原版 MemoryVLA VLM + 感知/认知双 Bank + DiT-L**，在 RMBench 原生数据与模拟器接口上训练、rollout。先建立尽可能接近原方法的基线，再研究 TurboVLA 的 VL interaction 替换；**本阶段没有接入 Turbo，也没有新增 TTT、学习式记忆选择或 OFT 方法**。

修改仅在 `/mnt/data/dylan/d2/memory_l/TTTchoiceMem`。原工程 `/mnt/data/dylan/d2/memory/TTTchoiceMem`、参考源码和历史训练/评测产物不作为本副本的输出目录，也未被本轮改写。

## 方法基准与默认配置

参考版本：MemoryVLA `openvla-codebase`，commit `d732ea9072bc063399ccc817aed74ab172eb50be`。上游介绍见[只读参考 README](../../memory/参考/MemoryVLA/README.md)及[项目网页](https://shihao1895.github.io/MemoryVLA)。本副本保留原 [vla/](vla/)、[prismatic/](prismatic/)、[action_model/](action_model/) 的架构与计算，由独立的 [choice_mem/](choice_mem/) 适配原生 RMBench。Prismatic 构造工厂另有默认值不变的离线随机初始化选项，仅供下述显式工程调试模式使用；只读参考仓库不修改。

| 项目 | 第一阶段默认 |
| --- | --- |
| VLM | 原 DINO + SigLIP / Prismatic / Llama，必须显式指定原资产 |
| 动作头 | 原 ActionModel / DiT-L，输出 14D 原生绝对关节目标 |
| 主干训练 | 全参数，不默认冻结 |
| 训练历史 | 每 episode 随机无放回选 16 个有效时刻并排序；不足重复最后有效时刻 |
| 图像增强 | train-only crop → brightness → contrast → saturation → hue |
| 扩散目标 | repeat4，直接调用原 ActionModel.loss，尾部 padding 不进入 mask 排除 |
| 记忆 | Per/Cog 分开读取，容量 16、读取 2 层、时间编码、门控、相邻 ToMe 合并 |
| 推理时间 | 查询计数 0、1、2…；runtime 仍检查真实环境动作计数防止重复消费 |
| 优化 | AdamW，统一 LR 2e-5、weight decay 0、constant、clip 1 |
| 观察与执行 | head_camera 单视角、224×224、预测 16、执行前 4、DDIM 10、CFG 1.5 |

上述视角、horizon、执行前缀和训练预算是显式实验选择，不是“RMBench 必须如此”。完整双 Bank 机制没有简化为 KV 缓存，旧历史仍 detach，不进行跨历史帧 BPTT。默认组长等于容量，组内训练不覆盖第 17 条之后的合并；长 rollout 才会触发容量合并。

## 保留的适配与明确差异

- 原生 HDF5、RMBench RGB codec、episode 级训练/验证划分与数据哈希。
- 14D 顺序为左臂 6 + 左夹爪 + 右臂 6 + 右夹爪；两个夹爪连续，不套用旧 7D 单夹爪二值化。
- 当前原生数据契约为 `obs[t] -> joint_action/vector[t+1:t+1+H]`；最后一帧不作 query，尾部重复末目标。`valid_mask` 保留为数据事实，但默认原版扩散目标不使用它。
- 训练集 q01/q99 统计、全通道归一化、严格 checkpoint、可恢复采样和回合隔离。原版也使用 q01/q99，这不是新增方法。
- 本地 HDF5/Pillow/NumPy 数据管道与 DDP 编排不是上游 RLDS/FSDP 的逐字复制。增强参数/顺序对齐，但不宣称与 TensorFlow/dlimp 随机数、插值逐位等价。
- 可用原 VLM 预训练 run、原 VLA `.pt` 或本地 audited asset manifest 初始化；**仅加载 VLM 模块，压缩器、双 Bank 与 14D 动作头重新初始化**。若使用 LIBERO 微调后的 VLM，这属于初始化来源差异，需在实验中说明，不应称为 OpenVLA 预训练初始化的完全复现。

## 可选：随机冻结 VLM 的工程闭环模式

`--backbone-initialization random-frozen-debug --backbone-initialization-seed 42 --freeze-backbone` 显式启用调试路径。它保持原 DINO + SigLIP + projector + Llama-2 7B 架构，但不读取或下载任何预训练大权重；`--model-path` 只消费本地 manifest 中经过哈希校验的 `llama_metadata` 小型资产。正式默认仍是 `pretrained` 和全量微调。

调试模式将整个 VLM 冻结并保持 BF16/eval，压缩器、双 Bank 和 DiT-L 保持 FP32 可训练参数。它不缩小模型，也不改变原数据、损失或动作协议。checkpoint 保存实际随机权重和完整恢复状态，seed 不能代替权重。服务必须显式传 `--allow-debug-checkpoint`；正式评测 suite 不接受该模式冒充预训练方法分数。

2026-10-17已实际通过随机冻结模式的**工程闭环**：3个optimizer steps（step1暂停后恢复至step3）、完成checkpoint检查、真实本地RPC推理及共享GPU7原生回合。固定任务 `observe_and_pickup / demo_clean`、seed900001，原生exit0、101个动作步、success=true，视频102帧可解码，源码/数据/metadata复核通过。详见[工程检验记录](runs/random_frozen_closed_loop_20261017/REPORT.md)及[最终验收JSON](runs/random_frozen_closed_loop_20261017/single_gpu_verification.json)。这不是正式方法成功率；正常任务失败也可通过工程验收，setup/OOM/RPC/超时异常则不能。本次共享GPU期间设备free最低仅23MiB，不代表共享部署显存稳定或适合长期运行。原计划见 [closed_loop_plan.json](runs/random_frozen_closed_loop_20261017/closed_loop_plan.json)。

## 文档与入口

- [模型、数据和兼容边界](README_lightweight.md)
- [准备、训练、恢复与 rollout 命令](README_full_finetuning.md)
- [train_rmbench.py](train_rmbench.py)：数据清单准备、单进程训练。
- [train_multitask.py](train_multitask.py)：可恢复 DDP 训练。
- [configure_training_job.py](configure_training_job.py)、[run_training_job.py](run_training_job.py)：独立计划与作业管理；新阶段不默认套用旧 Qwen 自动评测计划。
- [serve_rmbench.py](serve_rmbench.py)、[eval_rmbench.py](eval_rmbench.py)、[eval_multitask.py](eval_multitask.py)：原生 XPolicyLab 服务与 rollout，不修改共享模拟器源码。

Qwen/OFT、sequential 历史、masked loss 和 action-step PE 仅保留为**显式 legacy/消融选项**，不属于当前主路线。旧 v1 checkpoint 会被迁移成其原有方法配置，不会因新默认值改变训练时的语义。不要直接执行历史 `run_qwen35_*` shell 脚本来启动本阶段。

## 验收边界

CPU回归使用feature fixtures、小型DiT和mock原资产，检查数学/协议/恢复/接口；整套结果为377通过、1跳过、130 subtests通过，不能代替真实大模型或模拟器验收。另已实际完成原架构随机冻结模式的3步GPU训练与恢复，训练allocator峰值allocated约32.85GiB，并从最终checkpoint完成真实RPC推理。这不等于正式预训练全量微调验收，也没有取得正式RMBench成功率。真实训练前仍须核对独立环境、资产、显存和分布式条件，不能沿用旧Qwen0.8B估计。当前DDP在正式全量训练时每个rank复制完整FP32参数、梯度与AdamW状态，增加卡数不会分摊它们；7B级全量训练的静态状态就可能超过单张80GB显存。本阶段未改造成上游FSDP，未验收全量训练显存可行性。

本地资产清单为 `/mnt/data/dylan/d2/memory/RMBench/attention-test-memoryvla/assets/manifest.json`。本次随机冻结检验只使用其中经过校验的Llama config/tokenizer小型资产，没有读取、哈希或加载预训练大权重；不能据此宣称该清单中的预训练权重或正式全量训练已获验收。

## 上游引用

本项目基于 [MemoryVLA](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase)，上游论文分数不是本阶段 RMBench 实验结果。使用上游工作请引用：

```bibtex
@article{shi2025memoryvla,
  title={MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation},
  author={Shi, Hao and Xie, Bin and Liu, Yingfei and Sun, Lin and Liu, Fengrong and Wang, Tiancai and Zhou, Erjin and Fan, Haoqiang and Zhang, Xiangyu and Huang, Gao},
  journal={arXiv preprint arXiv:2508.19236},
  year={2025}
}
```
