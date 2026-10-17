# 模型、数据与兼容边界：MemoryVLA + RMBench

本文说明第一阶段基线，训练/rollout 命令见[操作手册](README_full_finetuning.md)。旧 Qwen 实验的耗时、参数量、已训练步数和分数均不代表当前开发副本。

## 1. 原版当前帧特征

默认 `MemoryVLABackbone` 使用原 DINO + SigLIP、projector 和 Prismatic/Llama，保留原 action-question prompt 与结尾 token，输出：

- 原视觉 patch token，供感知压缩和 Per Bank 使用；不是语言融合后序列，也不是新池化向量。
- 最后 EOS 位置的认知 token，供 Cog Bank 使用。
- 每个相机真实 patch 网格。默认只有 head_camera；显式选择多视角时逐相机压缩，不能把不相邻相机伪装成单个正方形网格。

主干全量训练时，当前视觉/语言/projector 均保留梯度。可选冻结模式才关闭主干梯度；梯度检查点使用 non-reentrant LLM API。每次 query 都显式禁用 KV cache，不存在第二条隐式跨 query 时间记忆。

`--model-path` 必须明确指定原 VLM run、原 VLA checkpoint 或 audited manifest，不会默认下载或加载 Qwen。只恢复 `projector`、`llm_backbone`、可选 `vision_backbone`；不恢复原 checkpoint 的动作头、Bank 或动作统计。压缩器、Bank 和 RMBench 14D 动作头从头训练。

本地可选 audited manifest 指向 LIBERO-100 微调权重；使用它时须记录这一初始化来源差异。支持该加载路径不等于已经真实加载/训练通过，也不等价于原论文从 OpenVLA 预训练权重初始化。

## 2. 双 Bank 不改变方法

Per Bank 与 Cog Bank 独立维护：

1. 当前 token 作 Query，旧历史作 Key/Value，时间编码仅加到 Key。
2. 两层单头 SDPA、残差 LayerNorm、FFN 和门控融合。
3. 先读旧历史，再写当前；默认 `update_fused=False`，写原始当前特征。
4. 历史 `detach().clone()`，不做跨历史帧 BPTT。
5. 容量溢出时，在相邻条目中选最相似的一对，按 token 对位平均并保留较早时间。

训练默认 group16，每组开始清 Bank；从同一个 episode 所有有效 query 时刻随机无放回选 16 帧并排序，短 episode 重复最后有效帧。不再以 `execute_prefix=4` 固定取 0/4/8…，也不默认按任务均衡抽组。

推理回合 `reset()` 清空两套 Bank、query 计数和主干状态；模型 PE 使用查询计数 0/1/2…，原生 runtime 的真实 `take_action_cnt` 仍用于观察单调性和重复请求检查。两种时钟有意独立。

默认容量与组长都为 16，因此一组训练内最后一帧至多读取 15 个历史条目，不训练超容量压缩后的长期状态。长 rollout 才会覆盖容量合并；没有据此新增训练方法。

## 3. 原版 DiT 与损失

默认原 `ActionModel` / `DiT-L`，认知条件与扩散时间进入前缀，感知通过各层 cross-attention 注入。

- 训练：每个条件与动作块 repeat4，再调用原 `ActionModel.loss`，噪声 MSE 对全部动作位置取平均。不使用 padding valid mask 排除尾部；native absolute-action 尾部必须是有限的末目标重复，NaN 会报错。
- 推理：DDIM 10、CFG 1.5，预测 `[B,16,14]`，执行前 4 个动作后再观察。
- 原 CFG 的“无条件”支路仍有感知条件，这是上游已有设计，不在本阶段顺手改成全无条件。

`mask_padding_loss=True` / repeat1 仍能显式选择以读取 legacy 权重或做消融，不是主默认。CPU 测试显式使用 DiT-S，避免测试分配 DiT-L；真实训练默认仍为 L。

OFT 与 Qwen 类保留以便旧 checkpoint 读取，不属于本阶段方法，不作为默认 fallback。

## 4. RMBench 原生数据契约

- HDF5 `joint_action/vector`：左臂 6 + 左夹爪 + 右臂 6 + 右夹爪，共 14D 绝对关节驱动目标。不是末端 delta、不是测量 qpos。
- 夹爪索引 6 与 13，保持连续值；部署仅限幅到 `[0,1]`，不二值化。
- 当前原生记录使用 `obs[t] -> vector[t+1:t+1+H]`，末尾重复最后目标，最后一帧无未来标签，不作为 query。`valid_mask` 标记哪些位置是真实标签，但默认不参与原版扩散目标。
- **标签语义限制：** 旧 HDF5 的“下一条记录目标”并非已经独立证明的“下一次实际发出命令”，不凭 loss 下降宣称闭环对齐已验收。其他转换数据不能无条件照搬 offset1。
- 使用 RMBench 自带 RGB codec 区分旧未标记 buffer 和 XPL-RGB1，不盲用 PIL 解码历史 JPEG。
- episode 级固定拆分，归一化只统计训练 episode 的真实 `vector[1:]`，不按重复采样后的标签计数，不泄漏验证数据。
- q01/q99 与全通道归一化；原版也有 q01/q99。训练集统计卫生与双夹爪语义保留，不属于新模型方法。
- 默认 train-only 增强：面积 0.9、归一化 ratio 1 的 crop；brightness ±0.2、contrast/saturation `[0.8,1.2]`、hue ±0.05，顺序与上游一致。CPU NumPy/Pillow 实现不宣称与 TensorFlow/dlimp 逐位等价；validation 不增强。

相机顺序、尺寸、codec 哈希、目标偏移、采样、增强及 episode 清单进入 manifest/checkpoint，不允许在评测端静默改变。

## 5. checkpoint 与训练编排

新 checkpoint schema 为 `choice-memory-rmbench-v2`，明确记录 `history_sampling`、`image_aug`、`repeated_diffusion_steps`、`mask_padding_loss`、`inference_timestep_mode` 五个方法字段，缺失则拒绝加载。

旧 v1 自动补入其旧语义：sequential、无增强、repeat1、masked loss、action-step PE。模型结构、相机、源资产哈希、归一化仍需匹配，不是把旧权重转成原版方法。

完整恢复状态包括 AdamW、RNG、采样步数和训练配置；推理 snapshot 没有这些状态，不能用作续训。恢复不得改变已冻结的数据、方法、世界大小或优化配置。

训练采用可恢复 DDP，不是原 FSDP。默认保持原配方统一学习率 2e-5、weight decay 0、constant schedule、gradient clip 1；有效 batch、卡数和步数是显式资源/预算选择，不把旧 Qwen global64/128/448 与 30k/55k 当作新阶段强制规则。

## 6. 运行与验收

服务通过共享 XPolicyLab WebSocket 代码对接，不创建共享 policy 包，不修改模拟器的动作、success、expert 或终止规则。只有 `get_action` 推理会更新 Bank，`update_obs` 不写历史。原生 runtime 消费观察一次，拒绝无 reset 跨回合与过期/重复 query。

CPU 测试验证 Bank 原源码数值对照、repeat4/RNG、feature path 梯度、原资产 VLM-only 加载、checkpoint 新旧语义、RGB/数据与模拟接口契约。网络 smoke 是可选项；本轮未据这些 fixture 宣称真实大模型、GPU 多卡或 SAPIEN rollout 已通过。也不宣称得到本阶段 benchmark 分数或记忆的因果效用。
