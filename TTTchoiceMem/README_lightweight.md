# TTTchoiceMem：Qwen、MemoryVLA记忆模块与RMBench原生适配

更新日期：2026-10-13。本文说明模型、数据和代码边界；实际训练/恢复/评测命令及环境路径以[中文全量训练操作手册](README_full_finetuning.md)为准。

## 1. 当前正式配置与文档入口

当前是**十任务联合训练一套Qwen + MemoryVLA双Bank + DiT-L模型**：4卡DDP、每卡16查询、累积1、全局64、55000次参数更新，全部1,287,535,982参数参与训练。完成后由tmux中的管理进程自动校验最终权重并运行500回合原生评测。

公司集群另准备了**单节点8卡A800、每卡16、累积1、全局128、OFT、55k**方案，约8.83亿参数全量训练，独立输出不覆盖本机任务。新提交默认主相机单图，`--cameras`可选1～3路，`--action-head`可切换OFT/diffusion，`--workers-per-gpu 1|2|3`可把评测上限设为8/16/24个回合。集群尚未启动，命令见操作手册第11～12节；不能因共享目录就认为A800内核、显存和Vulkan渲染已验收。

- [全量训练操作手册](README_full_finetuning.md)：当前任务列表、实际环境、tmux、从头启动、直接torchrun、恢复、评测、保存位置和磁盘占用。
- [项目中文总览](README.md)：本地改造入口，以及原MemoryVLA论文、模型和原版接口介绍。
- [中文实验日记](../document/experimental%20diary/2026-10-13_TTTchoiceMem_Qwen全量微调_DiTL_全局64_55k.md)：实际检查、启动、耗时及清理记录。

旧448/30k短试跑、512/DiT-S/OFT测速和小规模模型验收不是当前正式训练。11个非正式run目录与7个独立验收脚本已按用户要求删除，旧记录路径不可再访问；历史数字不能冒充本次最终结果。

## 2. 改了什么，没改什么

新增代码支持主干`memoryvla|qwen35`和动作头`diffusion|oft`组合。**本机运行中任务仍是Qwen全量微调和原版DiT-L；55k正式计划的4卡/global64及8卡/global128均可选两种动作头，动作头不绑定卡数，不把不同配置模型状态混用。** OFT不要求DiT型号、DDIM或CFG校验；扩散分支的检查仍保留。

本次没有实现TTT在线更新、记忆效用评分器、EventVLA关键帧选择或新增benchmark任务。工程名TTTchoiceMem不意味着这些方法已实现。

原`train.py`、`deploy.py`、`vla/`、`prismatic/`、`action_model/`源码保留；新增模型/数据/运行时适配在`choice_mem/`及独立入口中。按最新要求三个README改为中文，不改模型行为。原版7维动作入口和本地RMBench14维动作入口不可混用，旧7维checkpoint会被新加载器拒绝。

## 3. Qwen主干与可选相机特征

- 官方模型`Qwen/Qwen3.5-0.8B`，固定revision `2fc06364715b967f1860aea9cf38778875588b17`。
- 本地权重目录`/mnt/data/dylan/d2/memory/TTTchoiceMem/assets/Qwen3.5-0.8B`。
- 只加载Qwen自带的图像编码器、文字模型、官方处理器；不夹带原DINO、SigLIP或Llama。
- `--cameras head_camera`为主视角单图；也可选主视角加一个腕相机，或三路head/left/right，按命令有序列表送入模型。当前运行的本机任务仍是三路。先双线性缩放224×224，再各主干自己的处理器；Qwen实际得到256×256 patch布局、每相机8×8 merged网格，一/二/三路约64/128/192视觉token，视觉与认知宽度1024。
- `FeatureBatch`返回当前视觉token、任务条件认知token，以及各相机独立的时间/高/宽网格；网格来自实际编码结果，不把三相机拼成假的正方形。
- 图像压缩按每个相机网格单独执行原版reduce/SE/expand结构；当前宽度1024→512→256，空间token数不变。
- 原主干适配的历史实测为三相机各16×16网格、共768视觉token，视觉宽度2176、认知宽度4096；不同宽度不能直接共享新压缩/Bank状态。
- 每次查询`use_cache=False`、不传跨查询KV状态，并清空Qwen的位置状态；历史只能通过显式两套Bank传递。不会生成动作文本或自回归动作token。

官方Qwen权重只初始化主干；图像压缩、两套Bank和原版DiT-L均为本次新初始化，需要学习。未导入原版7维动作头、旧LIBERO权重或EventVLA动作权重。

## 4. 原版per/cog记忆模块

- `per_mem_bank`保存每次查询的视觉特征；`cog_mem_bank`保存任务条件认知特征，各自独立按episode管理。
- 每套容量16条，读取2层，时间编码、残差和前馈层保留；当前融合为可学习门控。
- 写入当前原始特征而不是融合后特征，旧条目断开计算图；读取层及当前输入仍可训练，但不反向穿过整个历史轨迹。
- 超容量时寻找最相似的相邻条目并按对应位置求平均合并；算法保持原版规则。同一episode内网格变化会明确报错，不隐式拼接错位。
- `reset()`清空两套Bank、主干查询状态和待处理观察，防止回合间泄漏。
- 当前每训练组16查询、相隔4个记录帧，组前清空历史，因此最后查询最多读15条历史；**训练组不覆盖写入第17条之后的容量合并，长回合推理会触发**。

没有把Bank换成简单KV缓存，也没有为了速度删除历史读取。注意力记录仍是可选回调，正式训练不默认累积张量记录：

```python
policy.per_mem_bank.set_attention_recorder(callback)
policy.cog_mem_bank.set_attention_recorder(callback)
```

## 5. DiT-L训练与推理

当前复用原`action_model.ActionModel`及DiT-L（24层、1024宽度、16头），只在独立适配中指定原生14维输出、条件宽度和有效动作mask，不修改原版DiT源码。

### 训练

原扩散日程100个时间点。每个样本随机选一个时间点，直接向真实动作加入已知噪声，**DiT前向一次预测噪声**，计算有效动作位置的噪声MSE并反向。不是每个样本执行100次去噪，也不是训练时循环10步DDIM；mask掉的尾部位置不参与损失，包括NaN填充防护。

### 推理

从随机动作噪声出发，用DDIM迭代10步、CFG1.5，生成`[batch,16,14]`，执行前4动作后重新观察。CFG把有/无认知条件分支合并成双倍batch，仍是10个串行采样步骤；同次查询的Qwen和Bank条件只计算一次，不重跑10次主干。

原版还提供DiT-S（6层、384宽度、4头）和DiT-B；换Qwen不要求用L。当前L由用户最终确认，不是把S称作“非原版”。同条件S组合整体896,788,334参数，L组合1,287,535,982参数。

### OFT分支边界

保留的可选OFT实现是不同未来动作位置的可学习查询，分别交叉读取两套融合后Bank特征，拼接投影并叠加位置残差，再通过两块MLPResNet输出全部16×14动作，采用masked L1。它不循环去噪，也不是重复全局向量；**当前未用它训练**。此前小规模测速不能用来断言同等训练预算下谁成功率更高。

## 6. 原生RMBench动作与数据规则

原数据目录：`/mnt/data/dylan/d2/memory/RMBench/data`。正式清单：

```text
/mnt/data/dylan/d2/memory/TTTchoiceMem/protocols/rmbench10_h16_p4_g16_host64_20261013.json
```

- 读取原生`data/episode*.hdf5`中的`joint_action/vector`，顺序是左臂6关节、左夹爪、右臂6关节、右夹爪。
- 表示绝对关节**驱动目标**，不是末端delta动作，也不是实测qpos。夹爪位于索引6和13，保留连续0～1，不二值化、不交换左右、不使用旧7维索引规则。
- 当前`RGB[t]`对应下一条记录目标`vector[t+1:t+1+16]`，末尾重复最后真实目标并用`valid_mask`屏蔽；最后记录RGB没有下一条目标，不训练该帧。
- **对齐风险保留说明：** 旧HDF5缺乏逐帧采集器commit/命令发出时刻证明；“下一条记录驱动目标”不等于已经证明“下一次实际发出的命令”。不凭训练loss宣称对齐已验证，需要闭环评测。
- 模型只使用所选1～3路相机和任务级指令；没有隐藏模拟状态、未来阶段标签或未经验证的本体状态输入。相机选择保存在独立manifest及checkpoint中，不能在评测时把三路模型静默改成单路；原生模拟器仍保留benchmark观察/渲染配置。
- 使用RMBench原RGB解码器，区分旧未标记channel-reversed JPEG与XPL-RGB1标准JPEG，不用PIL盲解全部历史buffer。
- 每任务episode级固定SHA256划分45训练/5验证；共450/50。训练集真实`vector[1:]`目标每行统计一次，总296455行，使用训练集q01/q99归一化；常量通道正确逆变换，不泄漏验证集。
- 完整组4436、不同查询70976；440个不完整尾组、3314查询排除并记录，不能为了拼batch跨episode。
- 清单冻结数据、指令、codec和episode哈希，内部SHA256 `3cbbe8ee38ee5c437bf9cb17162532eb299e7317cefedb0cc8873faf1db7a983`，文件SHA256 `6192b51c14557d631a555e5fa011b4ccd948b2594cc425e175e5a903f0404b98`。

## 7. checkpoint与原生服务接口

完整checkpoint保存模型、配置、来源revision/资产哈希、处理器、网格/维度、动作语义、归一化、训练清单、步数、AdamW与恢复状态。推理快照不保存AdamW/RNG/调度器；两种文件不能互换作续训。

加载按checkpoint恢复配置，而不是套用CLI默认值；主干/动作头/相机/标签语义断言不一致、来源文件变更、旧格式或维度不匹配均拒绝。只加载可信的本代码产物，使用`torch.load(weights_only=True)`。

- `serve_rmbench.py`在Qwen独立环境中启动原生XPolicyLab WebSocket模型服务，输出4×14执行动作。
- `eval_multitask.py`在Qwen环境中管理十任务；其模拟器子进程改用RMBench独立环境。
- `eval_rmbench.py`加载原生评测入口，只增加真实`take_action_cnt`字段；该文件也是正式评测依赖，不仅是小样本测试。
- 原生policy名字是结果标签，无需创建或修改共享`XPolicyLab/policy/TTTchoiceMem`目录。
- 模拟器每进程只暴露对应物理GPU为逻辑cuda:0，遵循原SAPIEN渲染约定；不能把未映射的物理cuda:7直接当renderer编号。
- 原生expert、动作、成功、步数上限和终止规则保留；正式评测十任务各50回合，成功失败均计数，回合开始清空历史。

本机训练和模拟器的两套venv、启动和评测命令均已在[操作手册第2～7节](README_full_finetuning.md)写明。不要套用原版LIBERO/EE动作命令或重新安装旧依赖来启动当前原生任务。

## 8. 保留代码与已清理内容

生产链路保留原6个入口，并新增`configure_training_job.py`生成不可覆盖的集群配置变体；多worker评测调度由独立适配模块支持。不是恢复已删除的测速/独立验收工具。

7个独立准备工具已删除：迁移审计、十任务资产检查、模型/OFT冒烟checkpoint检查、Qwen内核数值检查、2+2恢复验收。最终checkpoint校验仍在生产入口和`choice_mem/checkpoint.py`，已安装causal-conv1d/FLA未卸载。

11个非正式run目录已删除，主要是两套DiT-S完整验收状态和一套DiT-L状态，共约71.4GiB。原因是每份都保存全模型与两个AdamW moment，和试跑步数少没有比例关系；详见操作手册第9节。当前训练目录和管理目录保留，后续正式评测目录自动创建。

## 9. 测试与未完成项

清理后现有CPU回归：**98 passed、1个默认可选网络测试skipped、118 subtests passed**；真实网络RPC此前另行验证。测试覆盖Bank等价/隔离、压缩网格、动作slot、mask、数据划分/对齐、checkpoint、原生接口和训练/评测门控。

CPU测试命令，不使用训练GPU：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-qwen35-host512/bin/python -m pytest /mnt/data/dylan/d2/memory/TTTchoiceMem/tests -q
```

准备阶段已检查真实主干形状/梯度、优化内核前向/梯度和实际调用、十原生任务留出reset/render/action，以及真实全局64/DiT-L两步训练加两步恢复。对应旧原始文件已按用户要求删除，不能再声称可从这些文件独立复算。

当前训练启动不等于55k/500回合已完成；训练loss下降不等于闭环成功。尚无完整结果时，不宣称benchmark提升或历史记忆因果效用，也不将旧合成标签/三步OFT冒烟结果与当前DiT-L正式结果混合。
