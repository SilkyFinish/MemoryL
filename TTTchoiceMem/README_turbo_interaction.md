# 第二阶段：VL-only 替换的接入状态

日期：2026-10-17。开发位置：`/mnt/data/dylan/d2/memory_l/TTTchoiceMem`，按用户要求直接修改 `main`，不自动 commit/push。

## 当前状态：CPU验收已完成，真实训练/rollout待验收

`--backbone turbo_interaction` 现在由 `ChoiceMemoryPolicy` 延迟构造 `TurboInteractionBackbone`，不再是 `NotImplementedError` 占位，也不会静默退回原 MemoryVLA/Llama。仅移植窄依赖 text/interaction 模块；保留来源和许可于 `choice_mem/turbo_modules.py` 与 `TURBO_MODULES_LICENSE`。原参考树只读。

已完成适配器定向测试、一次完整 CPU suite、原视觉真实权重加载、生产尺寸随机冻结 Turbo + 原 DiT-L 的 CPU 前后向及完整 checkpoint 恢复。独立审查又发现并修复了一处 Turbo resume 协议字段不一致，新增两模式回归及修复后的完整 suite 均已通过；最新结果为521 passed、1 skipped、130 subtests passed。正式BERT和完整正式预训练适配器的生产尺寸CPU前后向均已通过；适配器使用测试特征损失，不是真实RMBench扩散训练。**第二阶段RMBench训练、真实RPC与原生回合尚未执行**。不能宣称已跑通第二阶段闭环；第一阶段已有闭环记录不变。

执行状态：独立生产BERT验证曾被auto的 `Stage 2 classifier error - blocking based on stage 1 assessment` 内部错误阻断，当时未重试或绕过。用户随后明确选择“先做CPU，GPU等待”，允许按原路径重试；同一独立BERT脚本现已成功完成，正式适配器验证随后也已实际通过。之后用户明确授权共享GPU4的固定3步正式训练与恢复测试，沿用35GiB allocator上限、失败即停。实际执行脚本时因启动空余不足38GiB而停止，未启动torchrun或GPU模型，不是OOM；退出后复查空余约31.2GiB。没有自动重试、换卡、降配或停止其他任务。见 [GPU4启动检查记录](runs/turbo_pretrained_single_gpu4_3step_20261017/REPORT.md)；历史auto拒绝记录保留，未绕过执行。

## 已完成的验证及其边界

- **移植前外围完整回归**：417 passed、1 skipped、130 subtests passed，pytest 耗时350.82秒，exit0。39个登记的开发 Python 文件当时重哈希一致。见 [历史外围报告](runs/turbo_plumbing_20261017/REPORT.md)。这是核心接入前的历史结果，不代表最新源码完整回归已通过。
- **窄核心模块测试**：代理报告 Torch2.6 / Transformers5.19 下35 passed（80.26秒），Torch2.4.1 / Transformers4.40.1 下同组35 passed（11.98秒）。包括真实 tiny 本地 BERT、mask/分段、混长 batch、逐样本一致性、padding 隔离、有限梯度与 FP32 interaction；不是生产 BERT-base/完整视觉适配器验收。HF5 测试仅在依赖导入边界窄 mock 了宿主机的 accelerator probe；生产代码没有该全局补丁。HF4 使用现有环境并只读借用已有 pytest，没有安装依赖。
- **初始化 mapper 测试**：53 passed，CPU/Torch2.6；覆盖严格 key/shape/dtype/finite、无部分写入、危险格式拒绝和普通权重归档检查。见 [JUnit](runs/turbo_vl_core_20261017/gd_mapper_tests_torch26.xml)。这些是合成权重测试，不是完整策略验收。
- **真实 GroundingDINO subset 初始化**：Torch2.4.1、CPU，本地940个 keys 中严格加载182个（fusion108、text72、text projection2），加载后全部逐位等于来源且选中张量均 finite。有意忽略758个 BERT/视觉/检测等 keys。见 [实际 strict-load 报告](runs/turbo_vl_core_20261017/gd_core_initialization_strict_cpu.json) 和 [来源元数据审计](runs/turbo_vl_core_20261017/gd_source_audit.json)。该结果证明当前核心 state 兼容性，不证明完整 Turbo 策略或数值行为等价。

- **集成后定向回归**：163 passed、1 skipped、13 subtests passed，73.95秒；登记119个源码文件，测试期间无变化。见 `runs/turbo_vl_core_20261017/targeted1.*`。
- **集成后完整回归**：第一次519 passed、1 skipped、130 subtests passed，308.71秒；新增 resume 修复与两项回归后，最新 `full2` 为 **521 passed、1 skipped、130 subtests passed**，pytest537.28秒、runner544.88秒。两次均核对119个源码文件，测试期间无变化。最新证据为 `runs/turbo_vl_core_20261017/full2.log`、`full2.status.json`、`full2.source.json` 和 `full2.junit.xml`。
- **真实原视觉 factory**：正式加载两塔，685个 target tensors 全部 finite，逐位等于 timm filter 后来源；SigLIP 实际走 OpenCLIP 映射、archive guard 和 weights_only。CPU 输出 `[1,256,2176]`，两塔16×16 grids；架构随机初始化与原 factory 的685项 state、transform、pixels、raw features 全部精确一致。见 [视觉报告](runs/turbo_vl_core_20261017/original_vision_cpu_probe_report.json)。
- **生产尺寸 debug 适配器与完整策略**：真实855,822,784参数 backbone；冻结/eval、BF16/FP32分区、RNG隔离、raw Per精确透传、图像/指令敏感性、无缓存及1068项实际 backbone state 恢复通过。完整1,691,342,574参数策略（下游835,519,790参数）实际执行两 query 合成标签的 diffusion loss/backward，有限 loss 为1.0071064；首步压缩器/双Bank梯度为0、动作头非零，符合原 DiT 零初始化，不据此声称 Bank 已学习。内存内 runtime 两次查询各输出有限 `[4,14]` 动作。见 [适配器报告](runs/turbo_vl_core_20261017/adapter_production_debug_cpu.json)。这不是RMBench数据训练、RPC或原生回合。
- **完整 debug checkpoint**：实际保存5,082,844,302字节全策略 checkpoint，1492项 state 全部逐位恢复一致；恢复只读取本次 checkpoint，不重新读取初始化权重。同种子真实 DDIM10/CFG1.5 动作 `[1,16,14]` 完全一致。见 [完整恢复报告](runs/turbo_vl_core_20261017/full_debug_checkpoint_cpu_probe.json)。该检查使用未训练 step0，不是 optimizer/scheduler 实际恢复验收。
- **resume 协议缺陷回归**：新增正式/随机冻结两模式的实际 save/read/validate-resume 测试，修复前两项均按预期报 protocol mismatch；修复后两项均通过，合成模型的 AdamW 恢复后下一步权重与不中断路径逐位一致。训练器现在提前写入 Turbo-only 顶层 `backbone`，与保存器一致，不改变原 MemoryVLA/Qwen 协议形状。

- **正式生产BERT独立验证**：真实BERT-base（12层/768宽/12heads）本地safetensors/eager/no-pooler/no-cache加载、来源embedding样本精确匹配、raw instruction与分段mask、混长batch/逐样本一致性、padding扰动隔离及全部199项参数张量finite backward通过，39.28秒。batch token最大误差8.5235e-6（容差2e-5），padding扰动后有效token误差为0。随机/config-only禁止权重parser仍构造成功，测试前后登记源码hashes不变。见 [正式BERT报告](runs/turbo_vl_core_20261017/bert_production_cpu_retry1.json)。这是实际预训练BERT结果，不是从debug/tiny路径外推。

- **完整正式预训练适配器**：真实855,822,784参数、不冻结，原视觉+BERT+GD subset一起加载，raw Per精确透传、图像/指令敏感性、无缓存、RNG隔离和六个子树finite/非零梯度均通过。测试特征损失0.0386105292，不是正式扩散训练。probe81.48秒、runner98.11秒，exit0、无timeout，119项源码前后哈希不变。见 [正式适配器报告](runs/turbo_vl_core_20261017/adapter_production_formal_cpu.json) 及 `formal_adapter_cpu1.*`。

以上不同环境和范围的测试不能简单相加。当前产物见 `runs/turbo_vl_core_20261017`。正式适配器通过不等于完整DiT-L策略的真实数据训练、optimizer恢复或GPU显存已验收。

## 最小替换边界

- 保留原 DINOv2 + SigLIP 视觉塔、原图像变换、按通道拼接的2176宽 raw perception tokens、压缩器及 Per Bank。Per 不经过新 interaction。
- 认知路径改为 `C_v→256` 视觉投影、原始指令的 BERT-base-uncased、`768→256` 文本投影及六层 Turbo VL interaction。
- 对交互后的有效指令 tokens 做 masked mean，排除 padding、tokenizer special tokens 及分段标点，再投影 `256→4096`，输出 `[B,1,4096]` 给 Cog Bank。不是 CLS，也不声称与原 Llama EOS 等价。
- Per/Cog 双 Bank、DiT-L、扩散目标、14D 动作与 RMBench 数据协议不变。没有 DINOv3、Turbo ACT、TTT、OFT 或新记忆选择器。
- 不从第一阶段 checkpoint 续训 Turbo；新表征必须建立新 run、下游初始化和恢复状态。
- 构造只调用原视觉 factory，不构造 Llama/完整 PrismaticVLM；原包的 materialize 模块仍会导入其类定义，不能声称完全没有语言类导入。

## 固定数值与 mask profile

固定 `manual / normalized / key_padding_mask / FP32 interaction compute`，六层、hidden256、配置8 heads/实际4 heads、enhancer inner1024、fusion DropPath0.1、gamma初始1e-4。不存在新增的视觉 self-attention/FFN。

为避免上游 batch/padding 风险，明确做了以下适配，不声称逐位复现上游任意 batch：

- CLS/SEP 始终 self-only；period/question 分段并重置位置，不因 batch padding 宽度改变 SEP 行为。
- 有效 query 看不到 padding keys；padding query 保留安全 self row，防止全 mask 的 NaN。
- text attention 按 batch-major 正确展开各 head 的 mask；融合 attention 的稳定化 scalar 按样本/head 排除 padding。
- HF4 使用3D bool BERT mask；HF5 使用4D additive mask。两者都固定 eager、无缓存。
- BERT 不构造未使用的 pooler，不改变 last_hidden_state 的 token 计算。
- 最终融合层的visual-update输出不进入损失：Per使用原视觉，Cog只读fused text。真实正式适配器反向确认interaction最终visual-update有5项unused参数；原两塔second-to-last patches提取另有43项末block/norm/pool参数未使用，属于保留原视觉的边界。因此全参DDP需要`--find-unused-parameters`（训练器现有默认值就是true）；明确禁用时提前报错，不偷偷给损失加零项、冻结额外分支或改变读出。

## 显式初始化与精度契约

默认 `backbone=memoryvla`、`backbone_initialization=pretrained` 不变，不增加旧配置必须填写的新字段。

- 原 `random-frozen-debug` 仍严格限定原 MemoryVLA，规则不因 Turbo 而放宽。
- 新 `turbo-random-frozen-debug` 仅适用于 `turbo_interaction`，要求显式冻结、BF16 配置及原 DiT-L；它是工程调试配方，不是预训练成绩。debug 只使用 BERT metadata，不读取初始化权重。
- 新路径固定原视觉架构标识、224×224、Per 压缩宽度256、DiT-L。`base_vlm_id=prism-dinosiglip-224px+7b` 在新路径只描述保留的视觉架构。
- 新 debug 中视觉/BERT/投影 BF16，interaction 冻结但始终 FP32；四个下游组件使用 FP32 可训练参数。不先将 interaction 舍入到 BF16 再转回 FP32。
- 正式初始化使用审计过的原 timm 视觉资产、独立本地 BERT，以及 GroundingDINO fusion/text/feat_map subset。新增视觉投影和认知投影随机。此处视觉初始化来源是 timm 权重，**不声称等于原 LIBERO 微调 VLA checkpoint 内的视觉权重**；正式效果对照需单独控制来源。
- 原视觉先按隔离的 CPU seed 构造，再构造新文本/交互模块；不让 BERT 消耗的 RNG 改变视觉随机初始化。已验证真实视觉 factory 的同 seed 随机 state/transform/raw features 逐位一致，以及适配器 RNG 隔离；没有重新分配原7B Llama进行比较。
- checkpoint 保存全部实际 state；恢复时架构构造不重新载入初始化权重，随后 strict 恢复。正式路径仍审计来源 hashes。跨架构、资产、recipe、精度或数据契约漂移应拒绝恢复。
- debug 服务必须显式 opt-in；正式 suite 拒绝把 debug 当作正式成绩。

## 数据资产清单格式

`choice_mem/turbo_assets.py` 仅使用标准库进行本地 JSON/文件哈希审计，不加载或执行候选权重。以下是 debug metadata 清单示意，路径和摘要必须替换为真实值；不是可运行训练命令。

```json
{
  "schema": "memoryvla-turbo-vl-assets-v1",
  "bert": {
    "path": "/absolute/path/to/bert-base-uncased",
    "files": {
      "config.json": "<sha256>",
      "tokenizer_config.json": "<sha256>",
      "vocab.txt": "<sha256>"
    }
  }
}
```

- 至少声明 BERT config、tokenizer config，以及 tokenizer.json/vocab.txt 之一；存在的受支持 tokenizer sidecar 也必须声明。禁止路径跳转、自定义 tokenizer 类与可执行 sidecar；检查标准 BERT-base 12层/768宽/30522词表。
- debug 不跟随视觉、interaction 或 BERT 权重路径，不要求其存在，也不 stat/hash 它们。
- `pretrained` 还要求 `bert.files` 声明单文件权重，以及 `vision` 恰好映射 `vit_large_patch14_reg4_dinov2.lvd142m` 和 `vit_so400m_patch14_siglip_224`；每个视觉条目与 `interaction` 条目均为 `{ "path": "/absolute/file", "sha256": "..." }`。
- BERT 优先 safetensors；可选 `.bin` 和原视觉 torch 权重需先经过普通权重归档检查。`torch.load` 必须 `weights_only=True`，不回退到 unrestricted pickle。Torch2.4 即使 weights_only 也可能分派 TorchScript，故另行拒绝含 code/constants 的归档；非 ZIP legacy 格式也明确不支持。
- 哈希匹配不等于格式或兼容性通过；GroundingDINO 不从形状猜测映射，也不导入检测扩展或完整 Turbo 包。
- 未下载模型、安装/升级共享依赖或读取密钥。

## 待完成

1. 共享GPU4短测已有明确授权，但实际启动检查因空余不足38GiB停止；退出后复查为31,943MiB。尚未执行真实训练，未自动重试、预约GPU或设置轮询。需要资源满足并决定重新尝试后才能继续，不能将此启动前停止当作模型OOM或整卡放不下的证明。
2. 在资源允许时执行独立run的RMBench短训练、step1恢复及完整optimizer/scheduler/RNG/sampler审计。正式全参训练须实测峰值，DDP不会分片参数或优化器状态；不更换小架构。真实RPC与固定原生回合属于后续另行确认的验收范围，本次未执行，不按策略结果筛seed。

本轮真实清单：debug 使用 `runs/turbo_vl_core_20261017/adapter_debug_assets.json`；正式候选使用 `runs/turbo_vl_core_20261017/turbo_formal_assets.json`。后者已独立通过全部9项 manifest/metadata/weight hashes 的 torch-free 审计（`formal_asset_manifest_audit.json`），哈希审计与上述实际正式BERT/适配器数值验收是不同证据，均不代替完整策略的真实训练验收。

仓库根现有 `.gitignore` 忽略 `TTTchoiceMem/tests/`。新增测试文件本地存在，但普通 `git add .` 不会包含它们；当前未改根 ignore、未 stage、未 commit/push。
