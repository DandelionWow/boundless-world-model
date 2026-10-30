# RoboTwin BWM 训练说明

本文记录当前服务器上 BWM 的八卡训练命令、参数含义、数据路径、断点恢复方法和已知限制。命令对应当前仓库：

```text
/data1/liuwenhao/Projects/boundless-world-model
```

## 当前可用性

- emb 模式已经完成修改后的八卡完整单步训练、compact ZeRO-2 状态保存和真实恢复测试；从 step 1 恢复后正常继续到 step 2，可以用于正式训练。
- raw 模式的数据和 VAE 编码链路已经完成八卡单步测试，但尚未进行长程训练验证，并且必须使用 `DATASET_NUM_WORKERS=0`。当前 `workers=2 + prefetch_factor=16` 会消耗过多主机内存。
- 训练集包含 75,133 个窗口、2,000 个 Episode、50 个任务。
- 八卡分布式采样会将每轮的 75,133 条补齐为 75,136 条，每张卡处理 9,392 个 micro-batch。补齐产生的 3 条重复是 Accelerate `even_batches=True` 的正常行为。
- 当前配置的每卡 micro-batch 为 1，梯度累积为 16，八卡的有效 batch size 为 `1 × 16 × 8 = 128` 个窗口/optimizer step。`training.batch_size` 或启动变量 `BATCH_SIZE` 现在会真正控制每卡 micro-batch。
- 一个数据 epoch 对应 `9392 / 16 = 587` 个 optimizer step。

## 路径

| 内容 | 路径 |
|---|---|
| Python | `/data1/liuwenhao/.conda/envs/bwm/bin/python` |
| Wan2.2 模型 | `/data1/liuwenhao/Projects/boundless-world-model/models/Wan2.2-TI2V-5B` |
| BWM 公开初始化权重 | `/data1/liuwenhao/Projects/boundless-world-model/ckpt/BLM/step-12000.safetensors` |
| 数据根目录 | `/data1/liuwenhao/Datasets/BWM_RoboTwin2.0_640_480_Wan_latents` |
| emb 元数据 | `meta/episodes_train.jsonl` |
| raw/emb 共用元数据 | `meta/episodes_train_with_video.jsonl` |
| action 统计 | `meta/stat.json` |
| 八卡 DeepSpeed 配置 | `configs/train/accelerate_deepspeed_8gpu.yaml` |
| emb 训练配置 | `configs/train/train_wan22_ti2v_5b_action_adaln_latent.yaml` |
| raw 训练配置 | `configs/train/train_wan22_ti2v_5b_action_adaln.yaml` |

## 正式启动 emb 训练

每个正式实验都应使用一个新的 `OUTPUT_PATH`。下面从公开 BWM step-12000 权重开始一个新的 shuffled finetuning：

```bash
cd /data1/liuwenhao/Projects/boundless-world-model

CONFIG_PATH=configs/train/train_wan22_ti2v_5b_action_adaln_latent.yaml \
TRAIN_MODE=fresh \
DATASET_METADATA_PATH=/data1/liuwenhao/Datasets/BWM_RoboTwin2.0_640_480_Wan_latents/meta/episodes_train.jsonl \
DATASET_NUM_WORKERS=2 \
OUTPUT_PATH=/data1/liuwenhao/Projects/boundless-world-model/outputs/bwm_emb_shuffle_seed42_run1 \
MAX_TRAIN_STEPS=20000 \
SAVE_STEPS=2000 \
SAVE_AT_TRAIN_END=1 \
bash scripts/train_distributed_example.sh
```

启动后，日志中的关键配置应为：

```text
dataset.data_keys: ['latents', 'action']
vae_mode: emb
action_type: eef_abs
num_history_frames: 9
history_template_sampling: 1
init_ckpt_path: .../step-12000.safetensors
resume_from: None
Dataset size: 75133
```

## 启动 raw 训练（尚未长程验证）

raw 模式在线读取 81 帧视频并通过 VAE 编码。当前服务器上应将 worker 设为 0：

```bash
cd /data1/liuwenhao/Projects/boundless-world-model

CONFIG_PATH=configs/train/train_wan22_ti2v_5b_action_adaln.yaml \
TRAIN_MODE=fresh \
DATASET_METADATA_PATH=/data1/liuwenhao/Datasets/BWM_RoboTwin2.0_640_480_Wan_latents/meta/episodes_train_with_video.jsonl \
DATASET_NUM_WORKERS=0 \
OUTPUT_PATH=/data1/liuwenhao/Projects/boundless-world-model/outputs/bwm_raw_shuffle_seed42_run1 \
MAX_TRAIN_STEPS=20000 \
SAVE_STEPS=2000 \
SAVE_AT_TRAIN_END=1 \
bash scripts/train_distributed_example.sh
```

raw 八卡单步测试完成一个 optimizer step 约用 3 分 5 秒，其中包含 16 个 local micro-batch。单步测试不足以给出稳定吞吐量，该数字只能作为当前服务器上的粗略参考。

## 从完整训练状态恢复

`step-N.safetensors` 只包含可训练模型权重，适合初始化或推理；`states/latest` 包含模型、优化器、scheduler、随机数状态和训练位置，才用于精确续训。

恢复时不要传 `INIT_CKPT`/`--ckpt_path`。启动脚本现在使用 `TRAIN_MODE` 明确区分两种模式：`fresh` 只接受初始化权重，`resume` 只接受完整训练状态。恢复命令为：

```bash
cd /data1/liuwenhao/Projects/boundless-world-model

TRAIN_MODE=resume \
RESUME_FROM=/data1/liuwenhao/Projects/boundless-world-model/outputs/bwm_emb_shuffle_seed42_run1/states/latest \
OUTPUT_PATH=/data1/liuwenhao/Projects/boundless-world-model/outputs/bwm_emb_shuffle_seed42_run1 \
MAX_TRAIN_STEPS=20000 \
SAVE_STEPS=2000 \
bash scripts/train_distributed_example.sh
```

这里的 `MAX_TRAIN_STEPS` 是恢复后的目标总 step，不是额外再训练的 step 数。例如 checkpoint 已经到 step 6000，设置为 20000 会继续训练至 step 20000。

恢复必须保持以下内容与原实验一致：训练配置、数据元数据、seed、GPU 数、梯度累积和数据集长度。当前恢复逻辑会根据保存的 `epoch_id` 重建该 epoch 的随机排列，再跳过 `batch_in_epoch` 个 local batch。

每个新实验会在输出目录写入 `.bwm_run.json`。其中记录 run ID、训练身份指纹、数据/配置哈希、基础模型文件信息、关键源码哈希和依赖版本。正常恢复会同时检查：

- `states/step-N/COMPLETE` 存在；
- checkpoint 中的 run ID、配置指纹和 RNG 方案与 manifest 一致；
- 当前代码、数据、模型文件、依赖版本和关键训练参数与新实验启动时一致。

不一致时默认拒绝恢复。`--allow_unsafe_resume` 只用于人工审计过的旧 checkpoint 或明确接受配置变化的迁移，不应作为普通恢复选项。

## 主要启动参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `TRAIN_MODE` | `fresh` 或 `resume` | `fresh` 要求且只允许 `INIT_CKPT`；`resume` 要求且只允许 `RESUME_FROM`。Python 入口还会再次检查两者互斥。 |
| `PYTHON_BIN` | bwm 环境 Python | 指定实际启动训练的 Python 解释器。 |
| `CONFIG_PATH` | emb 或 raw YAML | 选择 VAE 输入模式和训练超参数。emb 直接读取 latent；raw 在线编码视频。 |
| `ACCELERATE_CONFIG` | `accelerate_deepspeed_8gpu.yaml` | 使用 8 个进程和 DeepSpeed ZeRO-2、bf16。 |
| `MODEL_DIR` | Wan2.2-TI2V-5B 目录 | Wan DiT 和 VAE 基础权重目录。 |
| `INIT_CKPT` / `--ckpt_path` | step-12000 | 模型权重初始化，不包含 optimizer state。仅用于启动新实验。 |
| `RESUME_FROM` / `--resume_from` | `states/latest` | 恢复完整训练状态。恢复时不应再传初始化 checkpoint。 |
| `DATASET_DIR` | latent 数据根目录 | 相对 latent 路径的解析根目录。 |
| `DATASET_METADATA_PATH` | JSONL | 每一行是一个训练窗口。shuffle 发生在这些行之间，不改变窗口内部帧顺序。 |
| `ACTION_STAT_PATH` | `meta/stat.json` | 14 维 eef action 归一化所用的训练集统计。 |
| `DATASET_NUM_WORKERS` | emb=2，raw=0 | 每个 rank 的 DataLoader worker 数。raw 使用 2 会因并行预取占用大量主机内存。 |
| `BATCH_SIZE` | 默认读取 YAML（当前为 1） | 可选的每卡 micro-batch 覆盖值。当前服务器已实测 2 可以完成八卡 optimizer step；此时有效 batch 为 `2 × 16 × 8 = 256`。正式实验是否改为 2 仍属于训练超参数选择。 |
| `OUTPUT_PATH` | 每个实验的新目录 | 写入指标、模型权重和完整状态。不要让新实验复用旧实验目录。 |
| `MAX_TRAIN_STEPS` | 20000 | optimizer step 总数；不是 micro-batch 数。 |
| `GRADIENT_ACCUMULATION_STEPS` | 16 | 每 16 个 local micro-batch 更新一次参数。 |
| `SAVE_STEPS` | 2000 | 每隔多少 optimizer step 保存模型权重和完整训练状态。 |
| `SAVE_AT_TRAIN_END` | 1 | 若结束 step 不是 `SAVE_STEPS` 的整数倍，结束时再保存一次。smoke test 可设为 0。 |
| `DETERMINISTIC` | 0 | 0 仍保持 sampler 的固定 seed；1 额外启用 PyTorch 确定性算法，主要用于严格恢复调试。 |
| `USE_WANDB` | 0 | 设为 1 时启用 W&B。 |
| `USE_SWANLAB` | 0 | 设为 1 时启用 SwanLab。 |
| `RUN_NAME` | 空 | 远程 tracker 的 run/project 名称；默认使用输出路径。 |

`TRAIN_MODE=fresh` 会拒绝非空的 `OUTPUT_PATH`，因此不会无意覆盖或混入旧实验。`TRAIN_MODE=resume` 则必须复用原输出目录及其 `.bwm_run.json`。

YAML 中的关键参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `training.seed` | 42 | Python、NumPy、PyTorch 和 seedable sampler 的种子。epoch `e` 使用由 seed 和 epoch 派生的排列。 |
| `training.learning_rate` | `5e-5` | AdamW 的基础学习率。当前八卡 Accelerate 1.15.0 会让被包装的 scheduler 在一次 optimizer step 中推进 8 次，所以只有第 1 次参数更新使用约 `1.6667e-5`；此后为 `5e-5`。本次按要求保留原状。 |
| `training.weight_decay` | `0.01` | AdamW 权重衰减。 |
| `training.batch_size` | `1` | 每卡 micro-batch，现已传入 DataLoader 并由训练管线保留独立 batch/view 维。有效 batch 为 `batch_size × gradient_accumulation_steps × GPU 数`。 |
| `training.find_unused_parameters` | `true` | 允许 DDP 中存在未参与 forward 的参数。AdaLN 模式下 action encoder 的未用 `action_embedding` 现已显式冻结。 |
| `gradient.max_grad_norm` | `0.5` | 梯度裁剪阈值。日志里的 `grad_norm` 是裁剪前返回的范数。 |
| `gradient.use_gradient_checkpointing` | 启动脚本强制开启 | 用额外计算换取更低显存；`train_distributed_example.sh` 会传入 `--use_gradient_checkpointing`，覆盖 YAML 中的 `false`。 |
| `video.num_frames` | 81 | 每个训练样本的总像素帧数。 |
| `video.num_history_frames` | 9 | 条件历史帧数。当前模板为首帧 + 最近 8 个历史帧 + 72 个未来帧。 |
| `video.history_template_sampling` | 1 | 启用上述历史帧模板。 |
| `action.action_type` | `eef_abs` | 使用双臂末端绝对位姿，16 维四元数表示转换为 14 维 Euler 表示。 |
| `action.action_dim` | 14 | action encoder 输入维度。 |
| `training.trainable` | `dit, action_encoder` | 训练 DiT 和 action encoder。 |
| `training.mixed_precision` | bf16 | 训练精度。 |

## 输出与磁盘占用

输出目录包含：

```text
train_metrics.jsonl
step-2000.safetensors
step-4000.safetensors
...
states/latest -> step-N
states/latest.json
```

其中完整状态实际采用下面的版本化目录：

```text
states/step-2000/
states/step-4000/
states/latest -> step-4000
states/latest.json
```

新状态先写入隐藏的 `.step-N.incomplete`，写完 `COMPLETE` 标记后再用同一文件系统内的原子重命名发布，最后原子切换 `states/latest` 符号链接。保存失败时，旧的 `latest` 仍然有效；成功后保留最近两个完整状态，便于最新状态损坏时回退一版。

DeepSpeed 完整状态只保存可训练 module 参数；冻结的 Wan/VAE 参数在恢复时由 manifest 记录的基础模型文件重新加载。恢复会允许这些冻结参数不出现在 state 中，但会严格检查每一个可训练参数都存在，避免 compact checkpoint 掩盖损坏。

当前 `step-N.safetensors` 约 9.4 GiB，一个 compact 完整训练状态约 66 GiB。按 20,000 step、每 2,000 step 保存一次，最终约为 `10 × 9.4 + 2 × 66 ≈ 226 GiB`。保存新状态、尚未清理最旧状态的短暂峰值约为 `10 × 9.4 + 3 × 66 ≈ 292 GiB`，正式训练应按峰值预留空间。

## 当前代码限制

1. `prefetch_factor=16` 是代码中的固定值。它对 emb 可运行，但 raw 配合 2 workers 时会造成很高的主机内存占用，因此 raw 使用 0 workers。
2. `DETERMINISTIC=0` 时样本排列可以复现，但不能保证所有 CUDA kernel 的结果逐 bit 一致。
3. `ConstantLR` 没有显式传入 `factor=1.0`。在当前八卡 Accelerate 1.15.0 环境中，第 1 次参数更新使用约 `1.6667e-5`，之后为 `5e-5`；本次按要求未修改。
4. emb 与 raw 使用同一组像素帧和 action 索引，但两者的 latent 不保证数值等价：emb 读取整 Episode 一次编码得到的 latent，raw 则在训练时重新编码当前 81 帧窗口。两种模式的比较同时包含 VAE 编码上下文差异，并不只是“预计算与在线计算”的性能比较。
5. 当前训练入口只支持已经验证的 compact BWM profile：`task=sft`、`trainable=[dit, action_encoder]`、`raw/emb + AdaLN + no-text + no-image`。部分模块训练、LoRA、preset LoRA、FP8、模型 offload、额外模型源、旧版 text/noise/cross 配置和 `extra_inputs` 都会在加载大模型之前明确报错。这个限制保证 compact state 中包含所有相对基础 Wan 权重发生变化且会参与当前 BWM forward 的参数，避免恢复后静默退回基础权重或随机初始化权重。

已修复的项目：`batch_size` 现在真正生效；日志 loss 是本 optimizer step 内所有 rank、所有累计 micro-batch 的按样本加权全局均值；恢复时指标文件会原子截断到 checkpoint step 并去重；初始化和恢复有 Shell 与 Python 两层互斥保护；fresh 模式拒绝复用非空输出目录；完整状态采用版本化原子发布并保留上一版；不同 rank 使用 `seed + process_index` 的训练随机流；loss/grad norm 遇到 NaN/Inf 时所有 rank 同步 fail-fast；AdaLN action encoder 只训练私有 DiffSynth fork 使用的 `action_mlp1/2`，未用的 `action_embedding` 被冻结。

依赖方面，`requirements.txt` 和 `pyproject.toml` 已固定直接训练依赖；`requirements-lock-cu128-py310.txt` 是当前 Python 3.10 + CUDA 12.8 环境的完整版本快照。旧的 `scripts/train.sh` 和 `scripts/train_distributed.sh` 已改为退出并提示迁移，避免继续混淆模型初始化与完整状态恢复。

## 已完成的验证

- Python 静态编译、Shell 语法和 `git diff --check` 均通过。
- `tests/test_training_safety.py` 的 8 项回归测试通过：compact BWM profile 边界、batch/view 维度、extra-input 时间轴索引、NaN/Inf fail-fast、指标回滚去重、compact state 的可训练参数完整性、checkpoint 身份检查、fresh/resume manifest 边界。
- 75,133 行 emb 元数据与 raw/emb 元数据逐行一致，后者只额外增加 `video` 字段。
- 2,000 个 latent、2,000 个 action parquet 和 2,000 个视频路径均存在且非空；窗口边界合法，14 维 action 统计均为有限值。
- 此前已经逐个加载全部 2,000 个 latent，dtype、有限性、shape 和 Episode 长度均通过检查；视频帧数与 action 行数也已按 Episode 全量核对。
- sampler 已验证同 seed/epoch 可复现、不同 epoch 排列不同、八卡分片正确，且恢复后的索引 suffix 与原顺序一致。
- `batch_size=2` 已用真实 emb 数据验证：latent 为 `(2,1,48,21,30,40)`，action 为 `(2,81,14)`。
- 全局 loss reduce 已用两个独立进程验证：local `(sum=2,count=1)` 与 `(sum=10,count=2)` 在两个 rank 上都得到全局均值 4.0。
- checkpoint 发布已验证连续保存只保留最近两个完整版本，`latest` 原子切到最新版本；模拟新状态写盘失败时，旧 `latest` 保持可恢复。
- fresh/resume 两种启动命令的参数展开以及两层互斥报错均已验证。
- 本轮修改后的 emb 八卡完整 smoke 已通过：`batch_size=1` 在 16 个 local micro-batch 后完成 step 1，全局 loss `0.1904779`、grad norm `0.8419551`；`batch_size=2` 同样完成 step 1，全局 loss `0.1924345`、grad norm `0.8570692`。两组学习率均为 `5e-5`、退出码均为 0。
- 两组 smoke 是在每卡已有约 25.3 GiB 其他任务显存占用的情况下叠加运行；运行中观测到每卡总占用约 51.6 GiB，仍剩约 33.4 GiB，没有 OOM 或 rank 提前退出。
- 八卡 RNG probe 已验证：模型初始化摘要八卡一致，训练阶段的 noise、Python 和 NumPy 随机流八卡均不同，timestep 不再八卡相同。
- 最新真实 ZeRO-2 验证已完成：step 1 全局 loss `0.2648088932`，写出可读取的 833-key safetensors 和带 `COMPLETE` 的 compact 完整状态；恢复时进度条为 `9376 = 9392 - 16`，随后正常得到 step 2，`batch_in_epoch` 从 16 连续到 32。
- 另跑了一次不中断的两步对照。恢复与连续训练的 step 1 loss 完全一致；step 2 loss 只差 `8.34e-7`，grad norm 最大差 `9.32e-5`，符合 `DETERMINISTIC=0` 下非逐 bit 确定的预期。
- raw 已通过八卡单步测试。所有本轮临时 checkpoint 和对照输出均已清理，不属于正式训练数据。
