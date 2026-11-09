# ELF3 DWAQ：盲走与楼梯地形强化学习

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.1-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-yellow.svg)](LICENSE)

本项目将 DWAQ（DreamWaQ 风格的历史观测编码器 + PPO）适配到 **ELF3 29 自由度人形机器人**，用于无视觉地形适应、盲走和楼梯训练。训练框架基于 Isaac Lab、TienKung-Lab、Legged Lab 与 RSL-RL。

- Actor 仅使用本体感觉：单帧 100 维、5 帧历史，输出 29 维动作。
- 保留 CENet 速度／隐变量估计、上肢镜像及软默认姿态辅助损失。
- 17 级混合地形课程，楼梯高度上限 30 cm。
- 支持 TensorBoard、SwanLab、TorchScript／ONNX 导出和 MuJoCo Sim2Sim。

> 本项目不使用 AMP 动作数据或 AMP 权重。源码仓库不包含训练权重、ONNX、私有训练日志或真机数据；地形课程上限也不代表已达到对应的实机通过率。

## 公开训练任务

| 任务 | 用途 | 训练入口 |
| --- | --- | --- |
| `elf3_dwaq_feedback_v1` | 20 Hz 关节速度反馈基线 | `scripts/train_elf3_feedback.py` |
| `elf3_dwaq_still_v2` | 恢复的原始零指令静止／腿腰姿态任务 | `scripts/train_elf3_still_v2_restored.py` |
| `elf3_dwaq_still_v2_ankle_cost_v1` | 零指令静止、默认腿腰姿态及踝部成本版本 | `scripts/train_elf3_ankle_cost.py` |
| `elf3_dwaq_still_v2_force500_v1（可用于后续微调）` | 当前500 N足力峰值、动作平滑及运动高度版本 | `scripts/train_elf3_force500.py` |
| `elf3_dwaq_still_v2_curriculum_v1（推荐）` | 先学still_v2，随后线性开启踝部、运动高度及平滑约束 | `scripts/train_elf3_reward_curriculum.py` |

查看任务入口：

```bash
./elf3.sh scripts/elf3_task_catalog.py
```

以上五个任务公开注册。其他配置和脚本保留为继承、严格校验及历史模型导出的内部依赖，不再作为公开训练入口。

## 79000轮权重训练路线（历史记录）

[Release：`elf3-force500-79000-v1`](https://github.com/JKYovo/DWAQ/releases/tag/elf3-force500-79000-v1)中的`model_79000.pt`来自以下分阶段训练链，**不是新课程任务从零训练的产物，也不是从零直接使用最终500 N奖励的产物**。表中轮次为检查点编号，每阶段保留前一阶段的改动。

| 阶段 | 轮次 | 主要改动 |
| --- | --- | --- |
| 原始`still_v2` | 0～18500 | 从零学习站立与行走；零指令关闭周期驱动，保留实际静止、腿腰默认姿态约束；足力权重`−0.003`、起罚阈值`500 N`。 |
| `ankle_cost` | 18501～38500 | 从18500恢复；增加四踝估算力矩平方成本`−0.0005`、非零指令下相位支撑期足速成本`−1.0`。 |
| 高度／足力强化 | 38501～57000 | 从38500恢复；`action_rate_l2: −0.01 → −0.012`；增加运动高度下限项`−50`；足力权重`−0.003 → −0.005`、起罚阈值`500 → 400 N`。 |
| 动作平滑＋峰值采样 | 57001～76500 | 从57000恢复；增加`action_smooth=−0.001`；`dof_acc_l2: −2.5e−7 → −3e−7`（增强20%）；足力从最后一个物理子步改为整个控制周期峰值。 |
| 固定500 N版 | 76501～79000 | 从76500恢复；**只将足力起罚阈值`400 → 500 N`**，其他配置保留，在79000保存发布权重。 |

共同规格为16384环境、seed 42、24 steps/env、4 mini-batches、5 PPO epochs，每500轮保存；沿用ELF3资产、隐式PD、`armature=0.01`、20 Hz关节速度观测模型及`500 → 29`接口，不启用动作延迟。阶段切换恢复网络、Adam、学习率及轮次，但不恢复物理状态、地形等级、随机数流、观测历史或反馈滤波缓存。

峰值采样指4个5 ms物理子步中，**同时双脚竖直接触力Fz范数的最大值**；500 N是该指标的起罚阈值，不是单脚硬限幅。已放弃的轻落脚、膝角阈值及弱高度试验分支不属于这份权重的继承链。

使用该权重resume／同配方微调时，选择`elf3_dwaq_still_v2_force500_v1`和`scripts/train_elf3_force500.py --resume /path/to/model_79000.pt`，从79001开始；**不要使用课程任务resume**。下载、SHA256及完整命令见[79000权重说明](https://github.com/JKYovo/DWAQ/blob/main/docs/releases/elf3_force500_79000_v1.md)。新课程会改变奖励开启时间及早期足力采样，不能据此视为对上述历史路线的等价复现。

## 环境与安装

已验证环境：Ubuntu 22.04、RTX 4090、Python 3.10.20、PyTorch 2.5.1 + CUDA 12.4、Isaac Sim 4.5.0、Isaac Lab 2.1.1、MuJoCo 3.3.2。

准备 Conda 和兼容的 NVIDIA 驱动后安装：

```bash
git clone https://github.com/JKYovo/DWAQ.git
cd DWAQ
bash scripts/setup_elf3_env.sh
```

默认 Python／Isaac Sim 位于 `.venv/`，Isaac Lab 位于 `IsaacLab/`。安装路径可覆盖：

```bash
ELF3_CONDA=/path/to/conda \
ELF3_ISAACLAB_DIR=/path/to/IsaacLab \
bash scripts/setup_elf3_env.sh
```

后续命令均在仓库根目录执行，并使用 `./elf3.sh` 加载项目环境和 CUDA 动态库。如使用独立安装的 Isaac Sim，可设置 `ELF3_ISAAC_SIM_DIR=/path/to/isaacsim`。

## 训练

### 从零训练

奖励课程版（独立从零，不要求历史权重或本机日志）：

```bash
./elf3.sh scripts/train_elf3_reward_curriculum.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 50001 --seed 42 \
  --ramp-start 20000 --ramp-end 30000 \
  --run-name curriculum_scratch_16384_seed42 \
  --swanlab-project DWAQ
```

先用still_v2目标训练到20000轮，随后10000轮线性开启后续质量约束，30000轮后保持最终配方，结束于`model_50000.pt`。课程全阶段均使用500 N足力阈值和控制周期峰值采样；资产、PD、20 Hz反馈、命令采样及PPO不变。课程进度写入检查点，续训必须匹配原时间表；原始still_v2父模型显式迁移及提前开启二阶段的方法见[课程说明](docs/elf3_reward_curriculum_v1.md)。该新课程不保证复现历史多阶段训练效果。

示例使用 seed 42，24 steps/env、4 mini-batches、5 PPO epochs，每 500 轮保存。`--iterations` 表示**总目标更新数**，`100001` 对应编号 0～100000，最终保存 `model_100000.pt`。专用入口省略轮数／环境数时默认仅做 64 环境、20 轮技术检查。

<img width="488" height="307" alt="图片" src="https://github.com/user-attachments/assets/d5723ece-a289-4e10-b206-c52dac99f2d6" />


环境数可按显存调整；不同硬件、环境数或初始化历史不保证得到相同训练曲线与策略。上述命令从随机权重开始，不会自动加载历史模型。

### 同任务续训

**79000轮预训练权重**通过[GitHub Release](https://github.com/JKYovo/DWAQ/releases/tag/elf3-force500-79000-v1)单独提供。需要resume／同配方微调时，使用该权重和固定500 N任务`elf3_dwaq_still_v2_force500_v1`，**不要用课程任务resume**；下载、SHA256及完整命令见[79000权重说明](docs/releases/elf3_force500_79000_v1.md)。

下面以踝部成本版为例，将路径替换为该任务的实际检查点：

```bash
./elf3.sh scripts/train_elf3_ankle_cost.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 100001 \
  --resume "/path/to/ankle_cost_run/model_50000.pt" \
  --run-name ankle_cost_resume_seed42 \
  --swanlab-project DWAQ
```

反馈基线使用 `scripts/train_elf3_feedback.py` 的 `--resume`；500 N版使用 `scripts/train_elf3_force500.py --resume /path/to/model.pt`，仅接受同配方500 N检查点。续训需匹配检查点的任务、环境及 Agent 契约，包括环境数；不能混用不同任务、旧资产版本或 AMP 权重。

续训恢复网络、Adam、学习率和迭代编号，但不恢复物理状态、地形课程等级、随机数流、观测历史或反馈滤波历史。专用入口写入新日志目录，不覆盖父模型。已有兼容父模型的显式迁移方式见各配方说明；从零训练不等同于复现某次历史续训。

### 日志与 SwanLab

日志分别写入：

```text
logs/elf3_dwaq_feedback_v1/<时间戳>_<run_name>/
logs/elf3_dwaq_still_v2_ankle_cost_v1/<时间戳>_<run_name>/
logs/elf3_dwaq_still_v2_force500_v1/<时间戳>_<run_name>/
logs/elf3_dwaq_still_v2_curriculum_v1/<时间戳>_<run_name>/
```

TensorBoard 为主日志，SwanLab 使用独立进程补传。首次使用请登录自己的账号：

```bash
./elf3.sh -m swanlab login
```

训练命令中的 `--swanlab-project DWAQ` 创建新 run；省略该参数则只保留本地日志。技术中断不会自动重启训练。

## Play 与 Sim2Sim

### Isaac Play：反馈基线

将 `YOUR_RUN` 和检查点文件名替换为 `logs/elf3_dwaq_feedback_v1/` 下的实际名称。零指令平地回放：

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task elf3_dwaq_feedback_v1 --num_envs 20 \
  --load_run "YOUR_RUN" --checkpoint model_100000.pt \
  --terrain flat --terrain_color concrete --command 0 0 0
```

训练混合地形最高等级：

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task elf3_dwaq_feedback_v1 --num_envs 20 \
  --load_run "YOUR_RUN" --checkpoint model_100000.pt \
  --terrain rough --difficulty 1.0 --terrain_color concrete \
  --command 0.6 0 0
```

`--command VX VY WZ` 固定机体坐标系速度指令；不指定时默认前进 `1.0 m/s`。Play 关闭观测噪声及周期推力。`rough --difficulty 1.0` 对应训练 Level 16，楼梯最高 30 cm；独立 `stairs` 测试地形的台阶为 20～25 cm，不等于该训练等级。

以上原生 Play 示例仅针对反馈基线。踝部成本版、500 N版及课程版 runner 有额外的完整配置校验，尚未确认兼容修改地形／环境数的通用 Play；请优先导出后使用下述 MuJoCo 回放。

### 导出 ONNX／TorchScript

公开任务使用同一导出入口。替换实际检查点路径，输出目录必须尚不存在：

```bash
./elf3.sh scripts/export_elf3_dwaq.py \
  --checkpoint "/path/to/run/model_100000.pt" \
  --output artifacts/elf3_policy_100000
```

输出 `policy.pt`、`policy.onnx` 和 `policy.json`。导出使用确定性 CENet 均值路径，校验权重、接口和数值一致性；原检查点只读。踝部成本版的零指令相位处理已嵌入模型，无需部署侧重新实现。

### MuJoCo 回放

使用导出的 `policy.pt`：

```bash
# 平地
./elf3.sh scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_policy_100000/policy.pt \
  --headless --steps 3000 --command 0.3 0 0

# 楼梯
./elf3.sh scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_policy_100000/policy.pt \
  --scene stairs --step-height 0.10 --command 0.3 0 0
```

仓库未集成 ELF3 实物通信 SDK。实物控制器需逐项对齐 `policy.json` 中的关节顺序、默认姿态、Kp/Kd、动作缩放、限幅、控制周期及历史顺序。使用既有硬件报告速度，**不要额外叠加训练中的 20 Hz 低通模型**，也不要对未要求归一化的输入自行增加归一化。

## 观测与资产契约

单帧观测为 100 维：

| 范围 | 内容 | 维度 |
| --- | --- | ---: |
| `0:3` | 躯干角速度（body frame） | 3 |
| `3:6` | 重力投影 | 3 |
| `6:9` | `vx, vy, yaw_rate` 命令 | 3 |
| `9:38` | 关节位置减默认姿态 | 29 |
| `38:67` | 关节速度 | 29 |
| `67:96` | 上一步动作 | 29 |
| `96:100` | 左右腿相位 sin/cos | 4 |

历史按**旧帧到新帧、逐帧排列**展平。部署接口为 `history [batch, 500] → actions [batch, 29]`。物理步长 5 ms（200 Hz），策略步长 20 ms（50 Hz）；20 Hz 是速度观测模型的截止频率，不是策略频率。高度扫描和真实接触信息不进入 Actor。

保留的 ELF3 USD 约 43.22 kg，并非 MelodyAI 的整套物理模型。关节参数见 [elf3.py](TienKung-Lab/legged_lab/assets/elf3/elf3.py)，接口见 [contract.json](TienKung-Lab/legged_lab/configs/elf3/contract.json)，来源及哈希见 [资产说明](docs/elf3_asset_provenance.json)。

## 地形课程

共 17 级（Level 0～16）：Level 0～9 保留原课程，楼梯逐步增至约 23 cm；Level 10～16 继续增加，最高 30 cm。混合比例为 55% 楼梯、10% 随机方格、15% 随机粗糙、10% 波浪、10% 斜坡，不包含宽沟地形。

离出生点超过 4 m 时升级；位移不足时按命令速度相关阈值降级。通过最高级后会重新随机分配等级，因此平均 terrain level 不一定持续停留在 16。

## 验证与项目结构

安装环境后可运行 CPU 回归测试，不会启动仿真或训练：

```bash
./elf3.sh -m unittest discover -s tests
```

若需训练链路 smoke test，可将对应专用训练入口设置为 `--num-envs 64 --iterations 20`，并使用独立 `--run-name`。技术检查通过不代表步态、越障或真机性能验收通过。

```text
elf3.sh                          # 项目环境启动入口
scripts/
  setup_elf3_env.sh               # 安装
  elf3_task_catalog.py            # 五个公开任务
  train_elf3_reward_curriculum.py # still_v2→最终500 N奖励课程
  train_elf3_force500.py          # 当前500 N配方，从零／同配方续训
  train_elf3_feedback.py          # 反馈基线
  train_elf3_ankle_cost.py        # 静止／踝部成本版
  export_elf3_dwaq.py             # 导出
  sim2sim_elf3_dwaq.py            # MuJoCo 回放
TienKung-Lab/legged_lab/
  assets/elf3/                    # elf3.py、usd/、urdf/、mjcf/、meshes/
  configs/elf3/                   # 机器人及接口契约
  envs/elf3/                      # 环境、奖励、配置与 runner
  terrains/                      # 地形课程
tests/                           # 回归测试
docs/                            # 配方、资产来源及历史记录
```

历史快照：[elf3-dwaq-stable-100k](https://github.com/JKYovo/DWAQ/tree/elf3-dwaq-stable-100k)；[历史延迟实验](docs/ELF3_DWAQ_GITHUB_IMPROVED_20260922.md)。历史入口不是当前公开任务。

## 上游与致谢

本项目基于以下开源项目：

- [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [Legged Lab](https://github.com/Hellod035/LeggedLab)
- [DreamWaQ / Manaro-Alpha](https://github.com/Manaro-Alpha/DreamWaQ)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

ELF3 物理模型和默认姿态来自 Amp_mjlab 的 ELF3 适配，执行器配置参考 MelodyAI。

## 许可证

项目采用 [BSD-3-Clause License](LICENSE)。各上游组件和资产仍遵循其各自许可证。
