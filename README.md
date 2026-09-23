# ELF3 DWAQ：盲走与楼梯地形强化学习

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.1-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-yellow.svg)](LICENSE)

本项目将 DWAQ（DreamWaQ 风格的历史观测编码器 + PPO）适配到 **ELF3 29 自由度人形机器人**，用于无视觉地形适应、盲走和楼梯训练。训练框架基于 Isaac Lab、TienKung-Lab、Legged Lab 与 RSL-RL。

公开的 10 万轮稳定基线标签为 [`elf3-dwaq-stable-100k`](https://github.com/JKYovo/DWAQ/tree/elf3-dwaq-stable-100k)。当前本地 GitHub 改进版增加 0/20/40 ms 动作延迟训练和延迟缓冲区前的动作裁剪，已完成的改进及存档见 [docs/ELF3_DWAQ_GITHUB_IMPROVED_20260922.md](docs/ELF3_DWAQ_GITHUB_IMPROVED_20260922.md)。训练权重不提交到 Git 仓库。

## 功能

- ELF3 原生 29 关节顺序、默认姿态、PD、力矩限制和动作缩放
- 100 维本体观测，5 帧历史输入，无视觉/高度扫描 Actor 输入
- DWAQ VAE 速度估计与隐变量编码
- 上身左右镜像损失和软默认姿态损失
- 17 级渐进地形，最高 30 cm 台阶
- VAE 数值修复、有限值检查、梯度检查和损坏 checkpoint 拒绝保存
- TensorBoard 与 SwanLab 训练记录
- 确定性 TorchScript/ONNX 导出及 MuJoCo Sim2Sim 验证

> DWAQ 本身不包含 AMP。本项目的 ELF3 DWAQ 训练也不加载 AMP 动作数据或 AMP 权重。

## 稳定任务

推荐任务：

```text
elf3_dwaq_upper_symmetry_pose_delay
```

## 环境

已验证环境：

- Ubuntu 22.04
- NVIDIA RTX 4090
- Python 3.10.20
- PyTorch 2.5.1 + CUDA 12.4
- Isaac Sim 4.5.0（standalone）
- Isaac Lab 2.1.1
- MuJoCo 3.3.2

### 安装

先安装与机器匹配的 Isaac Sim 4.5 和 Isaac Lab 2.1，然后执行：

```bash
git clone https://github.com/JKYovo/DWAQ.git
cd DWAQ

bash scripts/setup_elf3_env.sh
```

安装脚本默认使用以下本机路径：

```text
Isaac Sim: /home/cheng/isaacsim
Isaac Lab: /home/cheng/Desktop/IsaacLab
Python:    <项目目录>/.venv/bin/python
```

可以通过环境变量覆盖：

```bash
ELF3_CONDA=/path/to/conda \
ELF3_SOURCE_ENV=/path/to/source/env \
ELF3_ISAACLAB_DIR=/path/to/IsaacLab \
bash scripts/setup_elf3_env.sh
```

启动训练与 Isaac Lab 仿真时使用根目录的 `elf3.sh`。如 Isaac Sim 不在默认目录：

```bash
ELF3_ISAAC_SIM_DIR=/path/to/isaacsim ./elf3.sh <python-script> [args...]
```

## 从头训练 10 万轮

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq_upper_symmetry_pose_delay \
  --headless \
  --num_envs=4096 \
  --seed=42 \
  --max_iterations=100001 \
  --run_name=elf3_dwaq_stable_100k \
  --logger=tensorboard \
  --swanlab_project=DWAQ \
  --swanlab_experiment_name=ELF3_DWAQ_stable_100k
```

训练日志保存在：

```text
logs/elf3_dwaq_upper_symmetry_pose_delay/<时间戳>_<run_name>/
```

默认每 100 轮保存一次 checkpoint。`max_iterations` 在续训时表示本次追加的更新次数，不是绝对终点。

### 续训

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq_upper_symmetry_pose_delay \
  --headless \
  --num_envs=4096 \
  --seed=42 \
  --resume=True \
  --load_run=<运行目录名> \
  --checkpoint=model_100000.pt \
  --max_iterations=10001 \
  --run_name=resume_from_100000 \
  --logger=tensorboard \
  --swanlab_project=DWAQ
```

续训恢复策略、优化器和迭代编号。并行环境的 terrain level 不保存在 checkpoint 中，新进程会从配置的初始等级重新建立课程。

## 地形课程

训练地形共 17 级（Level 0–16）：

- Level 0–9 保留原有课程，楼梯逐步增加到约 23 cm
- Level 10–16 继续增加楼梯高度
- Level 16 楼梯最高 30 cm
- 训练混合：55% 楼梯、10% 随机方格、15% 随机粗糙、10% 波浪、10% 斜坡
- 宽沟地形已移除，因为纯本体感觉策略无法在接触前感知宽沟

机器人离地形出生点超过 4 m 时升级；未达到与命令速度相关的距离阈值时降级。通过 Level 16 后，Isaac Lab 会将该环境随机送回其他等级，因此 TensorBoard 中的平均 level 不会长期停在 16。

## Play

### 训练地形最高等级（包含最高 30 cm 台阶）

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task=elf3_dwaq_upper_symmetry_pose_delay \
  --num_envs=20 \
  --load_run=<运行目录名> \
  --checkpoint=model_100000.pt \
  --terrain=rough \
  --difficulty=1.0 \
  --terrain_color=concrete
```

`--terrain=rough --difficulty=1.0` 对应训练课程的 Level 16。`--terrain=stairs` 使用独立的纯台阶测试配置，台阶范围为 20–25 cm，并不对应 30 cm 训练等级。

其他测试地形：

```bash
# 平地
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task=elf3_dwaq_upper_symmetry_pose_delay --terrain=flat \
  --load_run=<运行目录名> --checkpoint=model_100000.pt

# 纯台阶
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task=elf3_dwaq_upper_symmetry_pose_delay --terrain=stairs --difficulty=1.0 \
  --load_run=<运行目录名> --checkpoint=model_100000.pt
```

Play 会关闭观测噪声与周期推力，并固定前进命令为 `1.0 m/s`。

## 策略接口

Actor 单帧观测为 100 维：

| 范围 | 内容 | 维度 |
| --- | --- | ---: |
| `0:3` | 躯干角速度（body frame） | 3 |
| `3:6` | 重力投影（body frame） | 3 |
| `6:9` | `vx, vy, yaw_rate` 命令 | 3 |
| `9:38` | 关节位置减默认姿态 | 29 |
| `38:67` | 关节速度 | 29 |
| `67:96` | 上一步动作 | 29 |
| `96:100` | 左右腿相位的 sin/cos | 4 |

DWAQ 历史输入为 5 帧，共 500 维，顺序为旧帧到新帧。Actor 输出 29 维归一化关节动作。

## 导出

使用 ELF3 专用导出器。输出目录必须是尚不存在的新目录：

```bash
.venv/bin/python scripts/export_elf3_dwaq.py \
  --checkpoint logs/elf3_dwaq_upper_symmetry_pose_delay/<运行目录>/model_100000.pt \
  --output artifacts/elf3_dwaq_model_100000
```

输出文件：

```text
policy.pt    TorchScript
policy.onnx  ONNX
policy.json  机器人契约、关节顺序、控制参数和 checkpoint 哈希
```

导出使用编码器的 mean velocity 和 mean latent，不进行随机采样。ONNX 部署接口固定为：

```text
input:  history [batch, 500]
output: actions [batch, 29]
```

导出器会检查 ELF3 资产契约、全部权重、有限输出、TorchScript 一致性和 ONNX Runtime 数值误差。checkpoint 始终只读。

## MuJoCo Sim2Sim

```bash
# 平地、有界运行
.venv/bin/python scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_dwaq_model_100000/policy.pt \
  --headless --steps 3000 --command 0.3 0 0

# 楼梯
.venv/bin/python scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_dwaq_model_100000/policy.pt \
  --scene stairs --step-height 0.10 --command 0.3 0 0
```

该脚本用于 Sim2Sim 验证；本仓库没有集成 ELF3 实物通信 SDK。实物控制器必须严格遵循 `policy.json` 中的关节顺序、默认姿态、动作缩放、控制周期和历史顺序。

## 项目结构

```text
.
├── elf3.sh                              # Isaac Sim/Isaac Lab 启动入口
├── requirements-elf3.txt                # ELF3 环境依赖
├── scripts/
│   ├── setup_elf3_env.sh                # 环境安装
│   ├── export_elf3_dwaq.py              # 确定性导出
│   ├── sim2sim_elf3_dwaq.py             # MuJoCo 回放
│   └── check_elf3.py                    # 资产/接口/训练检查
├── TienKung-Lab/legged_lab/
│   ├── assets/elf3/                     # ELF3 USD、URDF、MJCF、网格和契约
│   ├── envs/elf3/                       # ELF3 环境、配置和 runner
│   ├── terrains/                        # 17 级课程与 30 cm 台阶映射
│   └── scripts/                         # train.py / play.py
└── docs/ELF3_VALIDATION.json            # 机器可读验证结果
```

## 验证

```bash
# 平地接口与 PPO 更新检查
./elf3.sh scripts/check_elf3.py \
  --headless --terrain flat --num_envs 16 \
  --output artifacts/check_flat.json

# 粗糙地形检查
./elf3.sh scripts/check_elf3.py \
  --headless --terrain rough --num_envs 32 \
  --output artifacts/check_rough.json
```

检查覆盖 ELF3 资产质量/惯量、关节顺序、默认姿态、动作缩放、脚接触索引、观测历史、PPO 更新、checkpoint 契约和有限值。

## 上游与致谢

本项目基于以下开源项目：

- [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [Legged Lab](https://github.com/Hellod035/LeggedLab)
- [DreamWaQ / Manaro-Alpha](https://github.com/Manaro-Alpha/DreamWaQ)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

ELF3 资产参数和默认姿态参考同级项目 Amp_mjlab 的 ELF3 适配，来源与哈希记录在 `TienKung-Lab/legged_lab/assets/elf3/provenance.json`。

## 许可证

项目采用 [BSD-3-Clause License](LICENSE)。各上游组件和资产仍遵循其各自许可证。
