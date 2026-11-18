# ELF3 DWAQ

ELF3 29 自由度机器人的无视觉行走、楼梯越障与平地起身训练，基于 Isaac Lab 和 DWAQ。
支持 17 级混合地形（楼梯最高 30 cm）、SwanLab 日志及 ONNX / TorchScript 导出。
起身阶段加入 AMP 动作先验，行走和起身由同一个策略完成。

[80000 轮预训练权重](https://github.com/JKYovo/DWAQ/releases/tag/elf3-recovery-initv4-80000-v1) · [课程说明](docs/elf3_reward_curriculum_v1.md)

<img width="488" height="307" alt="ELF3 仿真预览" src="https://github.com/user-attachments/assets/d5723ece-a289-4e10-b206-c52dac99f2d6" />

## 安装

环境：Ubuntu 22.04、Python 3.10、Isaac Sim 4.5.0、Isaac Lab 2.1.1。先准备 Conda 和 NVIDIA 驱动：

```bash
git clone https://github.com/JKYovo/DWAQ.git
cd DWAQ
bash scripts/setup_elf3_env.sh
```

以下命令均在仓库根目录执行。使用独立安装的 Isaac Sim 时，设置 `ELF3_ISAAC_SIM_DIR=/path/to/isaacsim`；其他安装选项见[安装脚本](scripts/setup_elf3_env.sh)。

## 训练课程

| 轮次 | 内容 |
| --- | --- |
| 0～20000 | 基础行走 |
| 20000～30000 | 渐进加入踝部、动作平滑及运动高度约束 |
| 30000～50000 | 保持最终行走配方 |
| 50001～80000 | 40% 平地起身＋60% 原地形行走 |

```bash
./elf3.sh scripts/train_elf3_recovery_curriculum.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 80001 \
  --swanlab-project DWAQ
```

保存 `model_50000.pt` 后自动进入起身阶段，最终保存 `model_80000.pt`。
`--iterations` 是总更新数，`80001` 对应编号 0～80000，每 500 轮保存。
已有同课程保存点时，追加 `--resume /path/to/model_50000.pt`；也支持课程中断点续训。

首次使用 SwanLab 先运行 `./elf3.sh -m swanlab login`；只需本地日志时去掉 `--swanlab-project DWAQ`。

## 预训练权重与续训

[80000 轮 Release](https://github.com/JKYovo/DWAQ/releases/tag/elf3-recovery-initv4-80000-v1) 提供完整检查点、SHA256 和配置摘要，包含策略、Adam 及 AMP 续训状态。

```bash
mkdir -p pretrained
curl --fail --location \
  https://github.com/JKYovo/DWAQ/releases/download/elf3-recovery-initv4-80000-v1/model_80000.pt \
  --output pretrained/model_80000.pt
```

从 80001 续训至 100000，使用起身阶段入口：

```bash
./elf3.sh scripts/train_elf3_recovery_stage.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 100001 \
  --resume pretrained/model_80000.pt --swanlab-project DWAQ
```

续训需匹配保存点的配方和环境数；仿真状态、地形等级及观测历史会重新初始化。

## 导出与仿真

导出 ONNX / TorchScript，输出目录需尚不存在：

```bash
./elf3.sh scripts/export_elf3_dwaq.py \
  --checkpoint pretrained/model_80000.pt --output artifacts/recovery80000
```

输出 `policy.onnx`、`policy.pt`、`policy.json`，使用确定性均值推理。

MuJoCo 平地起身：

```bash
./elf3.sh scripts/play_elf3_recovery_mujoco.py \
  --policy artifacts/recovery80000/policy.onnx \
  --command 0 0 0 --reset-seconds 0
```

MuJoCo 楼梯行走（示例台阶 10 cm）：

```bash
./elf3.sh scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/recovery80000/policy.pt \
  --scene stairs --step-height 0.10 --command 0.3 0 0
```

部署接口为 `history [batch, 500] → actions [batch, 29]`，历史按 5 帧、每帧 100 维从旧到新排列。
策略频率 50 Hz，物理频率 200 Hz。关节顺序、默认姿态、PD 和动作缩放以导出的 `policy.json` 为准。

## 更多

- [课程参数与起身配方](docs/elf3_reward_curriculum_v1.md)
- [机器人与观测接口](TienKung-Lab/legged_lab/configs/elf3/contract.json) · [资产来源](docs/elf3_asset_provenance.json)
- 其他训练入口：`./elf3.sh scripts/elf3_task_catalog.py`
- CPU 回归检查：`./elf3.sh -m unittest discover -s tests`

## 致谢与许可证

基于 [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)、[TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)、[Legged Lab](https://github.com/Hellod035/LeggedLab)、[DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)。ELF3 适配来自 Amp_mjlab，执行器配置参考 MelodyAI。

采用 [BSD-3-Clause](LICENSE)，上游组件和资产保留各自许可证。
