# ELF3 DWAQ：0918 训练基线与新资产

本版训练核心恢复到 `2362a3ade4f12846ee26c2da0d33a6634cad5349`，采用该提交完整注册的
`elf3_dwaq_upper_symmetry_pose` 环境、Agent 和 `Elf3DwaqUpperBodySymmetryRunner`，
不是给基础任务临时添加两个损失系数。有效训练配置与本机正在运行的 0918 实验逐项核对。

本版不启用后续的动作延迟、动态速度奖励、特殊指令采样、制动、躯干高度或默认腿姿态强化实验。
训练算法、奖励、地形和域随机化均恢复到上述提交；新增资产及启动／检查工具另行列明。

## 配置

| 项目 | 发布入口的有效值 |
| --- | --- |
| 任务 | `elf3_dwaq_upper_symmetry_pose` |
| Runner | `Elf3DwaqUpperBodySymmetryRunner` |
| 初始化 | 随机权重、全新优化器，不加载旧 checkpoint |
| 环境数／seed | 4096／42 |
| rollout／mini-batches／PPO epochs | 24／4／5 |
| 学习率 | 初始 `1e-4`，adaptive 范围 `1e-5～3e-4` |
| 上肢镜像／姿态辅助系数 | `0.1／0.02` |
| 速度跟踪奖励 | 线／角速度权重均 `2.0`，std 均 `0.5` |
| 迭代与保存 | 100001 次更新，编号 0～100000，每 500 次保存 |
| 策略接口 | 5×100 历史 → 29 动作，0.8 秒相位时钟 |
| 控制 | 物理 dt=0.005 秒，decimation=4，策略 50 Hz |

注册 Agent 原始默认值仍是 20000 次更新、每 100 次保存，以保持 0918 源码不变。
**本页启动器**明确覆盖训练预算、保存间隔、环境数、seed、TensorBoard 和 run name；
运行时核验最终 `params/agent.yaml` 及实际 PPO 辅助损失配置。

## 资产：参数对齐不等于整套物理模型相同

入口为 [elf3.py](TienKung-Lab/legged_lab/assets/elf3/elf3.py)，按 MelodyAI 的具名执行器分组格式维护。
[contract.json](TienKung-Lab/legged_lab/configs/elf3/contract.json) 是观测、动作及机器人参数契约；
构建 DWAQ 环境时逐关节检查两者一致，参数调整必须同步契约。资产包本身不依赖这些训练元数据。

- 保留当前约 **43.22 kg** 的 USD、URDF、MJCF、网格、碰撞与刚体惯量；没有换成 MelodyAI 训练仓库约 37.87 kg 的 USD。
- 使用 `ImplicitActuatorCfg`，求解器位置／速度迭代 8／4。
- 全部 29 个关节显式 `armature=0.01`；全部速度上限 **20 rad/s**。
- 腰三个关节及双侧髋 pitch／roll **100 Nm**；髋 yaw **50 Nm**；膝 **150 Nm**；
  踝 pitch／roll **50／20 Nm**；肩 pitch／roll、肘 **50 Nm**；肩 yaw 和腕 **25 Nm**。
- 默认关节姿态、Kp/Kd 与动作尺度不重新计算。目标仍为 `默认关节位置 + action × action_scale`。
- MelodyAI 原始 Isaac 配置未显式设置 armature，本版显式 0.01 是额外选择，不声称与其完全相同。
- 原始配置快照位于 `scripts/references/melody_elf3.py`，仅用于来源核验，不是另一个待使用的 USD。

资产目录按配置与物理文件组织：

```text
assets/elf3/
├── __init__.py
├── elf3.py
├── usd/
├── urdf/
├── meshes/
└── mjcf/
```

训练／部署契约在 `legged_lab/configs/elf3/contract.json`，加载和校验逻辑在 `legged_lab/elf3_contract.py`；
来源记录在 `docs/elf3_asset_provenance.json`，资产独立检查入口为 `scripts/check_elf3_asset.py`。
MJCF 位于 `mjcf/elf3.xml`，URDF 与 MJCF 均相对引用 `../meshes/`，复制整个资产目录即可保留几何依赖。
历史 checkpoint 契约中的物理文件路径仍作来源标识；检查器只允许这些已知路径迁移，仍拒绝几何或动力学数值变化。
历史 Amp 导入器只允许输出到新归档目录，不再覆盖当前资产和训练配置。

## 安装

已验证依赖栈：Python 3.10、PyTorch 2.5.1+cu124、Isaac Sim 4.5.0、Isaac Lab 2.1.1。
保持本机原有安装脚本和 `elf3.sh`；新机器应创建自己的环境，不复制其他机器的 `.venv`。

```bash
git clone https://github.com/JKYovo/DWAQ.git
cd DWAQ
bash scripts/setup_elf3_env.sh
```

可用 `ELF3_CONDA` 和 `ELF3_ISAACLAB_DIR` 指定安装位置；standalone Isaac Sim 可设置
`ELF3_ISAAC_SIM_DIR`。训练的 Python、PYTHONPATH 和日志均相对当前 checkout，
不依赖 `/home/cheng/DWAQ_0918` 或其历史日志。

## 检查与训练

先进行只读源码／资产指纹检查：

```bash
.venv/bin/python scripts/release_0918_checks.py
./elf3.sh scripts/train_0918_melody.py --headless --verify-config-only
```

独立技术短测（不上传 SwanLab，权重不用于正式训练）：

```bash
./elf3.sh scripts/train_0918_melody.py --headless \
  --num-envs 64 --iterations 20 --run-name release_smoke64_20

./elf3.sh scripts/check_melody_smoke.py \
  --run logs/elf3_dwaq_upper_symmetry_pose/<短测目录>
```

确认 SwanLab 已登录后，正式从零训练：

```bash
bash scripts/train_0918_melody.sh
```

该固定入口创建独立 run，不恢复已有模型，也不因技术崩溃自动重启。
仅检查／整理此仓库不会启动这个正式训练命令。

日志在 `logs/elf3_dwaq_upper_symmetry_pose/<时间戳>_<run_name>/`。
TensorBoard 是主日志；SwanLab 在独立进程补传，保留超时、心跳和已确认上传游标。
`pose_scratch_swanlab.py` 的文件名是历史命名，功能仅为通用日志传输，不加入旧实验奖励。

每次运行保存最终环境／Agent YAML、覆盖清单、源码指纹、从零初始化记录及运行时 PhysX 参数检查结果。
[训练清单](docs/elf3_0918_training_manifest.json) 固定训练核心和发布配方的 SHA256，
不要求 Git HEAD 永远等于旧提交，因此新提交、推送和重新 clone 后仍能核验。
有意修改配方时需重新验证并更新该清单；不要绕过检查后宣称仍与本配方相同。

## 导出与部署

```bash
./elf3.sh scripts/export_elf3_dwaq.py \
  --checkpoint logs/elf3_dwaq_upper_symmetry_pose/<运行目录>/model_100000.pt \
  --output artifacts/elf3_model_100000_export
```

输出 `policy.onnx`、`policy.pt`、`policy.json`；输出目录必须新建。
即使是 CPU 导出／短测校验，也建议使用 `elf3.sh`，以避免系统 CUDA 动态库覆盖当前环境的依赖。
采用确定性编码均值路径，输入 `float32[batch,500]`、输出 `[batch,29]`。
模型与 JSON 成对部署；JSON 本身不自动修改部署端力矩限制、armature、PD 或电机映射。
相同输入维度不代表旧资产契约的 checkpoint 可直接续训，导出器和 Runner 会拒绝不匹配契约。

MuJoCo 回放脚本及已有部署修改保留；回放不等于真机验证。
这次整理不宣称解决真机抖动，也不上传权重、日志、认证信息或本机归档。
仓库继承的 G1 示例权重不属于本次 ELF3 发布模型。

## 来源与历史边界

- 训练基线：`2362a3ade4f12846ee26c2da0d33a6634cad5349`。
- MelodyAI 配置来源及保留物理文件哈希：见 `legged_lab/configs/elf3/contract.json`、`docs/elf3_asset_provenance.json`。
- 历史延迟／split-root 文档仅为存档，不是本版训练说明。
- 旧指令响应、quality-v3、pose-scratch 和 blind-x 实验源码已从活动源码树移入本机备份；
  既有训练权重和日志保留。备份在 `artifacts/`，不提交到 Git。
