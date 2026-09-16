# ELF3 DWAQ：环境与适配说明

项目目录：`/home/cheng/G1DWAQ_Lab`，与 `/home/cheng/Amp_mjlab` 同级。

上游为 [liuyufei-nubot/G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab)，
本次拉取 `main` 最新提交 `bebb0ea413f3bc00860481c2f11acf30d9c416cf`（浅克隆）。
本次保留 **IsaacLab / PhysX 训练框架**，参考 Amp_mjlab 适配 ELF3。

## 1. 使用方式

环境已经装在本项目 `.venv`。在项目根目录运行：

```bash
cd /home/cheng/G1DWAQ_Lab

# 粗糙地形 DWAQ，从头训练；继承上游地形、课程、奖励和 PPO 参数。
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq --headless --num_envs=4096 --max_iterations=100001

# 平地任务，使用同一组奖励系数。
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq_flat --headless --num_envs=4096 --max_iterations=100001
```

默认使用本地 TensorBoard，日志分别写入 `logs/elf3_dwaq/`、`logs/elf3_dwaq_flat/`。
两个 ELF3 任务的默认训练轮数均为 100001，命令行可省略 `--max_iterations`。
每 100 轮保存；首次从头初始化，不自动加载 G1 或 AMP 的已有模型。

```bash
.venv/bin/tensorboard --logdir logs --port 6006

# 续训：填入本项目训练得到的运行目录名及 checkpoint 文件名。
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq --headless --num_envs=4096 --max_iterations=10000 \
  --resume True --load_run=<运行目录名> --checkpoint=model_1000.pt

# IsaacLab 回放：--terrain rough 使用训练地形，flat 使用平地。
./elf3.sh TienKung-Lab/legged_lab/scripts/play.py \
  --task=elf3_dwaq --num_envs=1 --terrain=rough --terrain_color=concrete \
  --load_run=<运行目录名> --checkpoint=model_1000.pt
```

续训沿用上游语义：`max_iterations` 是本次追加的轮数；输出进入新运行目录。
`play.py` 沿用上游的推断采样；下面的导出和 MuJoCo 回放使用上游导出器的编码均值路径。

## 2. 适配内容和参数来源

| 内容 | 本次处理 |
| --- | --- |
| 模型来源 | 复制 Amp_mjlab 的 `src/assets/robots/elf3/xmls`，含原始 MJCF、STL |
| IsaacLab 模型 | 由该 MJCF 生成 URDF，再用 IsaacLab 转成 USD；保留已合并的头部质量/惯量与碰撞选择 |
| 物理根/机身观测 | `torso_link`；`waist_z_link` 仍是骨盆语义体，不作为浮动根 |
| 默认姿态 | Amp_mjlab 的原生 ELF3 默认姿态，初始躯干高度 1.05 m |
| 控制 | 5 ms 物理步、4 子步、50 Hz 策略；原生每关节 PD、armature、力矩上限和动作缩放 |
| 执行器 | IsaacLab `IdealPDActuator` 显式位置 PD，实际力矩按每关节上限裁剪 |
| 动作/观测顺序 | 始终采用 Amp_mjlab `ELF3_JOINT_NAMES`；PhysX 内部顺序显式转换 |
| 脚 | 左 `l_ankle_x_link`，右 `r_ankle_x_link`，分别解析机器人和接触传感器编号 |
| 奖励 | 上游 26 项奖励函数、系数及数值阈值全部保留；仅映射身体和关节名称 |
| 域随机化、命令、课程 | 沿用上游；质体随机化目标映射到 ELF3 躯干 |
| 显示 | 地形使用本地 PreviewSurface 材质，避免启动时下载远程地面材质 |

`TienKung-Lab/legged_lab/assets/elf3/contract.json` 保存完整机器人参数。
同目录 `provenance.json` 保存来源提交与文件 SHA-256，`amp_elf3_constants.py.reference`
是用于核对的源文件快照。运行不依赖 Amp_mjlab 的 Python 包或 `.venv`。

未使用 Amp_mjlab 归档 V5 DWAQ 中的教师蒸馏、AMP 混合奖励或自定义障碍课程。
没有读取或改写 Amp_mjlab 的训练 checkpoint / ONNX。

### 上游改动范围

- `legged_lab/envs/__init__.py`：注册 `elf3_dwaq` 和 `elf3_dwaq_flat`。
- `legged_lab/envs/g1/g1_dwaq_env.py`：两处脚位置查询允许单独的机器人身体编号；
  地形估计的无接触回退高度允许按机器人提供。G1 默认仍使用原编号和 0.78 m。
- `legged_lab/utils/env_utils/scene.py`：允许 ELF3 使用无远程 HDR 的天空光照，G1 默认不变。
- `legged_lab/scripts/train.py` / `play.py`：ELF3 结束时显式清理仿真回调，
  避免本机 IsaacLab 的停止回调在关闭 stage 时继续渲染而卡住。
- `.gitignore`：忽略新环境、日志及验证输出。
- 其他 ELF3 功能均位于新增目录/脚本中。

上游 `actor_critic_DWAQ.py`、`dwaq_ppo.py`、`rollout_storage_dwaq.py`、
`dwaq_on_policy_runner.py`、`g1_dwaq_config.py`、`mdp/rewards.py` 保持字节一致。
新增训练器子类只保存/校验机器人契约，训练过程继续调用原版。
相同的 29 动作、100 维观测并不代表 G1 权重兼容 ELF3；加载不匹配契约会明确报错。

### 观测接口

当前 Actor 100 维：

```text
0:3     躯干角速度，body frame
3:6     单位重力投影，body frame
6:9     vx、vy、yaw_rate 命令
9:38    29 维关节位置减默认姿态
38:67   29 维关节速度
67:96   29 维上一次动作
96:100  sin(left), sin(right), cos(left), cos(right)
```

历史为 5×100=500 维，旧帧在前；重置后首帧填满历史。
步态周期 0.8 s、左右相差半周期。Critic 是 311 维，Actor 无地形扫描。
网络保持上游编码器、速度/隐变量头、解码器和 Actor/Critic；无经验归一化。
24 步 rollout、5 epoch、4 minibatch、初始学习率 1e-3、entropy_coef=0.01。

保留的上游行为也包括：`feet_swing_height` 使用世界坐标足高与 0.08 m 目标，
以及上游的扫描高度偏置、历史接触取值和 VAE 采样/损失定义。本次没有借适配调整这些设计。

## 3. 导出与 MuJoCo 回放

导出目录必须是新目录，checkpoint 始终只读。导出器要求所有权重完整匹配，
不会把缺失层保留成随机初始化，也不会裁剪、重排或拼接旧网络权重。

```bash
.venv/bin/python scripts/export_elf3_dwaq.py \
  --checkpoint logs/elf3_dwaq/<运行目录>/model_1000.pt \
  --output artifacts/elf3_model_1000_export

# 图形回放；输入 500 维历史，输出 29 维原生 ELF3 动作。
.venv/bin/python scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_model_1000_export/policy.pt --command 0.3 0 0

# 无界面检查。
.venv/bin/python scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_model_1000_export/policy.pt --headless --steps 1000

# 简单上升台阶场景。
.venv/bin/python scripts/sim2sim_elf3_dwaq.py \
  --policy artifacts/elf3_model_1000_export/policy.pt --scene stairs --step-height 0.05
```

输出包含 `policy.pt`、`policy.onnx`、`policy.json`。JSON 记录原 checkpoint 哈希、
关节顺序、默认姿态、缩放、相位和控制频率。ONNX Runtime 验证 batch 1/3/8 的导出一致性。
MuJoCo 脚本按名称查找电机，不能把 XML 中的 motor 声明顺序直接当成动作顺序。
该入口为仿真回放；没有接入 ELF3 实物通信 SDK。

## 4. 本机环境与复现

本次验证环境：Python 3.10.20、PyTorch 2.5.1+cu124、Isaac Sim 4.5.0、
IsaacLab 2.1.1（Python 包 `isaaclab==0.41.3`）、MuJoCo 3.3.2、RTX 4090。
上游 README 的 Python 版本建议与本机不同；这里采用兼容本机 Isaac Sim 4.5 的 Python 3.10，
并以实际运行验证为准。

- 独立 Python 环境：`/home/cheng/G1DWAQ_Lab/.venv`。
- 共享、只使用现有 Isaac Sim：`/home/cheng/isaacsim`。
- 共享、只使用现有 IsaacLab 源码：`/home/cheng/Desktop/IsaacLab`。
- 本项目安装自己的 `TienKung-Lab/rsl_rl`，没有替换 Amp_mjlab 环境中的 AMP fork。

`elf3.sh` 配好独立 Python、项目导入路径和 standalone Isaac Sim 动态库路径，
训练及 IsaacLab 回放应使用这个入口。
仅导出和 MuJoCo 回放可直接使用 `.venv/bin/python`。

依赖版本见 `requirements-elf3.txt`。在本机重建环境：

```bash
bash scripts/setup_elf3_env.sh
.venv/bin/python -m pip check
```

脚本默认克隆本机 `isaacsim45` 环境到新项目，再安装本项目依赖及 editable 包。
可用 `ELF3_CONDA`、`ELF3_SOURCE_ENV`、`ELF3_ISAACLAB_DIR` 覆盖这些路径；
启动器可用 `ELF3_ISAAC_SIM_DIR` 覆盖 Isaac Sim 路径。
另一台机器需要先安装对应的 Isaac Sim / IsaacLab，不能只复制 `.venv`。

需要重新从 Amp_mjlab 构建机器人资产时：

```bash
.venv/bin/python scripts/import_elf3_from_amp.py --amp /home/cheng/Amp_mjlab
./elf3.sh /home/cheng/Desktop/IsaacLab/scripts/tools/convert_urdf.py \
  TienKung-Lab/legged_lab/assets/elf3/urdf/elf3.urdf \
  TienKung-Lab/legged_lab/assets/elf3/usd/elf3.usd \
  --joint-stiffness 0 --joint-damping 0 --joint-target-type none --headless
```

## 5. 验证与当前边界

```bash
# 每次指定一个新的输出路径；脚本包含两轮真实 PPO 更新。
./elf3.sh scripts/check_elf3.py --headless --output artifacts/check_flat_new.json
./elf3.sh scripts/check_elf3.py --headless --terrain rough --num_envs 32 \
  --output artifacts/check_rough_new.json
```

验证覆盖 USD 刚体质量/惯量、MuJoCo/PhysX 默认姿态 FK、原生 PD 参数、动作与观测顺序、
两种脚编号、重置历史、相位、数据有限性、PPO 更新及 checkpoint 恢复/拒绝不匹配契约。
检查脚本为核对几何/映射暂时关闭随机化并缩小测试地形；正式训练配置保持上游默认。

已完成平地 16 环境、粗糙地形 32 环境的接口验证，以及 **4096 环境、默认粗糙地形和随机化的 3 轮训练**。
30 个刚体总质量约 43.222508 kg，默认姿态 FK 最大位置差约 3.6e-7 m。
最终导出 ONNX 与 PyTorch 最大绝对误差 1.49e-7。
环境依赖 `pip check` 无冲突；最终检查和 4096 环境训练均正常退出。
可核对的机器可读结果见 [docs/ELF3_VALIDATION.json](docs/ELF3_VALIDATION.json)。

这些是工程流程验证。`artifacts/*_ppo`、`logs/*/*adapter_smoke*` 下模型均为短测模型，
尚未完成正式行走/上台阶训练；MuJoCo 短测模型会跌倒，不能据此宣称策略已经可用。
正式长训和真机效果仍需后续训练与评估。

## 6. SwanLab（与 Amp_mjlab 相同的接入方式）

使用 SwanLab 0.8.5、现有登录账号和 `locomotion` 项目。新训练在创建
TensorBoard SummaryWriter 前调用 `swanlab.init()`、`swanlab.sync_tensorboard_torch()`：

```bash
./elf3.sh TienKung-Lab/legged_lab/scripts/train.py \
  --task=elf3_dwaq --headless --num_envs=4096 --logger=tensorboard \
  --swanlab_project=locomotion --swanlab_experiment_name=ELF3_DWAQ_100001_4096env
```

默认训练 100001 轮；链接保存到训练目录的 `swanlab_run.json`。
恢复同一个云端实验可指定 `--swanlab_id=<id> --swanlab_resume=must`，
同时使用训练入口原有 checkpoint 恢复参数。不要让两个写入进程同时更新同一个实验。

已经运行、只有 TensorBoard 的训练可以单独挂接，无需重启训练：

```bash
.venv/bin/python -u scripts/swanlab_tail.py \
  --log-dir logs/elf3_dwaq/<训练目录> --training-pid <训练进程PID> \
  --project locomotion --name ELF3_DWAQ_100001_4096env
```

该脚本回填已有标量，每 10 秒增量读取新数据；保留每个指标自己的 step，
包括以秒为横轴的 `/time` 指标。文件锁阻止重复挂接，状态记录在
`swanlab_bridge_state.json`；重启同步脚本时会自动恢复原实验并跳过已入队数据。
训练退出后再读取两次并结束 SwanLab 实验。同步脚本需自行放入持久终端或后台运行。
状态里的 `scalars_queued` 是提交给 SDK 的数量，云端上传由 SDK 异步完成。

当前 2026-09-15 17:51:51 的 4096 环境训练已用独立后台进程挂接：
[ELF3_DWAQ_100001_4096env](https://swanlab.cn/@Rainbowow/locomotion/runs/9lpz7vwy)。
同步控制台见该训练目录下 `swanlab_bridge.log`，启动参数见 `swanlab_bridge_launch.json`。
依赖中更新了克隆环境旧版 W&B/rl-games 以兼容 SwanLab 所需 protobuf 6；
ELF3 训练使用本仓库的 RSL-RL DWAQ，算法和奖励权重未因此调整。
