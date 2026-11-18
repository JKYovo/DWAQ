# ELF3 still_v2 奖励课程 v1

新任务 `elf3_dwaq_still_v2_curriculum_v1`。不修改历史任务和权重，不改变资产、
隐式PD、armature=0.01、20 Hz关节速度观测模型、Actor/CENet、PPO、命令采样或地形。

完整行走＋起身课程入口为 `scripts/train_elf3_recovery_curriculum.py`；以下 0～50000
时间表保留原样，50000 保存点后的新增阶段见本文末尾。

## 时间表

默认从零：0～20000使用still_v2奖励；20000～30000线性过渡；30000～50000保持最终奖励。
alpha = clamp((iteration-ramp_start)/(ramp_end-ramp_start), 0, 1)。
轮次是现有runner的绝对索引，`--iterations 50001`结束于`model_50000.pt`。

| 奖励项 | 起点权重 | 终点权重 |
| --- | ---: | ---: |
| ankle_torque_l2 | 0 | -0.0005 |
| stance_foot_speed_cost | 0 | -1 |
| base_height | 0 | -50 |
| action_rate_l2 | -0.01 | -0.012 |
| action_smooth | 0 | -0.001 |
| dof_acc_l2 | -2.5e-7 | -3e-7 |
| feet_force | -0.003 | -0.005 |

足力所有阶段均为500 N起罚，超限截断400 N；所有阶段从4个5 ms物理子步中取
同时双脚Fz范数峰值。采样修复不参与渐变，也不经过400 N阈值。
因此第一阶段不是历史still_v2逐字节等价目标：有全程峰值采样修复。
旧任务仍保持旧采样方式，不做全局monkeypatch。

零指令关闭周期驱动、原有实际静止与腿腰姿态约束不淡入；运动高度项保留原有
非零指令/有效支撑/局部平坦地形门控。扫描不进入Actor。

## 使用

从零（环境和PPO规格与历史主线相同）：

```bash
./elf3.sh scripts/train_elf3_reward_curriculum.py --headless --device cuda:0 \
  --num-envs 16384 --iterations 50001 --seed 42 \
  --ramp-start 20000 --ramp-end 30000 --run-name curriculum_scratch \
  --swanlab-project DWAQ
```

原始still_v2的18500检查点提前进入二阶段：使用`--parent-checkpoint`和显式
`--parent-sha256`，同时设`--ramp-start 18500 --ramp-end 28500`。
实际从18501继续，初始alpha=0.0001，不伪造已完成轮次，不一次性打开终点奖励。
历史父权重没有用峰值采样训练过；修复从本次恢复后的第一步开始生效。
恢复模型、完整Adam和自适应学习率；仿真、地形等级、RNG、历史和滤波缓存重新初始化。

同课程续训使用`--resume`，必须显式匹配原来的ramp-start/end，不能静默重置课程。
每次调用写入新的日志目录和可选的新SwanLab run，父检查点只读。
`--activate`只更新本机活跃任务指针；旧训练仍活着时拒绝启动，不隐式杀进程。
没有自动停训/重启监控。

## 记录与验证

`params/env.yaml`保存终点参考配置，不代表起始有效权重。
`params/agent.yaml`保存时间表；`initialization.json`保存初始有效权重、来源SHA256。
`RewardCurriculum/alpha`及`weight_*`记录每轮真实权重；每轮24个采样步共享同一组权重。
checkpoint的`infos.reward_curriculum_state`保存已完成轮次、下轮次、alpha与有效权重。
episode reset不重置课程；play不推进课程。奖励仍只由RewardManager乘一次控制dt。

保留20000/25000/30000/40000/50000等评估锚点。课程期间总reward不可直接跨阶段
比较；用相同条件的站立、速度、关节运动、足力和越障指标判断，不保证50000轮
等价历史79000轮，也不自动追加奖励或延长训练。

## 起身阶段50000到80000

完整入口先执行原行走课程，保存 `model_50000.pt` 后从索引 50001 开始起身阶段，
训练到索引 80000。`--iterations 80001` 是总更新数，新增起身更新恰好 30000 次。
各阶段用独立进程和日志，阶段成功退出且目标检查点存在后才交接，不自动重试。

```bash
# 从零跑完整课程
./elf3.sh scripts/train_elf3_recovery_curriculum.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 80001 --seed 42 \
  --run-name curriculum_recovery80k --swanlab-project DWAQ

# 已有原课程50000，直接接入恢复
./elf3.sh scripts/train_elf3_recovery_curriculum.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 80001 \
  --resume /path/to/curriculum/model_50000.pt \
  --run-name recovery_from50000 --swanlab-project DWAQ
```

`--dry-run` 打印阶段、绝对轮次、更新数和命令，不启动 Isaac。较早的课程检查点
使用同一 `--resume`，先补齐至 50000；init-v4 起身检查点则恢复起身阶段。

### 恢复配方

采用当前 init-v4 实验的最终训练配方，从 50001 就启用全部初始化覆盖。
历史实验是在 60000 后才补充仰躺初始化，因此新课程不逐轮复刻历史中间实验。

| 项目 | 数值／行为 |
| --- | --- |
| 默认环境数 | 16384；6553 个固定恢复环境、9831 个原地形行走环境 |
| 恢复地形／指令 | 独立平地区域，零 vx/vy/wz，关闭该40%环境的周期推力 |
| 行走地形／指令 | 其余60%保留原17级混合地形、命令采样、随机化和课程 |
| 恢复姿态 | 50%原Recovery片段；50%增强仰躺姿态库 |
| 增强姿态库 | 4096姿态：2048扰动参考仰躺＋2048独立左右腿／手臂组合 |
| 恢复初速度 | 75%严格零速；20%小速度；5%较大速度，独立于姿态采样 |
| 小速度逐轴限值 | 世界系根线速度±[0.4,0.4,0.3]m/s；角速度每轴±2rad/s；关节每轴±3rad/s |
| 较大速度逐轴限值 | 世界系根线速度±[0.8,0.8,0.5]m/s；角速度每轴±4rad/s；关节每轴±6rad/s |
| 行走初始化 | 原WalkandRun片段位置与速度 |
| AMP | 当前16个原始片段、按片段等概率采样；仅恢复转换使用判别器风格奖励 |
| 奖励混合 | 恢复样本为0.75×任务奖励＋0.25×AMP风格分数；风格分数系数0.1 |
| 起身退出 | 相对地面躯干高度≥0.84m立即恢复普通DWAQ奖励；不附加姿态／速度／接触等待条件 |
| 成功统计 | 独立检查高度、倾角<20°、双脚合计支撑>30N、水平速度<0.5m/s、角速度<1.5rad/s，持续1s |
| 摔倒终止 | 恢复环境沿用低高度／大倾角判定，连续250控制步后终止；timeout立即终止 |
| 控制／资产 | 原隐式PD、armature0.01、5ms物理／20ms策略步长、20Hz速度观测模型 |
| 网络／部署 | 同一个500输入、29输出Actor；原DWAQ＋CENet，确定性mean导出 |

恢复期间暂时屏蔽冲突奖励：零指令关节静止、腿腰及手臂默认姿态、速度追踪、
直立／接触／步态／足力／运动高度等当前27项；完整列表固定在
`recovery_v2_core.MUTED_TERMS`。起身高度奖励和AMP提供恢复目标，原动作平滑等
未列入屏蔽的正则项保留。PPO上肢默认姿态辅助损失只对恢复样本关闭，镜像损失保留。
上升到退出高度后恢复全部普通DWAQ奖励。没有新增运动中强制放倒的训练采样。

### 数据和续训

`data/elf3_amp_v4/`包含本次实际使用的1个Recovery和15个WalkandRun原始NPZ；
`data/elf3_recovery_init_v4/`包含姿态库及生成元数据。文件哈希和发布代码记录见
`docs/elf3_recovery_curriculum_v1.json`。这些数据从本地Amp_mjlab ELF3动作库迁入，
不加载该项目的Actor、Critic或AMP判别器权重。姿态库元数据中的旧绝对路径仅是
生成来源记录，运行时从当前仓库或显式`--motion-dir`／`--bank`读取。

50000交接精确恢复DWAQ策略、CENet、Critic、Adam和学习率；AMP判别器新建。
同阶段续训精确恢复上述状态以及AMP判别器、其Adam、归一化矩和回放池。
原init-v4检查点可迁移到不同目录；仅姿态库路径允许变化，数据／元数据／代码哈希、
物理配置和完整PPO配方仍须一致。物理状态、地形等级、RNG和历史缓存按原续训方式重置。

起身专用入口默认目标也是80000；若需从80000继续到更高编号：

```bash
./elf3.sh scripts/train_elf3_recovery_stage.py \
  --headless --device cuda:0 --num-envs 16384 --iterations 100001 \
  --resume /path/to/recovery/model_80000.pt \
  --run-name recovery_resume80k --swanlab-project DWAQ
```

初次从50000进入专用入口使用`--parent-checkpoint`及其`--parent-sha256`；
完整课程入口会计算和记录该SHA256，不要求手动填写。历史flat-v3的60000检查点也可
显式迁移；普通G1、其他恢复配方或AMP策略不能当作本阶段的resume。

每次调用创建新日志和可选的SwanLab run（默认示例项目`DWAQ`）；完整阶段顺序、
子进程退出状态、父模型与最终检查点SHA256记在`logs/elf3_dwaq_recovery_curriculum_v1/*/course.json`。
导出继续使用`export_elf3_dwaq.py`，MuJoCo起身回放使用`play_elf3_recovery_mujoco.py`。
