# ELF3 still_v2 奖励课程 v1

新任务 `elf3_dwaq_still_v2_curriculum_v1`。不修改历史任务和权重，不改变资产、
隐式PD、armature=0.01、20 Hz关节速度观测模型、Actor/CENet、PPO、命令采样或地形。

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
