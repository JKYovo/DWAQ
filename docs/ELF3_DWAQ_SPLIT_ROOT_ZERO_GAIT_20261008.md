# ELF3 DWAQ 本地存档：分离姿态约束与零指令踏步开关

基于 GitHub JKYovo/DWAQ 的 de9c841。本存档记录截至 2026-10-08 的已训练版本。

- 新增任务 elf3_dwaq_upper_symmetry_pose_delay_split_root。
- flat_orientation_l2 改为约束骨盆 waist_z_link，权重 -1；胸部 torso_link 继续使用 body_orientation_l2，权重 -2。
- vx、vy、wz 的绝对值均不超过 1e-6 时关闭 gait_phase_contact 和 feet_swing_height；非零指令与原地转向保留原计算。相位观测继续运行。
- 保留原有 0/20/40 ms 动作延迟、上身镜像和软姿态损失、地形、速度奖励、命令采样、PD、学习率和导出接口。

训练历程：分离姿态约束从头训练，在 model_21500.pt 增加零指令开关后续训；最后停止于 model_333600.pt。最近导出为 policy_319800.onnx（history [batch,500] → actions [batch,29]）。

检查点所在目录（权重不提交 Git）：
logs/elf3_dwaq_upper_symmetry_pose/2026-09-24_18-04-11_zero_gait_resume21500_target400000/

本机已有只读指令响应评估：logs/evaluations/20261008_response/REPORT.md。
