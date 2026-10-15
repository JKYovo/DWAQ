# ELF3 0918 发布配置核验（2026-10-15）

## 变更边界

- 训练逻辑恢复到 `2362a3ade4f12846ee26c2da0d33a6634cad5349`；为了保持 Melody 风格资产目录，少数 ELF3 集成文件只调整元数据导入路径及增加参数一致性校验，不修改训练参数。
- 使用完整的 `elf3_dwaq_upper_symmetry_pose` 注册任务、Agent 和 `Elf3DwaqUpperBodySymmetryRunner`。
- 保留新 `assets/elf3/elf3.py` 及原有 43.22 kg 物理文件；具名隐式执行器、显式 armature 0.01、速度上限 20 rad/s。
- `assets/elf3/` 仅保留配置与物理文件；契约在 `legged_lab/configs/elf3/contract.json`，来源记录在 `docs/elf3_asset_provenance.json`，上游配置参考在 `scripts/references/melody_elf3.py`，检查工具在 `scripts/check_elf3_asset.py`。
- 契约内容与语义哈希保持不变，当前 0918 checkpoint 无需转换；运行时校验迁至 DWAQ 配置构建阶段。
- 网格集中至 `assets/elf3/meshes/`，MJCF 集中至 `assets/elf3/mjcf/elf3.xml`；URDF/MJCF 的几何引用均为 `../meshes/`，不保留旧 `xmls/` 或兼容符号链接。
- USD 和 34 个 STL 字节保持不变；URDF/MJCF 仅更改网格路径。原 checkpoint 契约的来源哈希保留，检查器通过严格反向替换这两种路径后核验，其他内容变化仍会报错。
- 力矩限制、Kp/Kd、默认姿态和动作尺度与当前长训一致，不重新计算动作尺度。
- 启动脚本将本机绝对路径、固定 Git HEAD 和历史训练日志依赖改为本 checkout 相对路径及固定源码哈希清单。
- 保留现有环境安装器、部署和导出脚本；更新两份 README，标明延迟／split-root 文档仅为历史存档。
- 旧 blind-x、command-response、quality-v3、pose-scratch 实验代码已归档，不再注册或参与训练导入。
  日志传输工具继续保留；其历史文件名不代表启用旧奖励。
- 配置整理及发布不停止或重启正在运行的正式训练；Git 提交和推送仅在用户明确要求后执行。

## 已完成检查

1. 与正式长训保存的 `params/env.yaml`、`params/agent.yaml` 全量比较：
   除 USD 所在项目路径及 run name 外，无差异。没有读取其权重作为训练起点。
2. 核实保存 YAML 与实际 PPO：镜像系数 0.1、姿态系数 0.02，初始学习率 1e-4、adaptive 范围 1e-5～3e-4。
3. 独立 64 环境 × 20 轮从零短测完成：权重／优化器／标量有限，两个辅助损失参与训练，编号 0～19 完整。
   测试模型未用于正式训练。
4. 独立 4 环境几何、关节映射、默认姿态 FK、隐式 PD 参数、2 轮 PPO、checkpoint 往返和错误契约拒绝检查通过。
5. 当前 0918 的 `model_47500.pt` 可由本仓库导出器只读加载，500→29 CPU 推理输出有限。
6. 9 项 CPU 单元测试、源码／物理文件哈希检查及 `git diff --check` 通过。
7. 资产目录迁移后重新完成 64 环境 × 20 轮短测；新 `scripts/check_elf3_asset.py` 完成 4 环境 × 24 物理步检查，全部 29 关节的 PhysX 参数、模型质量和原始配置比较通过。
8. 迁移后与当前长训保存的环境／Agent 配置再次全量比较，无训练参数差异；已有 47500 checkpoint 确定性推理及 MuJoCo 5 个控制步接口检查通过。短回放仅验证路径与接口，不作为行走性能验收。
9. `meshes/`、`mjcf/` 迁移后，MuJoCo 新旧模型的 14 组几何／动力学数组完全相同；新路径的 4 环境几何／关节映射、2 轮 PPO、checkpoint 往返检查通过。

物理 USD SHA256：`80872360523c391ea413a00452c2042ed512451d280e749241ad2ca7c97c7c58`。
资产契约文件 SHA256：`90e4b49b10ff160002c09d5c735b51f8d46e9c710cd2e2c44257fe7edb320919`。

检查日志仍出现本机已有的 inotify watcher 数量耗尽警告及并行运行的缓存占用警告；
所有上述仿真检查最终正常完成、退出码为 0。没有通过修改系统设置宣称修复这些环境问题。
直接调用 Python 时系统 CUDA 库可能覆盖环境依赖，因此 PyTorch 检查／导出统一推荐使用 `elf3.sh`。
通用 USD 依赖扫描无法通过文件路径解析原模型的 `OmniPBR.mdl` 模块；检查器单独记录这一继承的材质模块，
不把它当成缺失网格。其他无法解析的依赖仍会导致检查失败；USD 与网格字节未改。

这些是配置一致性与技术稳定性检查，不是完整行走、越障或真机性能验收。

## 复核入口

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/release_0918_checks.py
./elf3.sh scripts/train_0918_melody.py --headless --verify-config-only
```

短测、正式训练和导出命令见根目录 README。正式入口始终从随机权重和全新优化器开始，
不能用它自动恢复旧版本训练。需要续训时另行核对配置和 checkpoint 契约。
