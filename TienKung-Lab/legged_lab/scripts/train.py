# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, DWAQOnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--swanlab_project", default=None, help="Mirror TensorBoard to this SwanLab project.")
parser.add_argument("--swanlab_experiment_name", default=None)
parser.add_argument("--swanlab_id", default=None)
parser.add_argument("--swanlab_resume", choices=("never", "allow", "must"), default="never")
parser.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="Load policy weights and iteration from a checkpoint, but initialize a fresh optimizer.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering for tasks that require RGB/depth sensing
if args_cli.task and ("sensor" in args_cli.task or "rgb" in args_cli.task or "depth" in args_cli.task):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
import json
from pathlib import Path
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.swanlab_project and agent_cfg.logger.lower() != "tensorboard":
        raise ValueError("SwanLab mirroring requires --logger=tensorboard")
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer_on_resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    swanlab_run = None
    try:
        if args_cli.swanlab_project and int(os.environ.get("RANK", "0")) == 0:
            import swanlab

            swanlab_run = swanlab.init(
                project=args_cli.swanlab_project,
                name=args_cli.swanlab_experiment_name or Path(log_dir).name,
                id=args_cli.swanlab_id, resume=args_cli.swanlab_resume,
                mode="online", log_dir=str(Path(log_dir) / "swanlab"),
                config={"task": args_cli.task, "num_envs": env_cfg.scene.num_envs,
                        "seed": agent_cfg.seed, "max_iterations": agent_cfg.max_iterations},
            )
            # Same integration as Amp_mjlab, before learn() creates SummaryWriter.
            swanlab.sync_tensorboard_torch()
            (Path(log_dir) / "swanlab_run.json").write_text(json.dumps({
                "id": swanlab_run.id, "url": swanlab_run.url,
                "project": args_cli.swanlab_project, "mode": "native",
            }, indent=2) + "\n")
            print(f"[INFO] SwanLab experiment: {swanlab_run.url}")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        if runner.writer is not None:
            runner.writer.close()
        if swanlab_run is not None:
            swanlab_run.finish()
        if isinstance(env, Elf3DwaqEnv):
            env.close()


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
