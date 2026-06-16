import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from trampoline_env import G1TrampolineEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-trampoline")
    parser.add_argument("--ckpt", type=int, default=3000)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--no-viewer", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env = G1TrampolineEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,
    )
    env.max_episode_length = float("inf")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    max_height = 0.0
    step = 0
    with torch.no_grad():
        while args.num_steps is None or step < args.num_steps:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            height = float((env.base_pos[0, 2] - env.base_height_ref).item())
            max_height = max(max_height, height)
            if env.episode_length_buf[0].item() % 100 == 0:
                print(f"max jump height: {max_height:.3f} m")
            step += 1


if __name__ == "__main__":
    main()

"""
python examples/trampoline/trampoline_eval.py -e g1-trampoline --ckpt 1200
"""
