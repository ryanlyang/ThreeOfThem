from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Quick PPO training on one fixed initial condition (proof-of-concept).")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--run-name", type=str, default="fixedinit_quick")
    p.add_argument("--save-dir", type=str, default="artifacts")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--fixed-init-profile", type=str, default="weird", choices=["none", "weird"])
    p.add_argument("--fixed-init-positions", type=str, default="")
    p.add_argument("--fixed-init-velocities", type=str, default="")
    p.add_argument("--horizon-steps", type=int, default=260)
    p.add_argument("--ent-coef", type=float, default=0.0015)
    p.add_argument("--reward-scale", type=float, default=800.0)
    p.add_argument("--max-action-norm", type=float, default=0.22)
    p.add_argument("--escape-radius", type=float, default=4.0)
    p.add_argument("--init-min-pair-distance", type=float, default=0.25)
    p.add_argument("--w-collision", type=float, default=100.0)
    p.add_argument("--w-escape", type=float, default=4.0)
    return p.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    root = Path(__file__).resolve().parent

    cmd = [
        sys.executable,
        str(root / "train_ppo_figure8.py"),
        "--updates",
        str(args.updates),
        "--num-envs",
        str(args.num_envs),
        "--rollout-steps",
        str(args.rollout_steps),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.seed),
        "--run-name",
        args.run_name,
        "--save-dir",
        args.save_dir,
        "--device",
        args.device,
        "--fixed-init-profile",
        args.fixed_init_profile,
        "--fixed-init-positions",
        args.fixed_init_positions,
        "--fixed-init-velocities",
        args.fixed_init_velocities,
        "--horizon-steps",
        str(args.horizon_steps),
        "--ent-coef",
        str(args.ent_coef),
        "--reward-scale",
        str(args.reward_scale),
        "--max-action-norm",
        str(args.max_action_norm),
        "--escape-radius",
        str(args.escape_radius),
        "--init-min-pair-distance",
        str(args.init_min_pair_distance),
        "--w-collision",
        str(args.w_collision),
        "--w-escape",
        str(args.w_escape),
    ]
    cmd.extend(passthrough)

    print("Running command:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
