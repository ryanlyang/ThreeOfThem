from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for TD3 evaluation. Install requirements-train.txt.") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from ppo_agent import RunningMeanStd
from td3_agent import TD3Actor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TD3 checkpoint on fixed-init Figure-8 env.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--obs-clip", type=float, default=10.0)
    p.add_argument("--fixed-init-pos-jitter-std", type=float, default=-1.0)
    p.add_argument("--fixed-init-vel-jitter-std", type=float, default=-1.0)
    p.add_argument("--fixed-init-jitter-tries", type=int, default=64)
    p.add_argument("--override-fixed-jitter-zero", action="store_true")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt_path = Path(args.checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if ckpt.get("algo") != "td3":
        raise SystemExit(f"Checkpoint is not TD3 (algo={ckpt.get('algo')!r}).")

    cfg = EnvConfig(**ckpt.get("eval_env_config", ckpt["env_config"]))
    rw = RewardWeights(**ckpt["reward_weights"])

    if args.override_fixed_jitter_zero:
        cfg.fixed_init_pos_jitter_std = 0.0
        cfg.fixed_init_vel_jitter_std = 0.0
        cfg.fixed_init_jitter_tries = 1
    else:
        if args.fixed_init_pos_jitter_std >= 0.0:
            cfg.fixed_init_pos_jitter_std = float(args.fixed_init_pos_jitter_std)
        if args.fixed_init_vel_jitter_std >= 0.0:
            cfg.fixed_init_vel_jitter_std = float(args.fixed_init_vel_jitter_std)
        cfg.fixed_init_jitter_tries = int(args.fixed_init_jitter_tries)

    obs_dim = cfg.num_bodies * cfg.dimensions * 2 + cfg.num_bodies + 2
    action_dim = cfg.num_bodies * cfg.dimensions

    actor = TD3Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.load_state_dict(ckpt["obs_rms"])

    rng = np.random.default_rng(args.seed)

    returns = []
    lengths = []
    collisions = []
    escapes = []
    pos_errs = []
    vel_errs = []

    for _ in range(int(args.episodes)):
        env = Figure8ChoreographyEnv(config=cfg, weights=rw)
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        done = False
        ep_return = 0.0
        ep_len = 0
        info = {}

        while not done:
            obs_n = obs_rms.normalize(obs[None, :], clip=args.obs_clip)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_norm = actor(obs_t).squeeze(0).cpu().numpy()
            action = (action_norm * cfg.max_action_norm).reshape((cfg.num_bodies, cfg.dimensions))
            action = np.clip(action, -cfg.max_action_norm, cfg.max_action_norm)

            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)
        collisions.append(float(bool(info.get("collided", False))))
        escapes.append(float(bool(info.get("escaped", False))))
        pos_errs.append(float(info.get("position_error", np.nan)))
        vel_errs.append(float(info.get("velocity_direction_error", np.nan)))

    summary = {
        "algo": "td3",
        "episodes": int(args.episodes),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
        "escape_rate": float(np.mean(escapes)),
        "failure_rate": float(np.mean(np.asarray(collisions) + np.asarray(escapes) > 0.0)),
        "final_pos_err_mean": float(np.nanmean(pos_errs)),
        "final_vel_err_mean": float(np.nanmean(vel_errs)),
    }

    print(json.dumps(summary, indent=2), flush=True)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ckpt_path.parent / "td3_eval_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
