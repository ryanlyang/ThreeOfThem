from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for evaluation. Install it in your venv.") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from ppo_agent import ActorCritic, RunningMeanStd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--policy", type=str, default="deterministic", choices=["deterministic", "stochastic"])
    p.add_argument("--compare-baselines", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(rows: list[dict]) -> dict:
    return {
        "return_mean": float(np.mean([r["return"] for r in rows])),
        "return_std": float(np.std([r["return"] for r in rows])),
        "length_mean": float(np.mean([r["length"] for r in rows])),
        "collision_rate": float(np.mean([float(r["collided"]) for r in rows])),
        "final_pos_err_mean": float(np.mean([r["final_pos_err"] for r in rows])),
        "final_vel_err_mean": float(np.mean([r["final_vel_err"] for r in rows])),
    }


def run_policy(
    model: ActorCritic,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    rw: RewardWeights,
    episodes: int,
    seed: int,
    device: torch.device,
    policy_mode: str,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []

    model.eval()
    for _ in range(episodes):
        env = Figure8ChoreographyEnv(config=cfg, weights=rw)
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))

        done = False
        ep_ret = 0.0
        ep_len = 0
        info = {}

        while not done:
            obs_n = obs_rms.normalize(obs[None, :])
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, std, _ = model(obs_t)
                mean = mean * cfg.max_action_norm
                std = std * cfg.max_action_norm

                if policy_mode == "stochastic":
                    dist = torch.distributions.Normal(mean, std)
                    act = dist.sample().squeeze(0)
                else:
                    act = mean.squeeze(0)

            action = act.cpu().numpy().reshape(env.action_shape)
            action = np.clip(action, -cfg.max_action_norm, cfg.max_action_norm)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

        rows.append(
            {
                "return": ep_ret,
                "length": ep_len,
                "collided": bool(info.get("collided", False)),
                "final_pos_err": float(info.get("position_error", np.nan)),
                "final_vel_err": float(info.get("velocity_direction_error", np.nan)),
            }
        )

    return rows


def run_baseline(cfg: EnvConfig, rw: RewardWeights, episodes: int, seed: int, mode: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(episodes):
        env = Figure8ChoreographyEnv(config=cfg, weights=rw)
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))

        done = False
        ep_ret = 0.0
        ep_len = 0
        info = {}

        while not done:
            if mode == "zero":
                action = np.zeros(env.action_shape, dtype=np.float64)
            elif mode == "random":
                action = rng.uniform(-cfg.max_action_norm, cfg.max_action_norm, size=env.action_shape)
            else:
                raise ValueError(mode)

            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1

        rows.append(
            {
                "return": ep_ret,
                "length": ep_len,
                "collided": bool(info.get("collided", False)),
                "final_pos_err": float(info.get("position_error", np.nan)),
                "final_vel_err": float(info.get("velocity_direction_error", np.nan)),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt_path = Path(args.checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions do not expose weights_only.
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg = EnvConfig(**ckpt["env_config"])
    rw = RewardWeights(**ckpt["reward_weights"])

    obs_dim = cfg.num_bodies * cfg.dimensions * 2 + cfg.num_bodies + 2
    action_dim = cfg.num_bodies * cfg.dimensions
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    model.load_state_dict(ckpt["model_state"])

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.load_state_dict(ckpt["obs_rms"])

    rows = run_policy(
        model=model,
        obs_rms=obs_rms,
        cfg=cfg,
        rw=rw,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
        policy_mode=args.policy,
    )
    s = summarize(rows)

    print("=== Trained Policy ===")
    for k, v in s.items():
        print(f"{k}: {v}")

    if args.compare_baselines:
        print("\n=== Zero Baseline ===")
        s0 = summarize(run_baseline(cfg, rw, args.episodes, args.seed + 999, mode="zero"))
        for k, v in s0.items():
            print(f"{k}: {v}")

        print("\n=== Random Baseline ===")
        sr = summarize(run_baseline(cfg, rw, args.episodes, args.seed + 1999, mode="random"))
        for k, v in sr.items():
            print(f"{k}: {v}")

    out_json = ckpt_path.parent / "eval_summary.json"
    with out_json.open("w") as f:
        payload = {"trained": s}
        if args.compare_baselines:
            payload["zero"] = s0
            payload["random"] = sr
        json.dump(payload, f, indent=2)
    print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()
