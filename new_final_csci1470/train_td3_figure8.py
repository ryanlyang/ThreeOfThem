from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for TD3 training. Install requirements-train.txt.") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from fixed_init_profiles import resolve_fixed_init
from ppo_agent import RunningMeanStd
from td3_agent import ReplayBuffer, TD3Actor, TD3Critic, hard_update, soft_update
from vec_env import SerialVecEnv, SubprocVecEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TD3 on Figure-8 choreography env (fixed-init capable).")

    # Training loop.
    p.add_argument("--total-env-steps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["sync", "subproc"])
    p.add_argument("--mp-start-method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])

    # TD3 hyperparameters.
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--updates-per-iter", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--target-policy-noise", type=float, default=0.20)
    p.add_argument("--target-noise-clip", type=float, default=0.50)
    p.add_argument("--hidden-size", type=int, default=256)

    # Exploration.
    p.add_argument("--exploration-noise", type=float, default=0.30)
    p.add_argument("--exploration-noise-final", type=float, default=0.05)
    p.add_argument("--exploration-decay-steps", type=int, default=500_000)

    # Normalization / reward scaling.
    p.add_argument("--reward-scale", type=float, default=800.0)
    p.add_argument("--reward-clip", type=float, default=20.0)
    p.add_argument("--obs-clip", type=float, default=10.0)

    # Env config overrides.
    p.add_argument("--horizon-steps", type=int, default=500)
    p.add_argument("--action-dt", type=float, default=0.05)
    p.add_argument("--integrator-dt", type=float, default=0.001)
    p.add_argument("--phase-search-radius", type=int, default=35)
    p.add_argument("--max-action-norm", type=float, default=1.8)
    p.add_argument("--near-collision-distance", type=float, default=0.35)
    p.add_argument("--escape-radius", type=float, default=8.0)
    p.add_argument("--init-min-pair-distance", type=float, default=0.25)
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "amuse"])

    p.add_argument(
        "--fixed-init-profile",
        type=str,
        default="offset_ref",
        choices=["none", "weird", "near_ref", "offset_ref", "offset_ref_far"],
    )
    p.add_argument("--fixed-init-positions", type=str, default="")
    p.add_argument("--fixed-init-velocities", type=str, default="")
    p.add_argument("--fixed-init-pos-jitter-std", type=float, default=0.006)
    p.add_argument("--fixed-init-vel-jitter-std", type=float, default=0.004)
    p.add_argument("--fixed-init-jitter-tries", type=int, default=64)

    # Eval jitter override (defaults to training jitter if negative).
    p.add_argument("--eval-fixed-init-pos-jitter-std", type=float, default=-1.0)
    p.add_argument("--eval-fixed-init-vel-jitter-std", type=float, default=-1.0)
    p.add_argument("--eval-fixed-init-jitter-tries", type=int, default=64)

    # Reward weights.
    p.add_argument("--w-pos", type=float, default=20.0)
    p.add_argument("--w-vel", type=float, default=12.0)
    p.add_argument("--w-fuel", type=float, default=0.0005)
    p.add_argument("--w-near-collision", type=float, default=20.0)
    p.add_argument("--w-collision", type=float, default=500.0)
    p.add_argument("--w-escape", type=float, default=50.0)
    p.add_argument("--w-switch", type=float, default=0.08)
    p.add_argument("--w-phase", type=float, default=0.20)

    # Eval/checkpointing.
    p.add_argument("--eval-every-env-steps", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-strict-mode", action="store_true")
    p.add_argument("--eval-pos-threshold", type=float, default=0.06)
    p.add_argument("--eval-vel-threshold", type=float, default=0.09)
    p.add_argument("--eval-consecutive-converged", type=int, default=260)
    p.add_argument("--eval-min-total-steps", type=int, default=320)
    p.add_argument("--eval-lock-to-end", action="store_true")
    p.add_argument("--save-topk", type=int, default=3)
    p.add_argument("--early-stop-on-strict-success", action="store_true")
    p.add_argument("--early-stop-success-rate", type=float, default=1.0)
    p.add_argument("--early-stop-max-failure-rate", type=float, default=0.0)
    p.add_argument("--early-stop-patience-evals", type=int, default=1)
    p.add_argument("--early-stop-min-evals", type=int, default=1)

    # Runtime.
    p.add_argument("--log-every-env-steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="td3_figure8")
    p.add_argument("--save-dir", type=str, default="artifacts")

    return p.parse_args()


def make_env_config(args: argparse.Namespace) -> EnvConfig:
    fixed_pos, fixed_vel = resolve_fixed_init(
        profile=args.fixed_init_profile,
        positions_spec=args.fixed_init_positions,
        velocities_spec=args.fixed_init_velocities,
    )
    return EnvConfig(
        backend=args.backend,
        seed=args.seed,
        horizon_steps=args.horizon_steps,
        action_dt=args.action_dt,
        integrator_dt=args.integrator_dt,
        phase_search_radius=args.phase_search_radius,
        max_action_norm=args.max_action_norm,
        near_collision_distance=args.near_collision_distance,
        escape_radius=args.escape_radius,
        init_min_pair_distance=args.init_min_pair_distance,
        fixed_init_positions=fixed_pos,
        fixed_init_velocities=fixed_vel,
        fixed_init_pos_jitter_std=args.fixed_init_pos_jitter_std,
        fixed_init_vel_jitter_std=args.fixed_init_vel_jitter_std,
        fixed_init_jitter_tries=args.fixed_init_jitter_tries,
    )


def make_eval_env_config(args: argparse.Namespace, train_cfg: EnvConfig) -> EnvConfig:
    cfg = EnvConfig(**asdict(train_cfg))
    if args.eval_fixed_init_pos_jitter_std >= 0.0:
        cfg.fixed_init_pos_jitter_std = float(args.eval_fixed_init_pos_jitter_std)
    if args.eval_fixed_init_vel_jitter_std >= 0.0:
        cfg.fixed_init_vel_jitter_std = float(args.eval_fixed_init_vel_jitter_std)
    cfg.fixed_init_jitter_tries = int(args.eval_fixed_init_jitter_tries)
    return cfg


def make_reward_weights(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        position=args.w_pos,
        velocity_direction=args.w_vel,
        fuel=args.w_fuel,
        near_collision=args.w_near_collision,
        collision=args.w_collision,
        escape=args.w_escape,
        permutation_switch=args.w_switch,
        phase_jump=args.w_phase,
    )


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def progress_bar(step: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    frac = min(1.0, max(0.0, float(step) / float(total)))
    filled = int(round(width * frac))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def evaluate_actor(
    actor: TD3Actor,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    rw: RewardWeights,
    eval_episodes: int,
    rng: np.random.Generator,
    device: torch.device,
    obs_clip: float,
    strict_mode: bool,
    pos_threshold: float,
    vel_threshold: float,
    consecutive_converged: int,
    min_total_steps: int,
    lock_to_end: bool,
) -> dict[str, float]:
    returns = []
    lengths = []
    collisions = []
    escapes = []
    failures = []
    final_pos_err = []
    final_vel_err = []
    strict_success = []
    max_streaks = []
    end_streaks = []

    actor.eval()

    for _ in range(eval_episodes):
        env = Figure8ChoreographyEnv(config=cfg, weights=rw)
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        done = False
        ep_return = 0.0
        ep_len = 0
        info: dict[str, Any] = {}
        converge_streak = 0
        max_converge_streak = 0
        converged_step = None
        broke_after_lock = False

        while not done:
            obs_n = obs_rms.normalize(obs[None, :], clip=obs_clip)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_norm = actor(obs_t).squeeze(0).cpu().numpy()
            action = (action_norm * cfg.max_action_norm).reshape(env.action_shape)
            action = np.clip(action, -cfg.max_action_norm, cfg.max_action_norm)

            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            pos_err = float(info.get("position_error", np.nan))
            vel_err = float(info.get("velocity_direction_error", np.nan))
            step_collided = bool(info.get("collided", False))
            step_escaped = bool(info.get("escaped", False))

            meets = (
                np.isfinite(pos_err)
                and np.isfinite(vel_err)
                and (pos_err <= pos_threshold)
                and (vel_err <= vel_threshold)
                and (not step_collided)
                and (not step_escaped)
            )
            if meets:
                converge_streak += 1
                max_converge_streak = max(max_converge_streak, converge_streak)
            else:
                if converged_step is not None:
                    broke_after_lock = True
                converge_streak = 0

            if converged_step is None and converge_streak >= max(1, int(consecutive_converged)):
                converged_step = (ep_len + 1) - int(consecutive_converged) + 1

        returns.append(ep_return)
        lengths.append(ep_len)
        collided = bool(info.get("collided", False))
        escaped = bool(info.get("escaped", False))
        collisions.append(float(collided))
        escapes.append(float(escaped))
        failures.append(float(collided or escaped))
        f_pos = float(info.get("position_error", np.nan))
        f_vel = float(info.get("velocity_direction_error", np.nan))
        final_pos_err.append(f_pos)
        final_vel_err.append(f_vel)

        no_failure = not (collided or escaped)
        final_under_threshold = (
            np.isfinite(f_pos)
            and np.isfinite(f_vel)
            and (f_pos <= pos_threshold)
            and (f_vel <= vel_threshold)
        )
        enough_steps = ep_len >= max(0, int(min_total_steps))
        sustained_after_lock = (
            (converged_step is not None)
            and (not broke_after_lock)
            and (converge_streak >= max(1, int(consecutive_converged)))
        )
        basic_success = converged_step is not None
        if strict_mode:
            ep_strict = basic_success and no_failure and final_under_threshold and enough_steps
            if lock_to_end:
                ep_strict = ep_strict and sustained_after_lock
        else:
            ep_strict = basic_success

        strict_success.append(float(ep_strict))
        max_streaks.append(float(max_converge_streak))
        end_streaks.append(float(converge_streak))

    actor.train()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_escape_rate": float(np.mean(escapes)),
        "eval_failure_rate": float(np.mean(failures)),
        "eval_final_pos_err": float(np.nanmean(final_pos_err)),
        "eval_final_vel_err": float(np.nanmean(final_vel_err)),
        "eval_strict_success_rate": float(np.mean(strict_success)),
        "eval_max_converge_streak_mean": float(np.mean(max_streaks)),
        "eval_end_converge_streak_mean": float(np.mean(end_streaks)),
    }


def save_checkpoint(
    path: Path,
    *,
    actor: TD3Actor,
    actor_target: TD3Actor,
    critic: TD3Critic,
    critic_target: TD3Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    obs_rms: RunningMeanStd,
    env_cfg: EnvConfig,
    eval_env_cfg: EnvConfig,
    rw: RewardWeights,
    args: argparse.Namespace,
    global_step: int,
    update_step: int,
    best_eval_key: tuple[float, ...] | None,
    best_eval_strict: float,
    metrics_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "algo": "td3",
        "actor_state_dict": actor.state_dict(),
        "actor_target_state_dict": actor_target.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "critic_target_state_dict": critic_target.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "critic_optimizer": critic_opt.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "env_config": asdict(env_cfg),
        "eval_env_config": asdict(eval_env_cfg),
        "reward_weights": asdict(rw),
        "args": vars(args),
        "global_env_steps": int(global_step),
        "update_step": int(update_step),
        "best_eval_key": list(best_eval_key) if best_eval_key is not None else None,
        "best_eval_strict": float(best_eval_strict),
        "metrics_rows": metrics_rows,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    cfg = make_env_config(args)
    eval_cfg = make_eval_env_config(args, cfg)
    rw = make_reward_weights(args)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{run_stamp}"
    run_dir = Path(args.save_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train-td3] device={device}", flush=True)
    print(f"[train-td3] run_dir={run_dir}", flush=True)

    with (run_dir / "train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "env_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with (run_dir / "eval_env_config.json").open("w") as f:
        json.dump(asdict(eval_cfg), f, indent=2)
    with (run_dir / "reward_weights.json").open("w") as f:
        json.dump(asdict(rw), f, indent=2)

    if args.vec_env == "subproc":
        vec_env = SubprocVecEnv(cfg, rw, args.num_envs, args.seed, start_method=args.mp_start_method)
    else:
        vec_env = SerialVecEnv(cfg, rw, args.num_envs, args.seed)

    def _cleanup() -> None:
        try:
            vec_env.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    obs, _ = vec_env.reset()
    obs_dim = int(obs.shape[-1])
    action_shape = (3, cfg.dimensions)
    action_dim = int(np.prod(action_shape))

    actor = TD3Actor(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    actor_target = TD3Actor(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    critic = TD3Critic(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    critic_target = TD3Critic(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    hard_update(actor_target, actor)
    hard_update(critic_target, critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.update(obs)

    replay = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=args.buffer_size)

    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    train_returns_window: list[float] = []
    train_lengths_window: list[float] = []

    global_step = 0
    update_step = 0
    next_eval = int(args.eval_every_env_steps)
    best_eval_key: tuple[float, ...] | None = None
    best_eval_strict = -np.inf
    best_eval_count = 0
    early_stop_hit_count = 0
    metrics_rows: list[dict[str, Any]] = []

    metrics_csv = run_dir / "metrics.csv"
    ckpt_latest = run_dir / "checkpoint_latest.pt"
    ckpt_best = run_dir / "checkpoint_best.pt"

    topk_paths: list[tuple[tuple[float, ...], Path]] = []

    start_time = time.perf_counter()
    last_log_time = start_time

    completed = False
    stopped_early = False

    try:
        while global_step < int(args.total_env_steps):
            decay_t = min(1.0, float(global_step) / float(max(1, int(args.exploration_decay_steps))))
            expl_std = (1.0 - decay_t) * float(args.exploration_noise) + decay_t * float(args.exploration_noise_final)

            if global_step < int(args.learning_starts):
                action_norm = np_rng.uniform(-1.0, 1.0, size=(args.num_envs, action_dim)).astype(np.float32)
            else:
                obs_n = obs_rms.normalize(obs, clip=args.obs_clip)
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_norm = actor(obs_t).cpu().numpy()
                noise = np_rng.normal(0.0, expl_std, size=action_norm.shape)
                action_norm = np.clip(action_norm + noise, -1.0, 1.0).astype(np.float32)

            action_env = (action_norm.reshape(args.num_envs, *action_shape) * cfg.max_action_norm).astype(np.float64)
            next_obs, rewards_raw, dones, _ = vec_env.step(action_env)

            rewards_scaled = np.clip(rewards_raw.astype(np.float64) / float(args.reward_scale), -args.reward_clip, args.reward_clip)

            for i in range(args.num_envs):
                replay.add(
                    obs=obs[i],
                    action=action_norm[i],
                    reward=float(rewards_scaled[i]),
                    next_obs=next_obs[i],
                    done=float(dones[i]),
                )

            ep_returns += rewards_raw.astype(np.float64)
            ep_lengths += 1
            done_idx = np.where(dones > 0.5)[0]
            if done_idx.size > 0:
                for j in done_idx:
                    train_returns_window.append(float(ep_returns[j]))
                    train_lengths_window.append(float(ep_lengths[j]))
                    ep_returns[j] = 0.0
                    ep_lengths[j] = 0
                if len(train_returns_window) > 200:
                    train_returns_window = train_returns_window[-200:]
                if len(train_lengths_window) > 200:
                    train_lengths_window = train_lengths_window[-200:]

            global_step += args.num_envs
            obs = next_obs
            obs_rms.update(obs)

            if global_step >= int(args.learning_starts) and replay.size >= int(args.batch_size):
                for _ in range(int(args.updates_per_iter)):
                    batch = replay.sample(int(args.batch_size), np_rng)

                    obs_b = torch.as_tensor(obs_rms.normalize(batch.obs, clip=args.obs_clip), dtype=torch.float32, device=device)
                    next_obs_b = torch.as_tensor(obs_rms.normalize(batch.next_obs, clip=args.obs_clip), dtype=torch.float32, device=device)
                    actions_b = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
                    rewards_b = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
                    dones_b = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        noise = torch.randn_like(actions_b) * float(args.target_policy_noise)
                        noise = torch.clamp(noise, -float(args.target_noise_clip), float(args.target_noise_clip))
                        next_actions = torch.clamp(actor_target(next_obs_b) + noise, -1.0, 1.0)

                        q1_t, q2_t = critic_target(next_obs_b, next_actions)
                        q_t = torch.minimum(q1_t, q2_t)
                        target_q = rewards_b + float(args.gamma) * (1.0 - dones_b) * q_t

                    q1, q2 = critic(obs_b, actions_b)
                    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                    critic_opt.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                    critic_opt.step()

                    update_step += 1

                    if update_step % int(args.policy_delay) == 0:
                        actor_actions = actor(obs_b)
                        actor_loss = -critic.q1_only(obs_b, actor_actions).mean()

                        actor_opt.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
                        actor_opt.step()

                        soft_update(actor_target, actor, float(args.tau))
                        soft_update(critic_target, critic, float(args.tau))

            if global_step >= next_eval:
                eval_rng = np.random.default_rng(args.seed + 100_000 + best_eval_count)
                eval_out = evaluate_actor(
                    actor=actor,
                    obs_rms=obs_rms,
                    cfg=eval_cfg,
                    rw=rw,
                    eval_episodes=int(args.eval_episodes),
                    rng=eval_rng,
                    device=device,
                    obs_clip=float(args.obs_clip),
                    strict_mode=bool(args.eval_strict_mode),
                    pos_threshold=float(args.eval_pos_threshold),
                    vel_threshold=float(args.eval_vel_threshold),
                    consecutive_converged=int(args.eval_consecutive_converged),
                    min_total_steps=int(args.eval_min_total_steps),
                    lock_to_end=bool(args.eval_lock_to_end),
                )
                best_eval_count += 1

                eval_key = (
                    float(eval_out["eval_strict_success_rate"]),
                    -float(eval_out["eval_failure_rate"]),
                    -float(eval_out["eval_collision_rate"]),
                    -float(eval_out["eval_final_pos_err"]),
                    -float(eval_out["eval_final_vel_err"]),
                    float(eval_out["eval_return_mean"]),
                )

                row = {
                    "env_step": int(global_step),
                    "update_step": int(update_step),
                    "train_return_mean": float(np.mean(train_returns_window)) if train_returns_window else float("nan"),
                    "train_return_std": float(np.std(train_returns_window)) if train_returns_window else float("nan"),
                    "train_length_mean": float(np.mean(train_lengths_window)) if train_lengths_window else float("nan"),
                    **eval_out,
                }
                metrics_rows.append(row)
                write_metrics_csv(metrics_csv, metrics_rows)

                save_checkpoint(
                    ckpt_latest,
                    actor=actor,
                    actor_target=actor_target,
                    critic=critic,
                    critic_target=critic_target,
                    actor_opt=actor_opt,
                    critic_opt=critic_opt,
                    obs_rms=obs_rms,
                    env_cfg=cfg,
                    eval_env_cfg=eval_cfg,
                    rw=rw,
                    args=args,
                    global_step=global_step,
                    update_step=update_step,
                    best_eval_key=best_eval_key,
                    best_eval_strict=best_eval_strict,
                    metrics_rows=metrics_rows,
                )

                if (best_eval_key is None) or (eval_key > best_eval_key):
                    best_eval_key = eval_key
                    best_eval_strict = float(eval_out["eval_strict_success_rate"])
                    shutil.copy2(ckpt_latest, ckpt_best)

                if int(args.save_topk) > 0:
                    ckpt_eval = run_dir / f"checkpoint_evalstep_{global_step:09d}.pt"
                    shutil.copy2(ckpt_latest, ckpt_eval)
                    topk_paths.append((eval_key, ckpt_eval))
                    topk_paths.sort(key=lambda x: x[0], reverse=True)
                    if len(topk_paths) > int(args.save_topk):
                        _, worst_path = topk_paths.pop(-1)
                        if worst_path.exists():
                            worst_path.unlink()

                elapsed = time.perf_counter() - start_time
                eta = (max(1, args.total_env_steps - global_step) / max(1.0, global_step)) * elapsed
                print(
                    f"[eval step={global_step}] return_mean={eval_out['eval_return_mean']:.3f} "
                    f"failure_rate={eval_out['eval_failure_rate']:.3f} "
                    f"strict_success_rate={eval_out['eval_strict_success_rate']:.3f} "
                    f"final_pos_err={eval_out['eval_final_pos_err']:.5f} "
                    f"final_vel_err={eval_out['eval_final_vel_err']:.5f}",
                    flush=True,
                )
                print(
                    f"{progress_bar(global_step, args.total_env_steps)} {100.0 * global_step / max(1, args.total_env_steps):6.2f}% "
                    f"step={global_step}/{args.total_env_steps} elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"train_return_mean={row['train_return_mean']:.3f} eval_strict_success={eval_out['eval_strict_success_rate']:.3f}",
                    flush=True,
                )

                if args.early_stop_on_strict_success:
                    # Require enough evals before early stop.
                    if best_eval_count >= int(args.early_stop_min_evals):
                        meets = (
                            eval_out["eval_strict_success_rate"] >= float(args.early_stop_success_rate)
                            and eval_out["eval_failure_rate"] <= float(args.early_stop_max_failure_rate)
                        )
                        if meets:
                            early_stop_hit_count += 1
                        else:
                            early_stop_hit_count = 0

                        if early_stop_hit_count >= max(1, int(args.early_stop_patience_evals)):
                            print(
                                f"[early-stop] strict success criterion met for {early_stop_hit_count} eval(s)",
                                flush=True,
                            )
                            stopped_early = True
                            completed = True
                            break

                next_eval += int(args.eval_every_env_steps)

            if global_step % int(args.log_every_env_steps) < args.num_envs:
                now = time.perf_counter()
                elapsed = now - start_time
                dt = now - last_log_time
                last_log_time = now
                sps = float(args.log_every_env_steps) / max(1e-6, dt)
                eta = (max(1, args.total_env_steps - global_step) / max(1.0, global_step)) * elapsed
                tr_mean = float(np.mean(train_returns_window)) if train_returns_window else float("nan")
                print(
                    f"{progress_bar(global_step, args.total_env_steps)} {100.0 * global_step / max(1, args.total_env_steps):6.2f}% "
                    f"step={global_step}/{args.total_env_steps} elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"sps~{sps:,.0f} replay={replay.size} train_return_mean={tr_mean:.3f}",
                    flush=True,
                )

        completed = True

    finally:
        _cleanup()

    # Final checkpoint and summary.
    save_checkpoint(
        ckpt_latest,
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        obs_rms=obs_rms,
        env_cfg=cfg,
        eval_env_cfg=eval_cfg,
        rw=rw,
        args=args,
        global_step=global_step,
        update_step=update_step,
        best_eval_key=best_eval_key,
        best_eval_strict=best_eval_strict,
        metrics_rows=metrics_rows,
    )
    if not ckpt_best.exists():
        shutil.copy2(ckpt_latest, ckpt_best)

    status = {
        "completed": bool(completed),
        "stopped_early": bool(stopped_early),
        "global_env_steps": int(global_step),
        "update_steps": int(update_step),
        "best_eval_key": list(best_eval_key) if best_eval_key is not None else None,
        "best_eval_strict": float(best_eval_strict),
        "run_dir": str(run_dir),
    }
    with (run_dir / "run_status.json").open("w") as f:
        json.dump(status, f, indent=2)

    print(f"[done] completed_steps={global_step}/{args.total_env_steps}", flush=True)
    print(f"[done] stopped_early={stopped_early}", flush=True)
    print(f"[done] best_eval_key={best_eval_key}", flush=True)
    print(f"[done] artifacts at: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
