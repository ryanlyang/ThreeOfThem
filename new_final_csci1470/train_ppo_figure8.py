from __future__ import annotations

import atexit
import argparse
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for training. Install it in your venv (see requirements-train.txt).") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from fixed_init_profiles import resolve_fixed_init
from ppo_agent import ActorCritic, PPOBatch, RunningMeanStd, gaussian_entropy, gaussian_log_prob
from vec_env import SerialVecEnv, SubprocVecEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Figure-8 choreography env.")

    # Training loop.
    p.add_argument("--updates", type=int, default=120)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=8)
    p.add_argument("--minibatch-size", type=int, default=256)

    # Parallel env stepping.
    p.add_argument("--vec-env", type=str, default="subproc", choices=["sync", "subproc"])
    p.add_argument("--mp-start-method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])

    # PPO params.
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--initial-log-std", type=float, default=-1.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Stabilization.
    p.add_argument("--reward-scale", type=float, default=1000.0)
    p.add_argument("--reward-clip", type=float, default=20.0)
    p.add_argument("--obs-clip", type=float, default=10.0)

    # Env config overrides.
    p.add_argument("--horizon-steps", type=int, default=300)
    p.add_argument("--action-dt", type=float, default=0.05)
    p.add_argument("--integrator-dt", type=float, default=0.001)
    p.add_argument("--phase-search-radius", type=int, default=25)
    p.add_argument("--max-action-norm", type=float, default=0.30)
    p.add_argument("--near-collision-distance", type=float, default=0.35)
    p.add_argument("--escape-radius", type=float, default=4.0)
    p.add_argument("--init-min-pair-distance", type=float, default=0.25)
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "amuse"])
    p.add_argument("--fixed-init-profile", type=str, default="none", choices=["none", "weird", "near_ref", "offset_ref"])
    p.add_argument("--fixed-init-positions", type=str, default="")
    p.add_argument("--fixed-init-velocities", type=str, default="")
    p.add_argument("--fixed-init-pos-jitter-std", type=float, default=0.0)
    p.add_argument("--fixed-init-vel-jitter-std", type=float, default=0.0)
    p.add_argument("--fixed-init-jitter-tries", type=int, default=32)

    # Reward weights.
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-vel", type=float, default=0.35)
    p.add_argument("--w-fuel", type=float, default=0.03)
    p.add_argument("--w-near-collision", type=float, default=0.0)
    p.add_argument("--w-collision", type=float, default=60.0)
    p.add_argument("--w-escape", type=float, default=2.0)
    p.add_argument("--w-switch", type=float, default=0.15)
    p.add_argument("--w-phase", type=float, default=0.01)

    # Eval/checkpointing.
    p.add_argument("--eval-every", type=int, default=6)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--eval-strict-mode", action="store_true")
    p.add_argument("--eval-pos-threshold", type=float, default=0.08)
    p.add_argument("--eval-vel-threshold", type=float, default=0.12)
    p.add_argument("--eval-consecutive-converged", type=int, default=180)
    p.add_argument("--eval-min-total-steps", type=int, default=220)
    p.add_argument("--eval-lock-to-end", action="store_true")
    p.add_argument("--save-topk", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="ppo_figure8")
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


def evaluate_policy(
    model: ActorCritic,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    rw: RewardWeights,
    eval_episodes: int,
    rng: np.random.Generator,
    device: torch.device,
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

    model.eval()

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
            obs_n = obs_rms.normalize(obs[None, :])
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _, _ = model(obs_t)
            mean = mean * cfg.max_action_norm
            act = mean.squeeze(0).cpu().numpy().reshape(env.action_shape)
            act = np.clip(act, -cfg.max_action_norm, cfg.max_action_norm)

            obs, reward, done, info = env.step(act)
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

    model.train()

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


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def maybe_plot_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    updates = [r["update"] for r in rows]
    train_r = [r["train_return_mean"] for r in rows]
    eval_r = [r["eval_return_mean"] for r in rows]
    col = [r["eval_collision_rate"] for r in rows]

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(updates, train_r, color="navy")
    ax[0].set_ylabel("Train Return")
    ax[0].grid(alpha=0.25)

    ax[1].plot(updates, eval_r, color="darkgreen")
    ax[1].set_ylabel("Eval Return")
    ax[1].grid(alpha=0.25)

    ax[2].plot(updates, col, color="crimson")
    ax[2].set_ylabel("Eval Collision")
    ax[2].set_xlabel("Update")
    ax[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _update_topk_checkpoints(
    run_dir: Path,
    ckpt: dict[str, Any],
    eval_key: tuple[float, ...],
    update: int,
    topk_limit: int,
    topk_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    qualifies = (len(topk_entries) < topk_limit) or (tuple(eval_key) < tuple(topk_entries[-1]["eval_key"]))
    if not qualifies:
        return topk_entries, False

    cand_path = run_dir / f"checkpoint_candidate_update_{int(update):05d}.pt"
    torch.save(ckpt, cand_path)

    topk_entries.append(
        {
            "update": int(update),
            "eval_key": tuple(float(x) for x in eval_key),
            "candidate_path": str(cand_path),
        }
    )
    topk_entries.sort(key=lambda e: tuple(e["eval_key"]))

    while len(topk_entries) > topk_limit:
        drop = topk_entries.pop(-1)
        drop_path = Path(drop["candidate_path"])
        if drop_path.exists():
            drop_path.unlink()

    for rank in range(1, topk_limit + 1):
        rp = run_dir / f"checkpoint_best_rank{rank}.pt"
        if rp.exists():
            rp.unlink()

    manifest = []
    for rank, entry in enumerate(topk_entries, start=1):
        src = Path(entry["candidate_path"])
        dst = run_dir / f"checkpoint_best_rank{rank}.pt"
        shutil.copy2(src, dst)
        manifest.append(
            {
                "rank": rank,
                "update": int(entry["update"]),
                "eval_key": list(entry["eval_key"]),
                "path": str(dst),
            }
        )

    if len(manifest) > 0:
        shutil.copy2(run_dir / "checkpoint_best_rank1.pt", run_dir / "checkpoint_best.pt")
    with (run_dir / "checkpoint_topk_manifest.json").open("w") as f:
        json.dump({"topk": topk_limit, "entries": manifest}, f, indent=2)

    return topk_entries, True


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    rng = np.random.default_rng(args.seed)

    cfg = make_env_config(args)
    rw = make_reward_weights(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.save_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "train_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (run_dir / "env_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with (run_dir / "reward_weights.json").open("w") as f:
        json.dump(asdict(rw), f, indent=2)

    print(f"[train] device={device}", flush=True)
    print(f"[train] run_dir={run_dir}", flush=True)

    probe_env = Figure8ChoreographyEnv(config=cfg, weights=rw)
    obs_dim = probe_env.observation_dim
    action_shape = probe_env.action_shape
    action_dim = int(np.prod(action_shape))
    max_action = cfg.max_action_norm
    del probe_env

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, initial_log_std=args.initial_log_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.vec_env == "subproc":
        envs: SubprocVecEnv | SerialVecEnv = SubprocVecEnv(
            cfg,
            rw,
            args.num_envs,
            seed=args.seed,
            start_method=args.mp_start_method,
        )
    else:
        envs = SerialVecEnv(cfg, rw, args.num_envs, seed=args.seed)

    reset_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(args.num_envs)]
    obs, _ = envs.reset(seeds=reset_seeds)
    atexit.register(envs.close)
    obs_rms.update(obs)

    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []

    best_eval = -np.inf
    best_eval_key: tuple[float, ...] | None = None
    topk_limit = max(1, int(args.save_topk))
    topk_entries: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for update in range(1, args.updates + 1):
        # Rollout buffers [T, N, ...]
        obs_buf = np.zeros((args.rollout_steps, args.num_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((args.rollout_steps, args.num_envs, action_dim), dtype=np.float32)
        logp_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        rew_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        done_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        val_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)

        for t in range(args.rollout_steps):
            obs_rms.update(obs)
            obs_n = obs_rms.normalize(obs, clip=args.obs_clip)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)

            with torch.no_grad():
                mean, std, value = model(obs_t)
                # Keep policy distribution in physical action units.
                mean = mean * max_action
                std = std * max_action
                dist = torch.distributions.Normal(mean, std)
                action_t = dist.sample()
                action_t = torch.clamp(action_t, -max_action, max_action)
                logp_t = dist.log_prob(action_t).sum(dim=-1)

            action_np = action_t.cpu().numpy()
            action_np = np.clip(action_np, -max_action, max_action)

            obs_buf[t] = obs
            act_buf[t] = action_np
            logp_buf[t] = logp_t.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            actions_env = action_np.reshape(args.num_envs, *action_shape)
            next_obs, reward_raw, done_arr, _ = envs.step(actions_env)

            rew_buf[t] = np.clip(reward_raw / args.reward_scale, -args.reward_clip, args.reward_clip)
            done_buf[t] = done_arr

            for i in range(args.num_envs):
                ep_returns[i] += float(reward_raw[i])
                ep_lengths[i] += 1
                if bool(done_arr[i]):
                    finished_returns.append(float(ep_returns[i]))
                    finished_lengths.append(int(ep_lengths[i]))
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            obs = next_obs

        # Bootstrap value for final obs.
        obs_n = obs_rms.normalize(obs, clip=args.obs_clip)
        with torch.no_grad():
            _, _, next_values_t = model(torch.as_tensor(obs_n, dtype=torch.float32, device=device))
        next_values = next_values_t.cpu().numpy()

        adv_buf = np.zeros_like(rew_buf)
        gae = np.zeros(args.num_envs, dtype=np.float32)

        for t in reversed(range(args.rollout_steps)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_values * not_done - val_buf[t]
            gae = delta + args.gamma * args.gae_lambda * not_done * gae
            adv_buf[t] = gae
            next_values = val_buf[t]

        ret_buf = adv_buf + val_buf

        # Flatten to [B, ...]
        B = args.rollout_steps * args.num_envs
        batch_obs = obs_buf.reshape(B, obs_dim)
        batch_obs_n = obs_rms.normalize(batch_obs, clip=args.obs_clip)

        batch = PPOBatch(
            obs=torch.as_tensor(batch_obs_n, dtype=torch.float32, device=device),
            actions=torch.as_tensor(act_buf.reshape(B, action_dim), dtype=torch.float32, device=device),
            old_logp=torch.as_tensor(logp_buf.reshape(B), dtype=torch.float32, device=device),
            returns=torch.as_tensor(ret_buf.reshape(B), dtype=torch.float32, device=device),
            advantages=torch.as_tensor(adv_buf.reshape(B), dtype=torch.float32, device=device),
            old_values=torch.as_tensor(val_buf.reshape(B), dtype=torch.float32, device=device),
        )

        # Advantage normalization.
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

        # PPO updates.
        idx = np.arange(B)
        for _ in range(args.ppo_epochs):
            rng.shuffle(idx)
            for start in range(0, B, args.minibatch_size):
                mb = idx[start : start + args.minibatch_size]

                obs_mb = batch.obs[mb]
                act_mb = batch.actions[mb]
                old_logp_mb = batch.old_logp[mb]
                adv_mb = batch.advantages[mb]
                ret_mb = batch.returns[mb]

                mean, std, value = model(obs_mb)
                mean = mean * max_action
                std = std * max_action

                new_logp = gaussian_log_prob(mean, std, act_mb)
                ratio = torch.exp(new_logp - old_logp_mb)

                pg1 = ratio * adv_mb
                pg2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_mb
                policy_loss = -torch.min(pg1, pg2).mean()

                value_loss = 0.5 * torch.mean((value - ret_mb) ** 2)
                entropy = gaussian_entropy(std).mean()

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                _ = nn_grad_norm
                optimizer.step()

        train_return_mean = float(np.mean(finished_returns[-100:])) if finished_returns else float("nan")
        train_len_mean = float(np.mean(finished_lengths[-100:])) if finished_lengths else float("nan")

        row: dict[str, Any] = {
            "update": update,
            "train_return_mean": train_return_mean,
            "train_length_mean": train_len_mean,
            "episodes_finished": len(finished_returns),
            "eval_return_mean": np.nan,
            "eval_return_std": np.nan,
            "eval_length_mean": np.nan,
            "eval_collision_rate": np.nan,
            "eval_escape_rate": np.nan,
            "eval_failure_rate": np.nan,
            "eval_final_pos_err": np.nan,
            "eval_final_vel_err": np.nan,
            "eval_strict_success_rate": np.nan,
            "eval_max_converge_streak_mean": np.nan,
            "eval_end_converge_streak_mean": np.nan,
        }

        if (update % args.eval_every == 0) or (update == args.updates):
            eval_stats = evaluate_policy(
                model=model,
                obs_rms=obs_rms,
                cfg=cfg,
                rw=rw,
                eval_episodes=args.eval_episodes,
                rng=rng,
                device=device,
                strict_mode=args.eval_strict_mode,
                pos_threshold=args.eval_pos_threshold,
                vel_threshold=args.eval_vel_threshold,
                consecutive_converged=args.eval_consecutive_converged,
                min_total_steps=args.eval_min_total_steps,
                lock_to_end=args.eval_lock_to_end,
            )
            row.update(eval_stats)

            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "obs_rms": obs_rms.state_dict(),
                "env_config": asdict(cfg),
                "reward_weights": asdict(rw),
                "train_args": vars(args),
                "update": update,
                "eval": eval_stats,
            }

            torch.save(ckpt, run_dir / "checkpoint_latest.pt")

            if eval_stats["eval_return_mean"] > best_eval:
                best_eval = eval_stats["eval_return_mean"]
            if args.eval_strict_mode:
                eval_key = (
                    1.0 - float(eval_stats["eval_strict_success_rate"]),
                    float(eval_stats["eval_failure_rate"]),
                    -float(eval_stats["eval_end_converge_streak_mean"]),
                    float(eval_stats["eval_final_pos_err"]),
                    float(eval_stats["eval_final_vel_err"]),
                    -float(eval_stats["eval_return_mean"]),
                )
            else:
                eval_key = (
                    float(eval_stats["eval_failure_rate"]),
                    float(eval_stats["eval_final_pos_err"]),
                    float(eval_stats["eval_final_vel_err"]),
                    -float(eval_stats["eval_return_mean"]),
                )
            topk_entries, changed = _update_topk_checkpoints(
                run_dir=run_dir,
                ckpt=ckpt,
                eval_key=tuple(float(x) for x in eval_key),
                update=update,
                topk_limit=topk_limit,
                topk_entries=topk_entries,
            )
            if changed and len(topk_entries) > 0:
                best_eval_key = tuple(topk_entries[0]["eval_key"])

        metrics_rows.append(row)

        print(
            f"[update {update:04d}] "
            f"train_return_mean={row['train_return_mean']:.3f} "
            f"eval_return_mean={row['eval_return_mean']:.3f} "
            f"eval_collision_rate={row['eval_collision_rate']:.3f} "
            f"eval_strict_success_rate={row['eval_strict_success_rate']:.3f}",
            flush=True,
        )

        write_metrics_csv(run_dir / "metrics.csv", metrics_rows)

    envs.close()
    maybe_plot_metrics(run_dir / "metrics.png", metrics_rows)

    print(f"[done] best_eval_return={best_eval:.3f}", flush=True)
    if best_eval_key is not None:
        print(f"[done] best_eval_key={best_eval_key}", flush=True)
    print(f"[done] artifacts at: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
