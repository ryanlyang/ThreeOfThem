from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for TD3 inference. Install requirements-train.txt.") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from ppo_agent import RunningMeanStd
from td3_agent import TD3Actor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TD3 checkpoint on multiple setups and save best-convergence GIF.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-setups", type=int, default=10)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--max-steps", type=int, default=420)

    p.add_argument("--pos-threshold", type=float, default=0.06)
    p.add_argument("--vel-threshold", type=float, default=0.09)
    p.add_argument("--consecutive-converged", type=int, default=260)
    p.add_argument("--strict-mode", action="store_true")
    p.add_argument("--lock-to-end", action="store_true")
    p.add_argument("--require-no-failure", action="store_true")
    p.add_argument("--require-final-threshold", action="store_true")
    p.add_argument("--min-total-steps-for-converged", type=int, default=320)

    p.add_argument("--trail-len", type=int, default=90)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--axis-pad", type=float, default=0.20)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--obs-clip", type=float, default=10.0)

    p.add_argument("--fixed-init-pos-jitter-std", type=float, default=-1.0)
    p.add_argument("--fixed-init-vel-jitter-std", type=float, default=-1.0)
    p.add_argument("--fixed-init-jitter-tries", type=int, default=64)
    p.add_argument("--override-fixed-jitter-zero", action="store_true")

    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _trail_segment(arr: np.ndarray, idx: int, length: int) -> np.ndarray:
    start = max(0, idx - length)
    return arr[start : idx + 1]


def save_rollout_gif(
    positions_hist: np.ndarray,
    reference_path: np.ndarray,
    out_gif: Path,
    title: str,
    trail_len: int,
    frame_stride: int,
    axis_pad: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    ax.plot(reference_path[:, 0], reference_path[:, 1], color="navy", linewidth=2.0, alpha=0.9, label="Reference")

    colors = ["darkorange", "forestgreen", "purple"]
    labels = ["Body 0", "Body 1", "Body 2"]

    points = []
    trails = []
    for c, label in zip(colors, labels):
        pt = ax.scatter([], [], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5, label=label)
        (tr,) = ax.plot([], [], color=c, linewidth=2.0, alpha=0.6)
        points.append(pt)
        trails.append(tr)

    t_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    pts_all = np.concatenate([reference_path, positions_hist.reshape(-1, 2)], axis=0)
    finite_mask = np.all(np.isfinite(pts_all), axis=1)
    pts_all = pts_all[finite_mask]
    if pts_all.shape[0] > 0:
        x_min = float(np.min(pts_all[:, 0]))
        x_max = float(np.max(pts_all[:, 0]))
        y_min = float(np.min(pts_all[:, 1]))
        y_max = float(np.max(pts_all[:, 1]))
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half = 0.5 * max(x_max - x_min, y_max - y_min)
        half = max(half * (1.0 + max(0.0, axis_pad)), 1.2)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", fontsize=9)

    n_frames = positions_hist.shape[0]
    frame_idx = np.arange(0, n_frames, max(1, frame_stride))

    def update(k: int):
        artists = []
        for i in range(3):
            traj = positions_hist[:, i, :]
            p = traj[k]
            points[i].set_offsets(p[None, :])
            artists.append(points[i])

            tr_seg = _trail_segment(traj, k, trail_len)
            trails[i].set_data(tr_seg[:, 0], tr_seg[:, 1])
            artists.append(trails[i])

        t_text.set_text(f"step = {k}")
        artists.append(t_text)
        return artists

    anim = FuncAnimation(fig, update, frames=frame_idx, interval=35, blit=False)
    writer = PillowWriter(fps=20)
    anim.save(out_gif, writer=writer)
    plt.close(fig)


def run_one_setup(
    actor: TD3Actor,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    rw: RewardWeights,
    setup_seed: int,
    max_steps: int,
    obs_clip: float,
    pos_threshold: float,
    vel_threshold: float,
    consecutive_converged: int,
    strict_mode: bool,
    lock_to_end: bool,
    require_no_failure: bool,
    require_final_threshold: bool,
    min_total_steps_for_converged: int,
    device: torch.device,
) -> dict[str, Any]:
    env = Figure8ChoreographyEnv(config=cfg, weights=rw)
    obs, _ = env.reset(seed=setup_seed)

    positions_hist = [env.sim.get_state().positions.copy()]
    pos_err_hist = []
    vel_err_hist = []
    reward_hist = []
    phase_hist = []

    converged_step = None
    converge_streak = 0
    max_converge_streak = 0
    broke_after_lock = False
    collided_any = False
    escaped_any = False
    done = False
    t = 0

    while not done and t < max_steps:
        obs_n = obs_rms.normalize(obs[None, :], clip=obs_clip)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_norm = actor(obs_t).squeeze(0).cpu().numpy()

        action = (action_norm * cfg.max_action_norm).reshape(env.action_shape)
        action = np.clip(action, -cfg.max_action_norm, cfg.max_action_norm)

        obs, reward, done, info = env.step(action)
        positions_hist.append(env.sim.get_state().positions.copy())

        pos_err = float(info.get("position_error", np.nan))
        vel_err = float(info.get("velocity_direction_error", np.nan))
        step_collided = bool(info.get("collided", False))
        step_escaped = bool(info.get("escaped", False))
        collided_any = collided_any or step_collided
        escaped_any = escaped_any or step_escaped
        pos_err_hist.append(pos_err)
        vel_err_hist.append(vel_err)
        reward_hist.append(float(reward))
        phase_hist.append(int(info.get("phase_idx", -1)))

        meets = (pos_err <= pos_threshold) and (vel_err <= vel_threshold) and (not step_collided) and (not step_escaped)
        if meets:
            converge_streak += 1
            max_converge_streak = max(max_converge_streak, converge_streak)
        else:
            if converged_step is not None:
                broke_after_lock = True
            converge_streak = 0

        if converged_step is None and converge_streak >= max(1, int(consecutive_converged)):
            converged_step = (t + 1) - int(consecutive_converged) + 1

        t += 1

    if len(pos_err_hist) == 0:
        min_pos_err = float("inf")
        min_pos_step = max_steps + 1
    else:
        min_pos_step = int(np.argmin(pos_err_hist) + 1)
        min_pos_err = float(np.min(pos_err_hist))

    final_pos_err = float(pos_err_hist[-1]) if pos_err_hist else float("nan")
    final_vel_err = float(vel_err_hist[-1]) if vel_err_hist else float("nan")
    collided = bool(collided_any)
    escaped = bool(escaped_any)
    no_failure = not (collided or escaped)

    final_under_threshold = (
        np.isfinite(final_pos_err)
        and np.isfinite(final_vel_err)
        and (final_pos_err <= pos_threshold)
        and (final_vel_err <= vel_threshold)
    )
    enough_total_steps = t >= max(0, int(min_total_steps_for_converged))
    sustained_after_lock = (
        (converged_step is not None)
        and (not broke_after_lock)
        and (converge_streak >= max(1, int(consecutive_converged)))
    )

    basic_converged = converged_step is not None
    strict_converged = basic_converged
    if strict_mode:
        if lock_to_end:
            strict_converged = strict_converged and sustained_after_lock
        if require_no_failure:
            strict_converged = strict_converged and no_failure
        if require_final_threshold:
            strict_converged = strict_converged and final_under_threshold
        strict_converged = strict_converged and enough_total_steps

    if strict_mode:
        has_converged = 0 if strict_converged else 1
        has_failure = 1 if (collided or escaped) else 0
        rank_step = converged_step if converged_step is not None else (max_steps + 1)
        rank_tuple = [
            has_converged,
            has_failure,
            rank_step,
            -int(max_converge_streak),
            final_pos_err,
            final_vel_err,
            min_pos_err,
            min_pos_step,
        ]
    else:
        has_converged = 0 if converged_step is not None else 1
        has_failure = 1 if (collided or escaped) else 0
        rank_step = converged_step if converged_step is not None else (max_steps + 1)
        rank_tuple = [has_converged, has_failure, rank_step, final_pos_err, min_pos_err, final_vel_err, min_pos_step]

    return {
        "setup_seed": int(setup_seed),
        "steps": int(t),
        "collided": bool(collided),
        "escaped": bool(escaped),
        "converged_step": converged_step,
        "basic_converged": bool(basic_converged),
        "strict_converged": bool(strict_converged),
        "max_converge_streak": int(max_converge_streak),
        "end_converge_streak": int(converge_streak),
        "sustained_after_lock": bool(sustained_after_lock),
        "final_under_threshold": bool(final_under_threshold),
        "no_failure": bool(no_failure),
        "enough_total_steps": bool(enough_total_steps),
        "min_pos_err": min_pos_err,
        "min_pos_step": int(min_pos_step),
        "final_pos_err": final_pos_err,
        "final_vel_err": final_vel_err,
        "return_sum": float(np.sum(reward_hist)),
        "rank_tuple": rank_tuple,
        "positions_hist": np.asarray(positions_hist, dtype=np.float64),
        "pos_err_hist": pos_err_hist,
        "vel_err_hist": vel_err_hist,
        "phase_hist": phase_hist,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, device=device)
    if ckpt.get("algo") != "td3":
        raise SystemExit(f"Checkpoint is not TD3 (algo={ckpt.get('algo')!r}).")

    cfg = EnvConfig(**ckpt["eval_env_config"] if "eval_env_config" in ckpt else ckpt["env_config"])
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

    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = ckpt_path.parent / "inference_td3_fixed_init"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    setup_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(args.num_setups)]

    runs = []
    for i, s in enumerate(setup_seeds):
        rec = run_one_setup(
            actor=actor,
            obs_rms=obs_rms,
            cfg=cfg,
            rw=rw,
            setup_seed=s,
            max_steps=args.max_steps,
            obs_clip=args.obs_clip,
            pos_threshold=args.pos_threshold,
            vel_threshold=args.vel_threshold,
            consecutive_converged=args.consecutive_converged,
            strict_mode=args.strict_mode,
            lock_to_end=args.lock_to_end,
            require_no_failure=args.require_no_failure,
            require_final_threshold=args.require_final_threshold,
            min_total_steps_for_converged=args.min_total_steps_for_converged,
            device=device,
        )
        print(
            f"[setup {i+1}/{args.num_setups} seed={s}] strict={rec['strict_converged']} "
            f"converged_step={rec['converged_step']} final_pos_err={rec['final_pos_err']:.5f} "
            f"final_vel_err={rec['final_vel_err']:.5f} collided={rec['collided']} escaped={rec['escaped']}",
            flush=True,
        )
        runs.append(rec)

    best_idx, best = min(enumerate(runs), key=lambda kv: tuple(kv[1]["rank_tuple"]))

    env_ref = Figure8ChoreographyEnv(config=cfg, weights=rw)
    reference_path = env_ref.reference.positions

    for i, r in enumerate(runs):
        gif_path = out_dir / f"setup_{i:02d}_seed_{r['setup_seed']}.gif"
        save_rollout_gif(
            positions_hist=r["positions_hist"],
            reference_path=reference_path,
            out_gif=gif_path,
            title=f"TD3 Setup {i} (seed={r['setup_seed']})",
            trail_len=args.trail_len,
            frame_stride=args.frame_stride,
            axis_pad=args.axis_pad,
        )
        r["gif_path"] = str(gif_path)

    best_gif = out_dir / f"best_convergence_seed_{best['setup_seed']}.gif"
    save_rollout_gif(
        positions_hist=best["positions_hist"],
        reference_path=reference_path,
        out_gif=best_gif,
        title=f"TD3 Best Convergence (seed={best['setup_seed']})",
        trail_len=args.trail_len,
        frame_stride=args.frame_stride,
        axis_pad=args.axis_pad,
    )

    summary_rows = []
    for r in runs:
        rr = {
            k: v
            for k, v in r.items()
            if k not in {"positions_hist", "pos_err_hist", "vel_err_hist", "phase_hist"}
        }
        summary_rows.append(rr)

    summary = {
        "controller": "td3",
        "num_setups": int(args.num_setups),
        "setup_seeds": setup_seeds,
        "strict_criteria": {
            "strict_mode": bool(args.strict_mode),
            "lock_to_end": bool(args.lock_to_end),
            "require_no_failure": bool(args.require_no_failure),
            "require_final_threshold": bool(args.require_final_threshold),
            "min_total_steps_for_converged": int(args.min_total_steps_for_converged),
        },
        "thresholds": {"pos": float(args.pos_threshold), "vel": float(args.vel_threshold)},
        "consecutive_converged": int(args.consecutive_converged),
        "best_index": int(best_idx),
        "best_seed": int(best["setup_seed"]),
        "best_rank_tuple": best["rank_tuple"],
        "best_gif": str(best_gif),
        "runs": summary_rows,
        "env_config": asdict(cfg),
        "reward_weights": asdict(rw),
        "checkpoint": str(ckpt_path.resolve()),
    }

    summary_path = out_dir / "inference_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {len(runs)} setup gifs under: {out_dir}", flush=True)
    print(f"saved best gif: {best_gif}", flush=True)
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
