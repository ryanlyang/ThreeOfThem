from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from fixed_init_profiles import resolve_fixed_init
from mpc_ilqr import Figure8ILQRMPC, ILQRConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-init Figure-8 control with receding-horizon MPC/iLQR.")

    p.add_argument("--num-setups", type=int, default=10)
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--max-steps", type=int, default=420)
    p.add_argument("--log-every", type=int, default=20)

    # Environment setup.
    p.add_argument(
        "--fixed-init-profile",
        type=str,
        default="offset_ref",
        choices=["none", "weird", "near_ref", "offset_ref", "offset_ref_far"],
    )
    p.add_argument("--fixed-init-positions", type=str, default="")
    p.add_argument("--fixed-init-velocities", type=str, default="")
    p.add_argument("--fixed-init-pos-jitter-std", type=float, default=0.0)
    p.add_argument("--fixed-init-vel-jitter-std", type=float, default=0.0)
    p.add_argument("--fixed-init-jitter-tries", type=int, default=1)
    p.add_argument("--horizon-steps", type=int, default=420)
    p.add_argument("--max-action-norm", type=float, default=2.0)
    p.add_argument("--near-collision-distance", type=float, default=0.35)
    p.add_argument("--escape-radius", type=float, default=8.0)
    p.add_argument("--phase-search-radius", type=int, default=35)
    p.add_argument("--action-dt", type=float, default=0.05)
    p.add_argument("--integrator-dt", type=float, default=0.001)

    # Matcher weights for phase+assignment tracking.
    p.add_argument("--w-pos-match", type=float, default=4.0)
    p.add_argument("--w-vel-match", type=float, default=2.5)
    p.add_argument("--w-switch-match", type=float, default=0.2)
    p.add_argument("--w-phase-match", type=float, default=0.12)

    # iLQR planner config.
    p.add_argument("--mpc-horizon", type=int, default=42)
    p.add_argument("--mpc-iters", type=int, default=14)
    p.add_argument("--mpc-model-substeps", type=int, default=4)
    p.add_argument("--mpc-fd-eps-state", type=float, default=1e-4)
    p.add_argument("--mpc-fd-eps-action", type=float, default=1e-3)
    p.add_argument("--mpc-q-pos", type=float, default=50.0)
    p.add_argument("--mpc-q-vel", type=float, default=16.0)
    p.add_argument("--mpc-r-action", type=float, default=0.0002)
    p.add_argument("--mpc-terminal-scale", type=float, default=40.0)
    p.add_argument("--mpc-near-collision-weight", type=float, default=40.0)
    p.add_argument("--mpc-near-collision-distance", type=float, default=0.28)
    p.add_argument("--replan-every", type=int, default=1)

    # Strict convergence criteria.
    p.add_argument("--pos-threshold", type=float, default=0.06)
    p.add_argument("--vel-threshold", type=float, default=0.09)
    p.add_argument("--consecutive-converged", type=int, default=260)
    p.add_argument("--min-total-steps-for-converged", type=int, default=320)

    p.add_argument("--strict-mode", dest="strict_mode", action="store_true")
    p.add_argument("--no-strict-mode", dest="strict_mode", action="store_false")
    p.set_defaults(strict_mode=True)

    p.add_argument("--lock-to-end", dest="lock_to_end", action="store_true")
    p.add_argument("--no-lock-to-end", dest="lock_to_end", action="store_false")
    p.set_defaults(lock_to_end=True)

    p.add_argument("--require-no-failure", dest="require_no_failure", action="store_true")
    p.add_argument("--no-require-no-failure", dest="require_no_failure", action="store_false")
    p.set_defaults(require_no_failure=True)

    p.add_argument("--require-final-threshold", dest="require_final_threshold", action="store_true")
    p.add_argument("--no-require-final-threshold", dest="require_final_threshold", action="store_false")
    p.set_defaults(require_final_threshold=True)

    # Output.
    p.add_argument("--trail-len", type=int, default=90)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--axis-pad", type=float, default=0.20)
    p.add_argument("--show-title", dest="show_title", action="store_true")
    p.add_argument("--no-show-title", dest="show_title", action="store_false")
    p.set_defaults(show_title=True)
    p.add_argument("--outdir", type=str, default="inference_mpc_ilqr_fixed_init")

    return p.parse_args()


def make_env_config(args: argparse.Namespace) -> EnvConfig:
    fixed_pos, fixed_vel = resolve_fixed_init(
        profile=args.fixed_init_profile,
        positions_spec=args.fixed_init_positions,
        velocities_spec=args.fixed_init_velocities,
    )
    return EnvConfig(
        backend="numpy",
        seed=args.seed,
        horizon_steps=args.horizon_steps,
        action_dt=args.action_dt,
        integrator_dt=args.integrator_dt,
        phase_search_radius=args.phase_search_radius,
        max_action_norm=args.max_action_norm,
        near_collision_distance=args.near_collision_distance,
        escape_radius=args.escape_radius,
        fixed_init_positions=fixed_pos,
        fixed_init_velocities=fixed_vel,
        fixed_init_pos_jitter_std=args.fixed_init_pos_jitter_std,
        fixed_init_vel_jitter_std=args.fixed_init_vel_jitter_std,
        fixed_init_jitter_tries=args.fixed_init_jitter_tries,
    )


def make_match_weights(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        position=args.w_pos_match,
        velocity_direction=args.w_vel_match,
        fuel=0.0,
        near_collision=0.0,
        collision=0.0,
        escape=0.0,
        permutation_switch=args.w_switch_match,
        phase_jump=args.w_phase_match,
    )


def make_ilqr_config(args: argparse.Namespace) -> ILQRConfig:
    return ILQRConfig(
        horizon=args.mpc_horizon,
        iterations=args.mpc_iters,
        model_substeps=args.mpc_model_substeps,
        fd_eps_state=args.mpc_fd_eps_state,
        fd_eps_action=args.mpc_fd_eps_action,
        q_pos=args.mpc_q_pos,
        q_vel=args.mpc_q_vel,
        r_action=args.mpc_r_action,
        terminal_scale=args.mpc_terminal_scale,
        near_collision_weight=args.mpc_near_collision_weight,
        near_collision_distance=args.mpc_near_collision_distance,
    )


def _trail_segment(arr: np.ndarray, idx: int, length: int) -> np.ndarray:
    start = max(0, idx - length)
    return arr[start : idx + 1]


def save_rollout_gif(
    positions_hist: np.ndarray,
    reference_path: np.ndarray,
    out_gif: Path,
    title: str,
    show_title: bool,
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
    if show_title:
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
    setup_seed: int,
    cfg: EnvConfig,
    match_weights: RewardWeights,
    ilqr_cfg: ILQRConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env = Figure8ChoreographyEnv(config=cfg, weights=match_weights)
    _, info0 = env.reset(seed=setup_seed)

    controller = Figure8ILQRMPC(
        env_cfg=cfg,
        reference=env.reference,
        phase_offsets=env.phase_offsets,
        expected_phase_stride=env.expected_phase_stride,
        ilqr_cfg=ilqr_cfg,
    )

    phase_idx = int(info0["phase_idx"])
    assignment = tuple(int(x) for x in info0["assignment"])
    state = env.sim.get_state()
    x = controller.state_to_vector(state.positions, state.velocities)

    positions_hist = [state.positions.copy()]
    action_norm_hist = []
    pos_err_hist = []
    vel_err_hist = []
    phase_hist = [phase_idx]

    plan_actions: np.ndarray | None = None
    warm_actions: np.ndarray | None = None
    steps_since_plan = max(1, int(args.replan_every))

    planning_calls = 0
    planning_time_sum = 0.0
    planning_iter_sum = 0.0

    converged_step = None
    converge_streak = 0
    max_converge_streak = 0
    broke_after_lock = False
    collided_any = False
    escaped_any = False
    done = False
    t = 0
    final_info: dict[str, Any] = {}

    while not done and t < int(args.max_steps):
        need_replan = plan_actions is None or steps_since_plan >= max(1, int(args.replan_every))
        if need_replan:
            t0 = time.perf_counter()
            plan = controller.plan(
                x0=x,
                phase_idx=phase_idx,
                assignment=assignment,
                warm_start=warm_actions,
            )
            dt = time.perf_counter() - t0
            planning_calls += 1
            planning_time_sum += dt
            planning_iter_sum += float(plan.iterations)
            plan_actions = plan.actions.copy()
            warm_actions = controller.shift_actions(plan.actions)
            steps_since_plan = 0

        assert plan_actions is not None
        action_vec = plan_actions[0].copy()
        plan_actions = controller.shift_actions(plan_actions)

        action = action_vec.reshape(env.action_shape)
        _, _, done, info = env.step(action)
        final_info = info

        step_collided = bool(info.get("collided", False))
        step_escaped = bool(info.get("escaped", False))
        collided_any = collided_any or step_collided
        escaped_any = escaped_any or step_escaped

        pos_err = float(info.get("position_error", np.nan))
        vel_err = float(info.get("velocity_direction_error", np.nan))
        phase_idx = int(info.get("phase_idx", phase_idx))
        assignment = tuple(int(z) for z in info.get("assignment", assignment))

        state = env.sim.get_state()
        x = controller.state_to_vector(state.positions, state.velocities)
        positions_hist.append(state.positions.copy())
        phase_hist.append(phase_idx)
        pos_err_hist.append(pos_err)
        vel_err_hist.append(vel_err)
        action_norm_hist.append(float(np.linalg.norm(action, axis=1).mean()))

        meets = (pos_err <= args.pos_threshold) and (vel_err <= args.vel_threshold) and (not step_collided) and (not step_escaped)
        if meets:
            converge_streak += 1
            max_converge_streak = max(max_converge_streak, converge_streak)
        else:
            if converged_step is not None:
                broke_after_lock = True
            converge_streak = 0

        if converged_step is None and converge_streak >= max(1, int(args.consecutive_converged)):
            converged_step = (t + 1) - int(args.consecutive_converged) + 1

        t += 1
        steps_since_plan += 1

        if (t % max(1, int(args.log_every)) == 0) or done or (t == int(args.max_steps)):
            print(
                f"[seed={setup_seed} step={t:4d}] pos_err={pos_err:.5f} vel_err={vel_err:.5f} "
                f"streak={converge_streak} collided={step_collided} escaped={step_escaped}",
                flush=True,
            )

    if len(pos_err_hist) == 0:
        min_pos_err = float("inf")
        min_pos_step = args.max_steps + 1
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
        and (final_pos_err <= float(args.pos_threshold))
        and (final_vel_err <= float(args.vel_threshold))
    )
    enough_total_steps = t >= max(0, int(args.min_total_steps_for_converged))
    sustained_after_lock = (
        (converged_step is not None)
        and (not broke_after_lock)
        and (converge_streak >= max(1, int(args.consecutive_converged)))
    )

    basic_converged = converged_step is not None
    strict_converged = basic_converged
    if args.strict_mode:
        if args.lock_to_end:
            strict_converged = strict_converged and sustained_after_lock
        if args.require_no_failure:
            strict_converged = strict_converged and no_failure
        if args.require_final_threshold:
            strict_converged = strict_converged and final_under_threshold
        strict_converged = strict_converged and enough_total_steps

    has_converged = 0 if strict_converged else 1
    has_failure = 1 if not no_failure else 0
    rank_step = converged_step if converged_step is not None else (int(args.max_steps) + 1)
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
        "mean_action_norm": float(np.mean(action_norm_hist)) if action_norm_hist else float("nan"),
        "planning_calls": int(planning_calls),
        "planning_time_total_sec": float(planning_time_sum),
        "planning_time_mean_sec": float(planning_time_sum / max(1, planning_calls)),
        "planning_iters_mean": float(planning_iter_sum / max(1, planning_calls)),
        "rank_tuple": rank_tuple,
        "positions_hist": np.asarray(positions_hist, dtype=np.float64),
        "pos_err_hist": pos_err_hist,
        "vel_err_hist": vel_err_hist,
        "phase_hist": phase_hist,
        "last_info": final_info,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_env_config(args)
    match_weights = make_match_weights(args)
    ilqr_cfg = make_ilqr_config(args)

    rng = np.random.default_rng(args.seed)
    setup_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(int(args.num_setups))]

    runs = []
    for i, s in enumerate(setup_seeds):
        print(f"=== setup {i + 1}/{len(setup_seeds)} seed={s} ===", flush=True)
        rec = run_one_setup(
            setup_seed=s,
            cfg=cfg,
            match_weights=match_weights,
            ilqr_cfg=ilqr_cfg,
            args=args,
        )
        print(
            f"[result seed={s}] strict_converged={rec['strict_converged']} "
            f"converged_step={rec['converged_step']} max_streak={rec['max_converge_streak']} "
            f"final_pos_err={rec['final_pos_err']:.6f} final_vel_err={rec['final_vel_err']:.6f} "
            f"collided={rec['collided']} escaped={rec['escaped']} "
            f"plan_time_mean={rec['planning_time_mean_sec']:.3f}s",
            flush=True,
        )
        runs.append(rec)

    best_idx, best = min(enumerate(runs), key=lambda kv: tuple(kv[1]["rank_tuple"]))
    reference_path = Figure8ChoreographyEnv(config=cfg, weights=match_weights).reference.positions

    for i, r in enumerate(runs):
        gif_path = out_dir / f"setup_{i:02d}_seed_{r['setup_seed']}.gif"
        save_rollout_gif(
            positions_hist=r["positions_hist"],
            reference_path=reference_path,
            out_gif=gif_path,
            title=f"MPC/iLQR Setup {i} (seed={r['setup_seed']})",
            show_title=bool(args.show_title),
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
        title=f"MPC/iLQR Best Convergence (seed={best['setup_seed']})",
        show_title=bool(args.show_title),
        trail_len=args.trail_len,
        frame_stride=args.frame_stride,
        axis_pad=args.axis_pad,
    )

    summary_rows = []
    for r in runs:
        rr = {
            k: v
            for k, v in r.items()
            if k not in {"positions_hist", "pos_err_hist", "vel_err_hist", "phase_hist", "last_info"}
        }
        summary_rows.append(rr)

    summary = {
        "controller": "mpc_ilqr",
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
        "match_weights": asdict(match_weights),
        "ilqr_config": asdict(ilqr_cfg),
    }

    summary_path = out_dir / "inference_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {len(runs)} setup gifs under: {out_dir}", flush=True)
    print(f"saved best gif: {best_gif}", flush=True)
    print(f"saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
