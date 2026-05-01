from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from fixed_init_profiles import resolve_fixed_init
from mpc_ilqr import Figure8ILQRMPC, ILQRConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect MPC/iLQR expert transitions for TD3 behavior cloning.")

    p.add_argument("--num-setups", type=int, default=64)
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--replan-every", type=int, default=1)

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
    p.add_argument("--horizon-steps", type=int, default=500)
    p.add_argument("--max-action-norm", type=float, default=2.5)
    p.add_argument("--near-collision-distance", type=float, default=0.35)
    p.add_argument("--escape-radius", type=float, default=8.0)
    p.add_argument("--phase-search-radius", type=int, default=35)
    p.add_argument("--action-dt", type=float, default=0.05)
    p.add_argument("--integrator-dt", type=float, default=0.001)

    p.add_argument("--w-pos-match", type=float, default=5.0)
    p.add_argument("--w-vel-match", type=float, default=3.0)
    p.add_argument("--w-switch-match", type=float, default=0.20)
    p.add_argument("--w-phase-match", type=float, default=0.15)

    p.add_argument("--mpc-horizon", type=int, default=60)
    p.add_argument("--mpc-iters", type=int, default=20)
    p.add_argument("--mpc-model-substeps", type=int, default=4)
    p.add_argument("--mpc-fd-eps-state", type=float, default=1e-4)
    p.add_argument("--mpc-fd-eps-action", type=float, default=1e-3)
    p.add_argument("--mpc-q-pos", type=float, default=80.0)
    p.add_argument("--mpc-q-vel", type=float, default=24.0)
    p.add_argument("--mpc-r-action", type=float, default=0.00005)
    p.add_argument("--mpc-terminal-scale", type=float, default=60.0)
    p.add_argument("--mpc-near-collision-weight", type=float, default=100.0)
    p.add_argument("--mpc-near-collision-distance", type=float, default=0.35)

    p.add_argument("--pos-threshold", type=float, default=0.06)
    p.add_argument("--vel-threshold", type=float, default=0.09)
    p.add_argument("--consecutive-converged", type=int, default=300)
    p.add_argument("--min-total-steps-for-converged", type=int, default=400)
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

    p.add_argument("--keep-only-strict", dest="keep_only_strict", action="store_true")
    p.add_argument("--keep-all-setups", dest="keep_only_strict", action="store_false")
    p.set_defaults(keep_only_strict=True)

    p.add_argument("--outdir", type=str, default="artifacts/mpc_datasets")
    p.add_argument("--tag", type=str, default="mpc_expert")
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


def run_one_setup(
    setup_seed: int,
    setup_index: int,
    cfg: EnvConfig,
    match_weights: RewardWeights,
    ilqr_cfg: ILQRConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = Figure8ChoreographyEnv(config=cfg, weights=match_weights)
    obs, info0 = env.reset(seed=setup_seed)

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

    obs_buf: list[np.ndarray] = []
    act_norm_buf: list[np.ndarray] = []
    next_obs_buf: list[np.ndarray] = []
    rew_buf: list[float] = []
    done_buf: list[float] = []
    step_id_buf: list[int] = []

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
    t = 0
    done = False
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

        denom = float(cfg.max_action_norm)
        if abs(denom) > 1e-12:
            action_norm = np.clip(action / denom, -1.0, 1.0).reshape(-1).astype(np.float32)
        else:
            action_norm = np.zeros((action.size,), dtype=np.float32)

        obs_next, reward, done, info = env.step(action)
        final_info = info

        obs_buf.append(obs.astype(np.float32))
        act_norm_buf.append(action_norm)
        next_obs_buf.append(obs_next.astype(np.float32))
        rew_buf.append(float(reward))
        done_buf.append(float(done))
        step_id_buf.append(int(t))

        step_collided = bool(info.get("collided", False))
        step_escaped = bool(info.get("escaped", False))
        collided_any = collided_any or step_collided
        escaped_any = escaped_any or step_escaped

        pos_err = float(info.get("position_error", np.nan))
        vel_err = float(info.get("velocity_direction_error", np.nan))
        phase_idx = int(info.get("phase_idx", phase_idx))
        assignment = tuple(int(z) for z in info.get("assignment", assignment))

        meets = (
            (pos_err <= float(args.pos_threshold))
            and (vel_err <= float(args.vel_threshold))
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

        if converged_step is None and converge_streak >= max(1, int(args.consecutive_converged)):
            converged_step = (t + 1) - int(args.consecutive_converged) + 1

        state = env.sim.get_state()
        x = controller.state_to_vector(state.positions, state.velocities)
        obs = obs_next
        t += 1
        steps_since_plan += 1

        if (t % max(1, int(args.log_every)) == 0) or done or (t == int(args.max_steps)):
            print(
                f"[setup={setup_index:03d} seed={setup_seed} step={t:4d}] "
                f"pos_err={pos_err:.5f} vel_err={vel_err:.5f} streak={converge_streak} "
                f"collided={step_collided} escaped={step_escaped}",
                flush=True,
            )

    final_pos_err = float(final_info.get("position_error", np.nan))
    final_vel_err = float(final_info.get("velocity_direction_error", np.nan))
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

    setup_summary = {
        "setup_index": int(setup_index),
        "setup_seed": int(setup_seed),
        "steps": int(t),
        "collided": bool(collided),
        "escaped": bool(escaped),
        "converged_step": converged_step,
        "basic_converged": bool(basic_converged),
        "strict_converged": bool(strict_converged),
        "max_converge_streak": int(max_converge_streak),
        "end_converge_streak": int(converge_streak),
        "final_pos_err": float(final_pos_err),
        "final_vel_err": float(final_vel_err),
        "planning_calls": int(planning_calls),
        "planning_time_total_sec": float(planning_time_sum),
        "planning_time_mean_sec": float(planning_time_sum / max(1, planning_calls)),
        "planning_iters_mean": float(planning_iter_sum / max(1, planning_calls)),
    }
    transitions = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions_norm": np.asarray(act_norm_buf, dtype=np.float32),
        "next_obs": np.asarray(next_obs_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "step_ids": np.asarray(step_id_buf, dtype=np.int32),
    }
    return setup_summary, transitions


def main() -> None:
    args = parse_args()

    cfg = make_env_config(args)
    match_weights = make_match_weights(args)
    ilqr_cfg = make_ilqr_config(args)

    out_dir = Path(args.outdir).resolve() / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "dataset.npz"
    out_meta = out_dir / "dataset_meta.json"

    rng = np.random.default_rng(args.seed)
    setup_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(int(args.num_setups))]

    all_obs: list[np.ndarray] = []
    all_actions_norm: list[np.ndarray] = []
    all_next_obs: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_dones: list[np.ndarray] = []
    all_step_ids: list[np.ndarray] = []
    all_setup_ids: list[np.ndarray] = []
    all_setup_seeds_by_transition: list[np.ndarray] = []
    setup_rows: list[dict[str, Any]] = []

    kept_setups = 0
    dropped_setups = 0

    for i, s in enumerate(setup_seeds):
        print(f"=== collecting setup {i + 1}/{len(setup_seeds)} seed={s} ===", flush=True)
        row, trans = run_one_setup(
            setup_seed=s,
            setup_index=i,
            cfg=cfg,
            match_weights=match_weights,
            ilqr_cfg=ilqr_cfg,
            args=args,
        )
        setup_rows.append(row)

        keep = bool(row["strict_converged"]) if args.keep_only_strict else True
        if keep:
            n = int(trans["obs"].shape[0])
            if n > 0:
                all_obs.append(trans["obs"])
                all_actions_norm.append(trans["actions_norm"])
                all_next_obs.append(trans["next_obs"])
                all_rewards.append(trans["rewards"])
                all_dones.append(trans["dones"])
                all_step_ids.append(trans["step_ids"])
                all_setup_ids.append(np.full((n,), i, dtype=np.int32))
                all_setup_seeds_by_transition.append(np.full((n,), s, dtype=np.int32))
                kept_setups += 1
        else:
            dropped_setups += 1

        print(
            f"[setup {i:03d}] strict={row['strict_converged']} collided={row['collided']} "
            f"escaped={row['escaped']} steps={row['steps']} keep={keep}",
            flush=True,
        )

    if len(all_obs) == 0:
        raise SystemExit("No transitions collected. Try --keep-all-setups or easier MPC settings.")

    obs = np.concatenate(all_obs, axis=0)
    actions_norm = np.concatenate(all_actions_norm, axis=0)
    next_obs = np.concatenate(all_next_obs, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    step_ids = np.concatenate(all_step_ids, axis=0)
    setup_ids = np.concatenate(all_setup_ids, axis=0)
    setup_seed_per_transition = np.concatenate(all_setup_seeds_by_transition, axis=0)

    np.savez_compressed(
        out_npz,
        obs=obs.astype(np.float32),
        actions_norm=actions_norm.astype(np.float32),
        next_obs=next_obs.astype(np.float32),
        rewards=rewards.astype(np.float32),
        dones=dones.astype(np.float32),
        step_ids=step_ids.astype(np.int32),
        setup_ids=setup_ids.astype(np.int32),
        setup_seed_per_transition=setup_seed_per_transition.astype(np.int32),
    )

    strict_success_rate = float(np.mean([1.0 if r["strict_converged"] else 0.0 for r in setup_rows]))
    meta = {
        "dataset_file": str(out_npz),
        "num_transitions": int(obs.shape[0]),
        "num_setups_requested": int(args.num_setups),
        "num_setups_kept": int(kept_setups),
        "num_setups_dropped": int(dropped_setups),
        "strict_success_rate_overall": strict_success_rate,
        "keep_only_strict": bool(args.keep_only_strict),
        "env_config": asdict(cfg),
        "match_weights": asdict(match_weights),
        "ilqr_config": asdict(ilqr_cfg),
        "collector_args": vars(args),
        "setup_rows": setup_rows,
    }
    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] dataset: {out_npz}", flush=True)
    print(f"[done] meta: {out_meta}", flush=True)
    print(
        f"[done] transitions={obs.shape[0]} kept_setups={kept_setups}/{args.num_setups} "
        f"strict_success_rate={strict_success_rate:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
