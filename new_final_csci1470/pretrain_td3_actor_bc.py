from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise SystemExit("Torch is required for BC pretraining. Install requirements-train.txt.") from exc

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from ppo_agent import RunningMeanStd
from td3_agent import TD3Actor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior clone TD3 actor from MPC/iLQR expert dataset.")
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset.npz from collect_mpc_ilqr_dataset.py")
    p.add_argument("--dataset-meta", type=str, default="", help="Optional dataset_meta.json path")
    p.add_argument("--outdir", type=str, default="artifacts/bc_td3")
    p.add_argument("--run-name", type=str, default="td3_bc_mpc")

    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=4301)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--obs-clip", type=float, default=10.0)

    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-max-steps", type=int, default=500)
    p.add_argument("--eval-pos-threshold", type=float, default=0.06)
    p.add_argument("--eval-vel-threshold", type=float, default=0.09)
    p.add_argument("--eval-consecutive-converged", type=int, default=260)
    p.add_argument("--eval-min-total-steps", type=int, default=320)
    p.add_argument("--eval-strict-mode", action="store_true")
    p.add_argument("--eval-lock-to-end", action="store_true")
    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta(dataset_path: Path, explicit_meta: str) -> dict[str, Any]:
    if explicit_meta:
        meta_path = Path(explicit_meta).resolve()
    else:
        meta_path = dataset_path.with_name("dataset_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing dataset meta json: {meta_path}")
    with meta_path.open("r") as f:
        return json.load(f)


def evaluate_actor(
    actor: TD3Actor,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    rw: RewardWeights,
    *,
    eval_episodes: int,
    max_steps: int,
    seed: int,
    obs_clip: float,
    pos_threshold: float,
    vel_threshold: float,
    consecutive_converged: int,
    min_total_steps: int,
    strict_mode: bool,
    lock_to_end: bool,
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    collisions = []
    escapes = []
    strict_success = []
    final_pos_err = []
    final_vel_err = []

    actor.eval()
    for _ in range(int(eval_episodes)):
        env = Figure8ChoreographyEnv(config=cfg, weights=rw)
        obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
        done = False
        t = 0
        info: dict[str, Any] = {}

        converge_streak = 0
        converged_step = None
        broke_after_lock = False
        collided_any = False
        escaped_any = False

        while not done and t < int(max_steps):
            obs_n = obs_rms.normalize(obs[None, :], clip=obs_clip)
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_norm = actor(obs_t).squeeze(0).cpu().numpy()
            action = (action_norm * cfg.max_action_norm).reshape((cfg.num_bodies, cfg.dimensions))
            action = np.clip(action, -cfg.max_action_norm, cfg.max_action_norm)

            obs, _, done, info = env.step(action)
            t += 1

            step_collided = bool(info.get("collided", False))
            step_escaped = bool(info.get("escaped", False))
            collided_any = collided_any or step_collided
            escaped_any = escaped_any or step_escaped

            pos_err = float(info.get("position_error", np.nan))
            vel_err = float(info.get("velocity_direction_error", np.nan))
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
            else:
                if converged_step is not None:
                    broke_after_lock = True
                converge_streak = 0

            if converged_step is None and converge_streak >= max(1, int(consecutive_converged)):
                converged_step = (t + 1) - int(consecutive_converged) + 1

        collided = bool(collided_any)
        escaped = bool(escaped_any)
        no_failure = not (collided or escaped)
        f_pos = float(info.get("position_error", np.nan))
        f_vel = float(info.get("velocity_direction_error", np.nan))
        under = np.isfinite(f_pos) and np.isfinite(f_vel) and (f_pos <= pos_threshold) and (f_vel <= vel_threshold)
        enough = t >= max(0, int(min_total_steps))
        sustained = (
            (converged_step is not None)
            and (not broke_after_lock)
            and (converge_streak >= max(1, int(consecutive_converged)))
        )

        basic = converged_step is not None
        strict = basic
        if strict_mode:
            strict = strict and no_failure and under and enough
            if lock_to_end:
                strict = strict and sustained

        collisions.append(float(collided))
        escapes.append(float(escaped))
        strict_success.append(float(strict))
        final_pos_err.append(f_pos)
        final_vel_err.append(f_vel)

    actor.train()
    return {
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_escape_rate": float(np.mean(escapes)),
        "eval_failure_rate": float(np.mean((np.asarray(collisions) + np.asarray(escapes)) > 0.0)),
        "eval_final_pos_err": float(np.nanmean(final_pos_err)),
        "eval_final_vel_err": float(np.nanmean(final_vel_err)),
        "eval_strict_success_rate": float(np.mean(strict_success)),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)

    ds_path = Path(args.dataset).resolve()
    ds = np.load(ds_path)
    meta = load_meta(ds_path, args.dataset_meta)

    obs = np.asarray(ds["obs"], dtype=np.float32)
    actions = np.asarray(ds["actions_norm"], dtype=np.float32)
    if obs.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected obs/actions_norm to be 2D, got {obs.shape=} {actions.shape=}")
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"Mismatched samples: obs={obs.shape[0]} actions={actions.shape[0]}")

    obs_dim = int(obs.shape[1])
    action_dim = int(actions.shape[1])
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("Invalid dataset dimensions.")

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.update(obs.astype(np.float64))
    obs_n = obs_rms.normalize(obs, clip=args.obs_clip).astype(np.float32)

    n = int(obs_n.shape[0])
    val_n = int(round(float(args.val_frac) * n))
    val_n = max(1, min(n - 1, val_n))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    x_train = torch.as_tensor(obs_n[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(actions[train_idx], dtype=torch.float32)
    x_val = torch.as_tensor(obs_n[val_idx], dtype=torch.float32)
    y_val = torch.as_tensor(actions[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )

    actor = TD3Actor(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    rows: list[dict[str, float]] = []

    t0 = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = actor(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            opt.step()
            train_losses.append(float(loss.item()))

        actor.eval()
        with torch.no_grad():
            val_pred = actor(x_val.to(device))
            val_loss = torch.mean((val_pred - y_val.to(device)) ** 2).item()

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        row = {"epoch": float(epoch), "train_mse": train_loss, "val_mse": float(val_loss)}
        rows.append(row)
        print(f"[epoch {epoch:03d}] train_mse={train_loss:.8f} val_mse={val_loss:.8f}", flush=True)

        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= max(1, int(args.early_stop_patience)):
                print(f"[early-stop] no val improvement for {bad_epochs} epochs", flush=True)
                break

    if best_state is None:
        raise RuntimeError("BC training did not produce a valid checkpoint.")
    actor.load_state_dict(best_state)

    cfg = EnvConfig(**meta["env_config"])
    rw = RewardWeights(**meta["match_weights"])
    eval_out = evaluate_actor(
        actor=actor,
        obs_rms=obs_rms,
        cfg=cfg,
        rw=rw,
        eval_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        seed=args.seed + 999,
        obs_clip=args.obs_clip,
        pos_threshold=args.eval_pos_threshold,
        vel_threshold=args.eval_vel_threshold,
        consecutive_converged=args.eval_consecutive_converged,
        min_total_steps=args.eval_min_total_steps,
        strict_mode=bool(args.eval_strict_mode),
        lock_to_end=bool(args.eval_lock_to_end),
        device=device,
    )

    run_dir = Path(args.outdir).resolve() / f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint_bc.pt"

    ckpt = {
        "algo": "td3",
        "source": "mpc_bc",
        "actor_state_dict": actor.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "env_config": asdict(cfg),
        "eval_env_config": asdict(cfg),
        "reward_weights": asdict(rw),
        "args": vars(args),
        "dataset_path": str(ds_path),
        "dataset_meta": meta,
        "bc_best_val_mse": float(best_val),
        "bc_rows": rows,
        "bc_eval": eval_out,
    }
    torch.save(ckpt, ckpt_path)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "dataset": str(ds_path),
        "num_samples": int(n),
        "train_samples": int(train_idx.shape[0]),
        "val_samples": int(val_idx.shape[0]),
        "best_val_mse": float(best_val),
        "elapsed_sec": float(time.perf_counter() - t0),
        "bc_eval": eval_out,
    }
    with (run_dir / "bc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] checkpoint={ckpt_path}", flush=True)
    print(
        f"[done] best_val_mse={best_val:.8f} strict_success_rate={eval_out['eval_strict_success_rate']:.3f} "
        f"failure_rate={eval_out['eval_failure_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
