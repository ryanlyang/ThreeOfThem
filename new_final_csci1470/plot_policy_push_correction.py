from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights
from ppo_agent import ActorCritic, RunningMeanStd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a 2-panel paper figure showing policy/controller pushes toward the figure-8 choreography."
    )
    p.add_argument("--checkpoint", type=str, default="", help="Optional PPO checkpoint (.pt).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--delta-steps", type=int, default=20)
    p.add_argument("--noise-pos", type=float, default=0.22)
    p.add_argument("--noise-vel", type=float, default=0.16)
    p.add_argument("--axis-pad", type=float, default=0.2)
    p.add_argument("--vel-arrow-len", type=float, default=0.18)
    p.add_argument("--push-arrow-len", type=float, default=0.25)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--basename", type=str, default="policy_push_correction")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(name: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("Torch is required to run this script.")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _resolve_ckpt_path(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_file():
        return p
    return None


def _load_checkpoint(path: Path, device: "torch.device") -> dict[str, Any]:
    assert torch is not None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_off_target_fixed_init(cfg: EnvConfig, seed: int, noise_pos: float, noise_vel: float) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    rng = np.random.default_rng(seed)
    # Build around one coherent figure-8 phase instant so bodies start "close but off".
    temp_env = Figure8ChoreographyEnv(config=cfg, weights=RewardWeights())
    n = temp_env.reference.num_samples
    offsets = temp_env.phase_offsets
    k = int(rng.integers(0, n))

    base_pos = np.zeros((3, 2), dtype=np.float64)
    base_vel = np.zeros((3, 2), dtype=np.float64)
    for i, off in enumerate(offsets):
        idx = (k + off) % n
        base_pos[i] = temp_env.reference.positions[idx]
        base_vel[i] = temp_env.reference.velocities[idx]

    pos = base_pos + rng.normal(loc=0.0, scale=noise_pos, size=(3, 2))
    vel = base_vel + rng.normal(loc=0.0, scale=noise_vel, size=(3, 2))

    pos_t = tuple((float(pos[i, 0]), float(pos[i, 1])) for i in range(3))
    vel_t = tuple((float(vel[i, 0]), float(vel[i, 1])) for i in range(3))
    return pos_t, vel_t


def _model_action(
    obs: np.ndarray,
    model: ActorCritic,
    obs_rms: RunningMeanStd,
    cfg: EnvConfig,
    device: "torch.device",
) -> np.ndarray:
    assert torch is not None
    obs_n = obs_rms.normalize(obs[None, :])
    obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, _, _ = model(obs_t)
    action = mean.squeeze(0).cpu().numpy().reshape(3, cfg.dimensions)
    action = action * cfg.max_action_norm
    return np.clip(action, -cfg.max_action_norm, cfg.max_action_norm).astype(np.float64)


def _heuristic_action(env: Figure8ChoreographyEnv) -> np.ndarray:
    # Simple PD-like corrective controller in the matched choreography frame.
    state = env.sim.get_state()
    match = env._last_match if env._last_match is not None else env._match_choreography(state)
    tgt_pos, tgt_vel = env._targets_for_phase(match.phase_idx)
    assigned_pos = tgt_pos[list(match.assignment)]
    assigned_vel = tgt_vel[list(match.assignment)]

    pos_err = assigned_pos - state.positions
    vel_err = assigned_vel - state.velocities
    action = 1.8 * pos_err + 0.9 * vel_err

    max_norm = env.cfg.max_action_norm
    norms = np.linalg.norm(action, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return (action * scale).astype(np.float64)


def _select_frame_pair(pos_err_hist: list[float], delta_steps: int) -> tuple[int, int]:
    if len(pos_err_hist) <= 1:
        return 0, max(0, len(pos_err_hist) - 1)

    n = len(pos_err_hist)
    best_i, best_j = 0, min(n - 1, delta_steps)
    best_gain = -1e18
    d = max(1, delta_steps)

    for i in range(0, n - 1):
        j = min(n - 1, i + d)
        gain = float(pos_err_hist[i] - pos_err_hist[j])
        if gain > best_gain:
            best_gain = gain
            best_i, best_j = i, j

    return best_i, best_j


def _fuel_cost_from_action(action: np.ndarray, max_action_norm: float) -> float:
    action_sq = np.sum(action**2, axis=1)
    return float(np.mean(action_sq) / (max_action_norm**2 + 1e-12))


def _plot_panel(
    ax: Any,
    reference_path: np.ndarray,
    state_pos: np.ndarray,
    state_vel: np.ndarray,
    push: np.ndarray,
    assigned_targets: np.ndarray,
    step: int,
    pos_err: float,
    vel_err: float,
    fuel_cost: float,
    fuel_penalty: float,
    max_action_norm: float,
    vel_arrow_len: float,
    push_arrow_len: float,
    axis_pad: float,
) -> None:
    ax.plot(reference_path[:, 0], reference_path[:, 1], color="0.65", linewidth=1.8, alpha=0.9)

    colors = ["darkorange", "forestgreen", "royalblue"]
    labels = ["Body 1", "Body 2", "Body 3"]

    all_pts = np.concatenate([reference_path, state_pos, assigned_targets], axis=0)
    x_min, x_max = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    y_min, y_max = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    half = max(half * (1.0 + max(0.0, axis_pad)), 1.2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    for i, (c, lbl) in enumerate(zip(colors, labels)):
        p = state_pos[i]
        t = assigned_targets[i]

        ax.scatter(p[0], p[1], s=70, color=c, edgecolor="black", linewidth=0.6, zorder=4)
        ax.scatter(t[0], t[1], s=55, marker="x", color=c, linewidths=1.7, zorder=4)
        ax.plot([p[0], t[0]], [p[1], t[1]], linestyle="--", color=c, alpha=0.45, linewidth=1.2)

        v_dir = _unit(state_vel[i]) * vel_arrow_len
        ax.annotate(
            "",
            xy=(p[0] + v_dir[0], p[1] + v_dir[1]),
            xytext=(p[0], p[1]),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5, alpha=0.9),
            zorder=5,
        )

        # Keep red arrow lengths proportional to actual action magnitudes.
        a_vec = push[i] * (push_arrow_len / max(1e-12, max_action_norm))
        ax.annotate(
            "",
            xy=(p[0] + a_vec[0], p[1] + a_vec[1]),
            xytext=(p[0], p[1]),
            arrowprops=dict(arrowstyle="->", color="crimson", linewidth=2.2, alpha=0.95),
            zorder=6,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Step {step}")

    txt = f"pos_err={pos_err:.3f}\nvel_err={vel_err:.3f}\nfuel_penalty={fuel_penalty:.4f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.72, edgecolor="0.7"))


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve_ckpt_path(args.checkpoint)

    if ckpt_path is not None:
        if torch is None:
            raise SystemExit("Torch not available but checkpoint was provided.")
        device = choose_device(args.device)
        ckpt = _load_checkpoint(ckpt_path, device=device)
        cfg = EnvConfig(**ckpt["env_config"])
        rw = RewardWeights(**ckpt["reward_weights"])
    else:
        device = None
        ckpt = None
        cfg = EnvConfig()
        rw = RewardWeights()

    fixed_pos, fixed_vel = _build_off_target_fixed_init(cfg=cfg, seed=args.seed, noise_pos=args.noise_pos, noise_vel=args.noise_vel)
    cfg = EnvConfig(**{**asdict(cfg), "fixed_init_positions": fixed_pos, "fixed_init_velocities": fixed_vel})

    env = Figure8ChoreographyEnv(config=cfg, weights=rw)
    obs, info = env.reset(seed=args.seed)

    model = None
    obs_rms = None
    if ckpt is not None:
        obs_dim = env.observation_dim
        action_dim = cfg.num_bodies * cfg.dimensions
        model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])

    frames: list[dict[str, Any]] = []
    done = False
    step = 0
    while not done and step < args.max_steps:
        state = env.sim.get_state()
        phase_idx = int(info["phase_idx"])
        assignment = tuple(int(x) for x in info["assignment"])
        tgt_pos, _ = env._targets_for_phase(phase_idx)
        assigned_targets = tgt_pos[list(assignment)]

        if model is not None and obs_rms is not None and device is not None:
            action = _model_action(obs=obs, model=model, obs_rms=obs_rms, cfg=cfg, device=device)
        else:
            action = _heuristic_action(env)

        fuel_cost = _fuel_cost_from_action(action, max_action_norm=cfg.max_action_norm)
        fuel_penalty = float(rw.fuel * fuel_cost)

        frames.append(
            {
                "step": step,
                "pos": state.positions.copy(),
                "vel": state.velocities.copy(),
                "push": action.copy(),
                "assigned_targets": assigned_targets.copy(),
                "position_error": float(info.get("position_error", np.nan)),
                "velocity_direction_error": float(info.get("velocity_direction_error", np.nan)),
                "fuel_cost": fuel_cost,
                "fuel_penalty": fuel_penalty,
            }
        )

        obs, _, done, info = env.step(action)
        step += 1

    if len(frames) == 0:
        raise SystemExit("No rollout frames generated.")

    pos_err_hist = [float(fr["position_error"]) for fr in frames]
    i0, i1 = _select_frame_pair(pos_err_hist, delta_steps=args.delta_steps)
    fr0, fr1 = frames[i0], frames[i1]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.basename}.png"
    out_pdf = out_dir / f"{args.basename}.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=300)

    _plot_panel(
        ax=axes[0],
        reference_path=env.reference.positions,
        state_pos=fr0["pos"],
        state_vel=fr0["vel"],
        push=fr0["push"],
        assigned_targets=fr0["assigned_targets"],
        step=fr0["step"],
        pos_err=fr0["position_error"],
        vel_err=fr0["velocity_direction_error"],
        fuel_cost=fr0["fuel_cost"],
        fuel_penalty=fr0["fuel_penalty"],
        max_action_norm=cfg.max_action_norm,
        vel_arrow_len=args.vel_arrow_len,
        push_arrow_len=args.push_arrow_len,
        axis_pad=args.axis_pad,
    )
    axes[0].set_title(f"Before Correction (step {fr0['step']})")

    _plot_panel(
        ax=axes[1],
        reference_path=env.reference.positions,
        state_pos=fr1["pos"],
        state_vel=fr1["vel"],
        push=fr1["push"],
        assigned_targets=fr1["assigned_targets"],
        step=fr1["step"],
        pos_err=fr1["position_error"],
        vel_err=fr1["velocity_direction_error"],
        fuel_cost=fr1["fuel_cost"],
        fuel_penalty=fr1["fuel_penalty"],
        max_action_norm=cfg.max_action_norm,
        vel_arrow_len=args.vel_arrow_len,
        push_arrow_len=args.push_arrow_len,
        axis_pad=args.axis_pad,
    )
    axes[1].set_title(f"After Correction (step {fr1['step']})")

    mode = "PPO policy actions" if ckpt_path is not None else "Heuristic controller actions (no checkpoint found)"
    fig.suptitle(f"Pushes Toward Figure-8 Choreography\n{mode}", fontsize=13)
    push_label = "RL model push/action" if ckpt_path is not None else "Controller push/action"
    legend_handles = [
        Line2D([0], [0], color="0.65", linewidth=2.0, label="Figure-8 guide path"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="darkorange", markeredgecolor="black", markersize=7, label="Body positions (orange/green/blue = Body 1/2/3)"),
        Line2D([0], [0], marker="x", linestyle="None", color="darkorange", markersize=7, markeredgewidth=1.8, label="Assigned target slots"),
        Line2D([0], [0], color="darkorange", linestyle="--", linewidth=1.5, label="Position error to target"),
        Line2D([0], [0], color="black", linewidth=1.7, marker=">", markersize=6, label="Velocity direction"),
        Line2D([0], [0], color="crimson", linewidth=2.2, marker=">", markersize=6, label=push_label),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
        ncol=3,
        frameon=True,
        framealpha=0.93,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_png.resolve()}")
    print(f"saved: {out_pdf.resolve()}")
    print(f"mode: {mode}")
    if ckpt_path is not None:
        print(f"checkpoint: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
