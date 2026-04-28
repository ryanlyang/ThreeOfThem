from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from config import EnvConfig
from reference_orbit import generate_figure8_reference


def main() -> None:
    cfg = EnvConfig()
    ref = generate_figure8_reference(
        num_samples=cfg.reference_samples,
        period=cfg.reference_period,
        g_const=cfg.gravitational_constant,
    )

    pos = ref.positions
    n = ref.num_samples
    offsets = [int(round(frac * n)) % n for frac in cfg.phase_offsets_fraction]

    frame_stride = 3
    frames = np.arange(0, n, frame_stride)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reference_figure8_animation.gif"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    ax.plot(pos[:, 0], pos[:, 1], color="navy", linewidth=2.0, alpha=0.9, label="Reference figure-8 path")

    colors = ["darkorange", "forestgreen", "purple"]
    labels = ["slot 0 (k)", "slot 1 (k+N/3)", "slot 2 (k+2N/3)"]

    points = []
    trails = []
    trail_len = 40
    for c, label in zip(colors, labels):
        point = ax.scatter([], [], s=90, color=c, edgecolor="black", linewidth=0.6, zorder=5, label=label)
        (trail,) = ax.plot([], [], color=c, linewidth=2.0, alpha=0.55)
        points.append(point)
        trails.append(trail)

    phase_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)

    ax.set_title("Animated Reference Figure-8 Choreography")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", fontsize=9)

    def _trail_segment(arr: np.ndarray, idx: int, length: int) -> np.ndarray:
        start = idx - length
        if start >= 0:
            return arr[start : idx + 1]
        # Wrap-around trail for early frames.
        return np.vstack((arr[start:], arr[: idx + 1]))

    def update(frame_idx: int):
        artists = []
        for i, offset in enumerate(offsets):
            idx = (frame_idx + offset) % n
            p = pos[idx]

            points[i].set_offsets(p[None, :])
            artists.append(points[i])

            slot_path = np.roll(pos, -offset, axis=0)
            tr = _trail_segment(slot_path, frame_idx, trail_len)
            trails[i].set_data(tr[:, 0], tr[:, 1])
            artists.append(trails[i])

        phase_text.set_text(f"phase k = {frame_idx}/{n-1}")
        artists.append(phase_text)
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    writer = PillowWriter(fps=20)
    anim.save(out_path, writer=writer)
    plt.close(fig)

    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
