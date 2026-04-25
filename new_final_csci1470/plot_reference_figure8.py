from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import EnvConfig
from reference_orbit import generate_figure8_reference


def main() -> None:
    cfg = EnvConfig()
    ref = generate_figure8_reference(
        num_samples=cfg.reference_samples,
        period=cfg.reference_period,
        g_const=cfg.gravitational_constant,
    )

    n = ref.num_samples
    pos = ref.positions

    # Choreography slots at a single global phase k=0.
    offsets = [int(round(frac * n)) % n for frac in cfg.phase_offsets_fraction]
    slot_pts = pos[offsets]

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reference_figure8.png"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)

    # Full canonical orbit curve.
    ax.plot(pos[:, 0], pos[:, 1], color="navy", linewidth=2.0, alpha=0.95, label="Canonical figure-8 path")

    # Start point (upper-left by convention) and a small early segment.
    ax.scatter(pos[0, 0], pos[0, 1], s=70, color="crimson", zorder=5, label="Start (k=0)")
    early = np.arange(0, 25)
    ax.plot(pos[early, 0], pos[early, 1], color="crimson", linewidth=2.8, alpha=0.8, label="Initial traversal")

    # Direction arrows sampled along the orbit.
    arrow_idx = np.linspace(0, n - 2, 12, dtype=int)
    dxy = pos[arrow_idx + 1] - pos[arrow_idx]
    ax.quiver(
        pos[arrow_idx, 0],
        pos[arrow_idx, 1],
        dxy[:, 0],
        dxy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        color="steelblue",
        alpha=0.9,
    )

    # Show choreography target slots for one phase instant.
    slot_colors = ["darkorange", "forestgreen", "purple"]
    for i, (idx, p, c) in enumerate(zip(offsets, slot_pts, slot_colors)):
        ax.scatter(p[0], p[1], s=85, color=c, edgecolor="black", linewidth=0.6, zorder=6)
        ax.text(p[0] + 0.03, p[1] + 0.03, f"slot {i} (k+{idx})", fontsize=9, color=c)

    ax.set_title("Reference Figure-8 Guide Orbit (Precomputed)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
