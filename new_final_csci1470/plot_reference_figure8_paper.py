from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import EnvConfig
from reference_orbit import generate_figure8_reference


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def main() -> None:
    p = argparse.ArgumentParser(description="Create a paper-style figure-8 setup plot with body velocity arrows.")
    p.add_argument("--phase-index", type=int, default=0, help="Global phase index k used to place the 3 bodies.")
    p.add_argument("--arrow-len", type=float, default=0.22, help="Length of normalized velocity arrows in plot units.")
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--basename", type=str, default="reference_figure8_paper")
    args = p.parse_args()

    cfg = EnvConfig()
    ref = generate_figure8_reference(
        num_samples=cfg.reference_samples,
        period=cfg.reference_period,
        g_const=cfg.gravitational_constant,
    )

    pos = ref.positions
    vel = ref.velocities
    n = ref.num_samples

    offsets = [int(round(frac * n)) % n for frac in cfg.phase_offsets_fraction]
    k = args.phase_index % n

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.basename}.png"
    out_pdf = out_dir / f"{args.basename}.pdf"

    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=300)
    ax.plot(pos[:, 0], pos[:, 1], color="black", linewidth=1.6, alpha=0.75)

    colors = ["darkorange", "forestgreen", "royalblue"]
    labels = ["Body 1", "Body 2", "Body 3"]

    for color, label, offset in zip(colors, labels, offsets):
        idx = (k + offset) % n
        pxy = pos[idx]
        vxy = _unit(vel[idx]) * args.arrow_len

        ax.scatter(pxy[0], pxy[1], s=62, color=color, edgecolor="black", linewidth=0.6, zorder=4, label=label)
        ax.annotate(
            "",
            xy=(pxy[0] + vxy[0], pxy[1] + vxy[1]),
            xytext=(pxy[0], pxy[1]),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.8),
            zorder=5,
        )

    min_x, max_x = float(np.min(pos[:, 0])), float(np.max(pos[:, 0]))
    min_y, max_y = float(np.min(pos[:, 1])), float(np.max(pos[:, 1]))
    span = max(max_x - min_x, max_y - min_y)
    pad = 0.16 * span
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    ax.set_title("Reference Figure-8 Choreography Setup")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_png.resolve()}")
    print(f"saved: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
