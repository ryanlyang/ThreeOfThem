from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict evaluation + GIF for a fixed-init checkpoint (single deterministic initial state)."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--num-setups", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=360)
    p.add_argument("--pos-threshold", type=float, default=0.08)
    p.add_argument("--vel-threshold", type=float, default=0.12)
    p.add_argument("--consecutive-converged", type=int, default=240)
    p.add_argument("--min-total-steps-for-converged", type=int, default=300)
    p.add_argument("--trail-len", type=int, default=70)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--axis-pad", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--override-fixed-jitter-zero", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    ckpt_path = Path(args.checkpoint).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else ckpt_path.parent / "inference_fixed_init"
    outdir.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable,
        str(root / "evaluate_trained_policy.py"),
        "--checkpoint",
        str(ckpt_path),
        "--episodes",
        str(args.episodes),
        "--policy",
        "deterministic",
        "--compare-baselines",
        "--device",
        args.device,
    ]

    infer_cmd = [
        sys.executable,
        str(root / "run_inference_best_gif.py"),
        "--checkpoint",
        str(ckpt_path),
        "--num-setups",
        str(args.num_setups),
        "--max-steps",
        str(args.max_steps),
        "--policy",
        "deterministic",
        "--pos-threshold",
        str(args.pos_threshold),
        "--vel-threshold",
        str(args.vel_threshold),
        "--consecutive-converged",
        str(args.consecutive_converged),
        "--strict-mode",
        "--lock-to-end",
        "--require-no-failure",
        "--require-final-threshold",
        "--min-total-steps-for-converged",
        str(args.min_total_steps_for_converged),
        "--trail-len",
        str(args.trail_len),
        "--frame-stride",
        str(args.frame_stride),
        "--axis-pad",
        str(args.axis_pad),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--outdir",
        str(outdir),
    ]
    if args.override_fixed_jitter_zero:
        infer_cmd.append("--override-fixed-jitter-zero")

    print("Running evaluation:", flush=True)
    print(" ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True)

    print("Running inference + GIF:", flush=True)
    print(" ".join(infer_cmd), flush=True)
    subprocess.run(infer_cmd, check=True)

    print(f"Done. Inference outputs: {outdir}", flush=True)


if __name__ == "__main__":
    main()
