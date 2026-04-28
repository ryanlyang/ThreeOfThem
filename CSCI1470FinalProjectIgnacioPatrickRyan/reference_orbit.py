from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Figure8Reference:
    positions: np.ndarray  # [N, 2]
    velocities: np.ndarray  # [N, 2]
    period: float

    @property
    def num_samples(self) -> int:
        return int(self.positions.shape[0])


def canonical_figure8_initial_conditions() -> tuple[np.ndarray, np.ndarray]:
    """
    Canonical equal-mass figure-8 initial conditions in normalized units.

    Values are the standard Chenciner-Montgomery set (G=1, m_i=1).
    """
    x0 = np.array(
        [
            [-0.97000436, 0.24308753],
            [0.97000436, -0.24308753],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    v0 = np.array(
        [
            [0.4662036850, 0.4323657300],
            [0.4662036850, 0.4323657300],
            [-0.93240737, -0.86473146],
        ],
        dtype=np.float64,
    )
    return x0, v0


def _gravity_acceleration(x: np.ndarray, masses: np.ndarray, g_const: float = 1.0) -> np.ndarray:
    n = x.shape[0]
    acc = np.zeros_like(x)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = x[j] - x[i]
            dist = np.linalg.norm(r)
            dist = max(dist, 1e-9)
            acc[i] += g_const * masses[j] * r / (dist**3)
    return acc


def _rk4_step(
    x: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    dt: float,
    g_const: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    def derivs(px: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pv, _gravity_acceleration(px, masses, g_const=g_const)

    k1_x, k1_v = derivs(x, v)
    k2_x, k2_v = derivs(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
    k3_x, k3_v = derivs(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
    k4_x, k4_v = derivs(x + dt * k3_x, v + dt * k3_v)

    x_next = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return x_next, v_next


def _roll_path(pos: np.ndarray, vel: np.ndarray, start_idx: int) -> tuple[np.ndarray, np.ndarray]:
    return np.roll(pos, -start_idx, axis=0), np.roll(vel, -start_idx, axis=0)


def _apply_direction_and_start_convention(
    pos: np.ndarray,
    vel: np.ndarray,
    lookahead: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize to start near upper-left, then move toward lower-right.

    This enforces your requested traversal orientation convention.
    """
    # Upper-left proxy score: y - x.
    start_idx = int(np.argmax(pos[:, 1] - pos[:, 0]))
    pos, vel = _roll_path(pos, vel, start_idx)

    dx = np.mean(np.diff(pos[: lookahead + 1, 0]))
    dy = np.mean(np.diff(pos[: lookahead + 1, 1]))

    # Desired local trend: rightward and downward.
    if not (dx > 0.0 and dy < 0.0):
        pos = pos[::-1].copy()
        vel = -vel[::-1].copy()
        start_idx = int(np.argmax(pos[:, 1] - pos[:, 0]))
        pos, vel = _roll_path(pos, vel, start_idx)

    return pos, vel


def generate_figure8_reference(
    num_samples: int = 900,
    period: float = 6.32591398,
    g_const: float = 1.0,
) -> Figure8Reference:
    """
    Precompute one canonical figure-8 path for a single body.

    The 3-body choreography targets are then phase-shifted versions of this path.
    """
    x, v = canonical_figure8_initial_conditions()
    masses = np.ones(3, dtype=np.float64)

    dt = period / float(num_samples)

    traj_x = np.zeros((num_samples, 3, 2), dtype=np.float64)
    traj_v = np.zeros((num_samples, 3, 2), dtype=np.float64)

    for k in range(num_samples):
        traj_x[k] = x
        traj_v[k] = v
        x, v = _rk4_step(x, v, masses, dt, g_const=g_const)

    # Use body 0 curve as canonical path; choreography slots are phase offsets.
    path_pos = traj_x[:, 0, :]
    path_vel = traj_v[:, 0, :]

    path_pos, path_vel = _apply_direction_and_start_convention(path_pos, path_vel)
    return Figure8Reference(positions=path_pos, velocities=path_vel, period=period)
