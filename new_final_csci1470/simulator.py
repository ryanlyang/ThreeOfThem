from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import EnvConfig


def _min_pair_distance(points: np.ndarray) -> float:
    n = points.shape[0]
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            min_dist = min(min_dist, d)
    return float(min_dist)


def _resolve_fixed_state(cfg: EnvConfig, n: int, dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    pos_cfg = cfg.fixed_init_positions
    vel_cfg = cfg.fixed_init_velocities
    if pos_cfg is None and vel_cfg is None:
        return None
    if (pos_cfg is None) != (vel_cfg is None):
        raise ValueError("Both fixed_init_positions and fixed_init_velocities must be set together.")

    pos = np.asarray(pos_cfg, dtype=np.float64)
    vel = np.asarray(vel_cfg, dtype=np.float64)
    expected = (n, dim)
    if pos.shape != expected or vel.shape != expected:
        raise ValueError(
            f"Fixed init state must have shape {expected}; "
            f"got positions={pos.shape}, velocities={vel.shape}"
        )
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        raise ValueError("Fixed init positions/velocities must be finite.")
    return pos, vel


@dataclass
class SimulationState:
    positions: np.ndarray  # [3, D]
    velocities: np.ndarray  # [3, D]
    masses: np.ndarray  # [3]
    time: float


class NumpyThreeBodySimulator:
    """
    Minimal 3-body simulator with continuous acceleration control.

    Uses a fixed-action macro step (action_dt) and RK4 sub-steps (integrator_dt).
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.dim = config.dimensions
        self.n = config.num_bodies

        self.masses = np.full(self.n, config.mass_each, dtype=np.float64)
        self.radii = np.full(self.n, config.collision_radius, dtype=np.float64)

        self.positions = np.zeros((self.n, self.dim), dtype=np.float64)
        self.velocities = np.zeros((self.n, self.dim), dtype=np.float64)
        self.time = 0.0

    def _sample_initial_positions(self) -> np.ndarray:
        best_pos: np.ndarray | None = None
        best_min_dist = -np.inf
        tries = max(1, int(self.cfg.init_sample_tries))

        for _ in range(tries):
            radii = self.rng.uniform(self.cfg.init_radius_min, self.cfg.init_radius_max, size=(self.n, 1))
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.n, 1))
            pos = np.concatenate((radii * np.cos(angles), radii * np.sin(angles)), axis=1)

            min_dist = _min_pair_distance(pos)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos

            if min_dist >= self.cfg.init_min_pair_distance:
                return pos

        assert best_pos is not None
        return best_pos

    def reset(self, seed: int | None = None) -> SimulationState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        fixed_state = _resolve_fixed_state(self.cfg, self.n, self.dim)
        if fixed_state is not None:
            pos, vel = fixed_state
            self.positions = pos.copy()
            self.velocities = vel.copy()
            self.time = 0.0
            return self.get_state()

        # Random positions in an annulus, with minimum pairwise separation.
        self.positions = self._sample_initial_positions()

        # Random initial velocity with near-zero center-of-mass momentum.
        self.velocities = self.rng.normal(0.0, self.cfg.init_speed_scale, size=(self.n, self.dim))
        com_vel = np.average(self.velocities, axis=0, weights=self.masses)
        self.velocities -= com_vel

        # Shift center of mass to origin.
        com_pos = np.average(self.positions, axis=0, weights=self.masses)
        self.positions -= com_pos

        self.time = 0.0
        return self.get_state()

    def _gravity_acceleration(self, x: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                r = x[j] - x[i]
                dist = np.linalg.norm(r)
                dist = max(dist, 1e-9)
                acc[i] += self.cfg.gravitational_constant * self.masses[j] * r / (dist**3)
        return acc

    def _dynamics(self, x: np.ndarray, v: np.ndarray, a_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = v
        dv = self._gravity_acceleration(x) + a_control
        return dx, dv

    def _rk4_step(self, x: np.ndarray, v: np.ndarray, dt: float, a_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k1_x, k1_v = self._dynamics(x, v, a_control)
        k2_x, k2_v = self._dynamics(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v, a_control)
        k3_x, k3_v = self._dynamics(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v, a_control)
        k4_x, k4_v = self._dynamics(x + dt * k3_x, v + dt * k3_v, a_control)

        x_next = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return x_next, v_next

    def _check_collision(self) -> tuple[bool, float]:
        min_dist = np.inf
        collided = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                min_dist = min(min_dist, d)
                if d < (self.radii[i] + self.radii[j]):
                    collided = True
        return collided, min_dist

    def _total_energy(self) -> float:
        kinetic = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        potential = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = np.linalg.norm(self.positions[j] - self.positions[i])
                d = max(d, 1e-9)
                potential -= self.cfg.gravitational_constant * self.masses[i] * self.masses[j] / d
        return float(kinetic + potential)

    def step(self, acceleration: np.ndarray) -> tuple[SimulationState, bool, dict[str, Any]]:
        acceleration = np.asarray(acceleration, dtype=np.float64)
        if acceleration.shape != (self.n, self.dim):
            raise ValueError(f"Expected acceleration shape {(self.n, self.dim)}, got {acceleration.shape}")

        # Clip each action vector to max norm.
        norms = np.linalg.norm(acceleration, axis=1, keepdims=True)
        factors = np.minimum(1.0, self.cfg.max_action_norm / np.maximum(norms, 1e-12))
        acceleration = acceleration * factors

        substeps = max(1, int(round(self.cfg.action_dt / self.cfg.integrator_dt)))
        dt = self.cfg.action_dt / float(substeps)

        for _ in range(substeps):
            self.positions, self.velocities = self._rk4_step(self.positions, self.velocities, dt, acceleration)

        self.time += self.cfg.action_dt

        collided, min_dist = self._check_collision()
        state = self.get_state()
        info = {
            "collided": collided,
            "min_pair_distance": min_dist,
            "energy": self._total_energy(),
        }
        return state, collided, info

    def get_state(self) -> SimulationState:
        return SimulationState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
            time=float(self.time),
        )


class AmuseThreeBodySimulator:
    """
    AMUSE backend in the style of ThreeBodyProblem_astronomy/env/ThreeBP_env.py.

    Notes:
    - This backend requires AMUSE to be installed locally.
    - Action is applied as a macro-step velocity kick: dv = a * action_dt.
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        try:
            from amuse.units import units, nbody_system
            from amuse.community.hermite.interface import Hermite
            from amuse.community.ph4.interface import ph4
            from amuse.community.huayno.interface import Huayno
            from amuse.community.symple.interface import symple
            from amuse.lab import Particles
        except Exception as exc:  # pragma: no cover - AMUSE not available in this environment
            raise ImportError(
                "AMUSE backend requested but AMUSE is not installed. "
                "Install AMUSE or use backend='numpy'."
            ) from exc

        self.units = units
        self.nbody_system = nbody_system
        self.Particles = Particles
        self._integrator_classes = {
            "Hermite": Hermite,
            "Ph4": ph4,
            "Huayno": Huayno,
            "Symple": symple,
        }

        self.n = config.num_bodies
        self.dim = config.dimensions
        self.masses = np.full(self.n, config.mass_each, dtype=np.float64)
        self.radii = np.full(self.n, config.collision_radius, dtype=np.float64)

        self.gravity = None
        self.channel = None
        self.particles = None

        self.time = 0.0

    def _build_integrator(self):
        cls = self._integrator_classes[self.cfg.integrator_name]
        converter = self.nbody_system.nbody_to_si(self.n * self.cfg.mass_each | self.units.MSun, 1 | self.units.AU)

        if self.cfg.integrator_name == "Hermite":
            return cls(converter)
        if self.cfg.integrator_name == "Ph4":
            return cls(converter, number_of_workers=1)
        if self.cfg.integrator_name == "Huayno":
            return cls(converter)
        integ = cls(converter, redirection="none")
        integ.initialize_code()
        return integ

    def _sample_initial_positions(self) -> np.ndarray:
        best_pos: np.ndarray | None = None
        best_min_dist = -np.inf
        tries = max(1, int(self.cfg.init_sample_tries))

        for _ in range(tries):
            radii = self.rng.uniform(self.cfg.init_radius_min, self.cfg.init_radius_max, size=(self.n, 1))
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.n, 1))
            pos = np.concatenate((radii * np.cos(angles), radii * np.sin(angles)), axis=1)

            min_dist = _min_pair_distance(pos)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos

            if min_dist >= self.cfg.init_min_pair_distance:
                return pos

        assert best_pos is not None
        return best_pos

    def reset(self, seed: int | None = None) -> SimulationState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.gravity is not None:
            self.gravity.stop()
            self.gravity = None

        self.gravity = self._build_integrator()

        parts = self.Particles(self.n)
        parts.mass = np.full(self.n, self.cfg.mass_each) | self.units.MSun

        fixed_state = _resolve_fixed_state(self.cfg, self.n, self.dim)
        if fixed_state is None:
            pos = self._sample_initial_positions()
            vel = self.rng.normal(0.0, self.cfg.init_speed_scale, size=(self.n, self.dim))
            vel -= np.average(vel, axis=0)
            pos -= np.average(pos, axis=0)
        else:
            pos, vel = fixed_state

        for i in range(self.n):
            parts[i].position = (pos[i, 0], pos[i, 1], 0.0) | self.units.AU
            # acceleration unit ~ AU / yr^2 => velocity unit ~ AU / yr
            parts[i].velocity = (vel[i, 0], vel[i, 1], 0.0) | (self.units.AU / self.units.yr)

        self.particles = parts
        self.gravity.particles.add_particles(self.particles)
        self.channel = self.gravity.particles.new_channel_to(self.particles)

        self.time = 0.0
        return self.get_state()

    def _check_collision(self, state: SimulationState) -> tuple[bool, float]:
        min_dist = np.inf
        collided = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(state.positions[i] - state.positions[j]))
                min_dist = min(min_dist, d)
                if d < (self.radii[i] + self.radii[j]):
                    collided = True
        return collided, min_dist

    def step(self, acceleration: np.ndarray) -> tuple[SimulationState, bool, dict[str, Any]]:
        acceleration = np.asarray(acceleration, dtype=np.float64)
        if acceleration.shape != (self.n, self.dim):
            raise ValueError(f"Expected acceleration shape {(self.n, self.dim)}, got {acceleration.shape}")

        norms = np.linalg.norm(acceleration, axis=1, keepdims=True)
        factors = np.minimum(1.0, self.cfg.max_action_norm / np.maximum(norms, 1e-12))
        acceleration = acceleration * factors

        # Apply control as velocity kick dv = a * dt.
        dv = acceleration * self.cfg.action_dt
        for i in range(self.n):
            self.gravity.particles[i].vx += dv[i, 0] | (self.units.AU / self.units.yr)
            self.gravity.particles[i].vy += dv[i, 1] | (self.units.AU / self.units.yr)

        self.time += self.cfg.action_dt
        self.gravity.evolve_model(self.time | self.units.yr)
        self.channel.copy()

        state = self.get_state()
        collided, min_dist = self._check_collision(state)
        info = {"collided": collided, "min_pair_distance": min_dist, "energy": np.nan}
        return state, collided, info

    def get_state(self) -> SimulationState:
        pos = np.zeros((self.n, self.dim), dtype=np.float64)
        vel = np.zeros((self.n, self.dim), dtype=np.float64)

        for i in range(self.n):
            pos[i, 0] = self.particles[i].x.value_in(self.units.AU)
            pos[i, 1] = self.particles[i].y.value_in(self.units.AU)
            vel[i, 0] = self.particles[i].vx.value_in(self.units.AU / self.units.yr)
            vel[i, 1] = self.particles[i].vy.value_in(self.units.AU / self.units.yr)

        return SimulationState(
            positions=pos,
            velocities=vel,
            masses=self.masses.copy(),
            time=float(self.time),
        )


def build_simulator(config: EnvConfig) -> NumpyThreeBodySimulator | AmuseThreeBodySimulator:
    if config.backend == "amuse":
        return AmuseThreeBodySimulator(config)
    return NumpyThreeBodySimulator(config)
