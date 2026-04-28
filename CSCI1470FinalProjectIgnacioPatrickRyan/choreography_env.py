from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from config import EnvConfig, RewardWeights
from reference_orbit import Figure8Reference, generate_figure8_reference
from simulator import SimulationState, build_simulator


@dataclass
class MatchResult:
    phase_idx: int
    assignment: tuple[int, int, int]
    position_error: float
    velocity_direction_error: float
    switch_cost: float
    phase_jump_cost: float
    total_cost: float


class Figure8ChoreographyEnv:
    """
    Figure-8 choreography environment with:
    - One global orbit phase index k
    - 3 choreography slots at phase offsets k, k+N/3, k+2N/3
    - Permutation matching + continuity penalties
    """

    def __init__(self, config: EnvConfig | None = None, weights: RewardWeights | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.w = weights if weights is not None else RewardWeights()

        self.reference: Figure8Reference = generate_figure8_reference(
            num_samples=self.cfg.reference_samples,
            period=self.cfg.reference_period,
            g_const=self.cfg.gravitational_constant,
        )
        self.phase_offsets = self._build_phase_offsets()

        self.expected_phase_stride = (
            self.cfg.expected_phase_stride_override
            if self.cfg.expected_phase_stride_override is not None
            else max(1, int(round(self.reference.num_samples * self.cfg.action_dt / self.reference.period)))
        )

        self.sim = build_simulator(self.cfg)

        self._all_assignments = list(permutations(range(3)))
        self._prev_phase_idx: int | None = None
        self._prev_assignment: tuple[int, int, int] | None = None
        self._last_match: MatchResult | None = None

        self.steps = 0

    def _build_phase_offsets(self) -> tuple[int, int, int]:
        n = self.reference.num_samples
        offsets = []
        for frac in self.cfg.phase_offsets_fraction:
            offsets.append(int(round(frac * n)) % n)
        return tuple(offsets)  # type: ignore[return-value]

    def _targets_for_phase(self, phase_idx: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.reference.num_samples
        target_pos = np.zeros((3, 2), dtype=np.float64)
        target_vel = np.zeros((3, 2), dtype=np.float64)

        for slot, offset in enumerate(self.phase_offsets):
            idx = (phase_idx + offset) % n
            target_pos[slot] = self.reference.positions[idx]
            target_vel[slot] = self.reference.velocities[idx]

        return target_pos, target_vel

    @staticmethod
    def _cosine_direction_error(v: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
        v_norm = np.linalg.norm(v, axis=1)
        v_ref_norm = np.linalg.norm(v_ref, axis=1)
        denom = np.maximum(v_norm * v_ref_norm, 1e-12)
        cos_sim = np.sum(v * v_ref, axis=1) / denom
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return 1.0 - cos_sim

    @staticmethod
    def _cyclic_distance(a: int, b: int, n: int) -> int:
        d = abs(a - b)
        return min(d, n - d)

    @staticmethod
    def _hamming_fraction(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        return sum(int(x != y) for x, y in zip(a, b)) / 3.0

    def _phase_candidates(self) -> np.ndarray:
        n = self.reference.num_samples
        if self._prev_phase_idx is None:
            return np.arange(n, dtype=np.int32)

        expected = (self._prev_phase_idx + self.expected_phase_stride) % n
        rad = self.cfg.phase_search_radius

        return np.array([(expected + d) % n for d in range(-rad, rad + 1)], dtype=np.int32)

    def _match_choreography(self, state: SimulationState) -> MatchResult:
        positions = state.positions
        velocities = state.velocities

        n = self.reference.num_samples
        candidates = self._phase_candidates()

        prev_assignment = self._prev_assignment
        prev_phase = self._prev_phase_idx
        expected_phase = None if prev_phase is None else (prev_phase + self.expected_phase_stride) % n

        best: MatchResult | None = None

        for phase_idx in candidates:
            phase_idx = int(phase_idx)
            target_pos, target_vel = self._targets_for_phase(phase_idx)

            for assignment in self._all_assignments:
                # assignment[i] = slot index used for body i.
                assigned_pos = target_pos[list(assignment)]
                assigned_vel = target_vel[list(assignment)]

                pos_err = float(np.mean(np.sum((positions - assigned_pos) ** 2, axis=1)))
                vel_err = float(np.mean(self._cosine_direction_error(velocities, assigned_vel)))

                switch_cost = 0.0
                if prev_assignment is not None:
                    switch_cost = self._hamming_fraction(assignment, prev_assignment)

                phase_jump_cost = 0.0
                if expected_phase is not None:
                    phase_jump_cost = self._cyclic_distance(phase_idx, expected_phase, n) / float(n)

                total = (
                    self.w.position * pos_err
                    + self.w.velocity_direction * vel_err
                    + self.w.permutation_switch * switch_cost
                    + self.w.phase_jump * phase_jump_cost
                )

                cand = MatchResult(
                    phase_idx=phase_idx,
                    assignment=assignment,
                    position_error=pos_err,
                    velocity_direction_error=vel_err,
                    switch_cost=switch_cost,
                    phase_jump_cost=phase_jump_cost,
                    total_cost=total,
                )

                if best is None or cand.total_cost < best.total_cost:
                    best = cand

        assert best is not None
        return best

    def _observation_from_state(self, state: SimulationState, phase_idx: int | None) -> np.ndarray:
        flat_pos = state.positions.reshape(-1)
        flat_vel = state.velocities.reshape(-1)
        masses = state.masses.reshape(-1)

        n = self.reference.num_samples
        if phase_idx is None:
            phase_sin, phase_cos = 0.0, 1.0
        else:
            theta = 2.0 * np.pi * (phase_idx / float(n))
            phase_sin, phase_cos = np.sin(theta), np.cos(theta)

        obs = np.concatenate([flat_pos, flat_vel, masses, np.array([phase_sin, phase_cos], dtype=np.float64)])
        return obs.astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.sim.reset(seed=seed)

        self._prev_phase_idx = None
        self._prev_assignment = None
        self._last_match = None
        self.steps = 0

        match = self._match_choreography(state)
        self._prev_phase_idx = match.phase_idx
        self._prev_assignment = match.assignment
        self._last_match = match

        obs = self._observation_from_state(state, phase_idx=match.phase_idx)
        info = {
            "phase_idx": match.phase_idx,
            "assignment": match.assignment,
            "position_error": match.position_error,
            "velocity_direction_error": match.velocity_direction_error,
            "sim_time": state.time,
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3, self.cfg.dimensions):
            raise ValueError(f"Expected action shape {(3, self.cfg.dimensions)}, got {action.shape}")

        self.steps += 1

        state, collided, sim_info = self.sim.step(action)
        match = self._match_choreography(state)

        action_sq = np.sum(action**2, axis=1)
        fuel_cost = float(np.mean(action_sq) / (self.cfg.max_action_norm**2 + 1e-12))

        collision_cost = 1.0 if collided else 0.0

        reward = -(
            self.w.position * match.position_error
            + self.w.velocity_direction * match.velocity_direction_error
            + self.w.fuel * fuel_cost
            + self.w.collision * collision_cost
            + self.w.permutation_switch * match.switch_cost
            + self.w.phase_jump * match.phase_jump_cost
        )

        self._prev_phase_idx = match.phase_idx
        self._prev_assignment = match.assignment
        self._last_match = match

        terminated = collided or (self.steps >= self.cfg.horizon_steps)

        obs = self._observation_from_state(state, phase_idx=match.phase_idx)
        info = {
            "phase_idx": match.phase_idx,
            "assignment": match.assignment,
            "position_error": match.position_error,
            "velocity_direction_error": match.velocity_direction_error,
            "switch_cost": match.switch_cost,
            "phase_jump_cost": match.phase_jump_cost,
            "fuel_cost": fuel_cost,
            "collision_cost": collision_cost,
            "collided": collided,
            "min_pair_distance": sim_info.get("min_pair_distance", np.nan),
            "energy": sim_info.get("energy", np.nan),
            "sim_time": state.time,
        }

        return obs, float(reward), bool(terminated), info

    @property
    def observation_dim(self) -> int:
        # 3*2 position + 3*2 velocity + 3 masses + phase sin/cos
        return 6 + 6 + 3 + 2

    @property
    def action_shape(self) -> tuple[int, int]:
        return (3, self.cfg.dimensions)
