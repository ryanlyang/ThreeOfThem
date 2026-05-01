from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import EnvConfig
from reference_orbit import Figure8Reference


@dataclass
class ILQRConfig:
    horizon: int = 36
    iterations: int = 10
    model_substeps: int = 4
    fd_eps_state: float = 1e-4
    fd_eps_action: float = 1e-3
    reg_init: float = 1e-3
    reg_min: float = 1e-6
    reg_max: float = 1e5
    improvement_tol: float = 1e-5

    q_pos: float = 40.0
    q_vel: float = 12.0
    r_action: float = 0.001
    terminal_scale: float = 30.0
    near_collision_weight: float = 20.0
    near_collision_distance: float = 0.20

    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    # Cross-entropy trajectory optimization (open-loop warm-start)
    cem_iters: int = 10
    cem_population: int = 128
    cem_elite_frac: float = 0.12
    cem_init_std_frac: float = 0.45
    cem_min_std_frac: float = 0.04
    cem_smoothing: float = 0.25
    cem_seed: int = 7


@dataclass
class ILQRPlanResult:
    actions: np.ndarray  # [H, action_dim]
    states: np.ndarray  # [H+1, state_dim]
    total_cost: float
    iterations: int
    converged: bool
    regularization: float


class Figure8ILQRMPC:
    """
    iLQR-based receding horizon controller for fixed-start Figure-8 tracking.

    Notes:
    - Planning dynamics are a fast approximation of the simulator (fewer substeps).
    - Reference assignment is provided by the choreography matcher in the env.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        reference: Figure8Reference,
        phase_offsets: Sequence[int],
        expected_phase_stride: int,
        ilqr_cfg: ILQRConfig | None = None,
    ):
        self.env_cfg = env_cfg
        self.reference = reference
        self.phase_offsets = tuple(int(x) for x in phase_offsets)
        self.expected_phase_stride = max(1, int(expected_phase_stride))
        self.ilqr_cfg = ilqr_cfg if ilqr_cfg is not None else ILQRConfig()

        self.n = int(env_cfg.num_bodies)
        self.dim = int(env_cfg.dimensions)
        self.pos_dim = self.n * self.dim
        self.vel_dim = self.n * self.dim
        self.state_dim = self.pos_dim + self.vel_dim
        self.action_dim = self.n * self.dim

        q_diag = np.concatenate(
            [
                np.full(self.pos_dim, float(self.ilqr_cfg.q_pos), dtype=np.float64),
                np.full(self.vel_dim, float(self.ilqr_cfg.q_vel), dtype=np.float64),
            ]
        )
        self.Q = np.diag(q_diag)
        self.Q_terminal = float(self.ilqr_cfg.terminal_scale) * self.Q
        self.R = np.eye(self.action_dim, dtype=np.float64) * float(self.ilqr_cfg.r_action)
        self._identity_state = np.eye(self.state_dim, dtype=np.float64)
        self._identity_action = np.eye(self.action_dim, dtype=np.float64)

        self.regularization = float(self.ilqr_cfg.reg_init)
        self._rng = np.random.default_rng(int(self.ilqr_cfg.cem_seed))

    def state_to_vector(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float64).reshape(self.pos_dim)
        vel = np.asarray(velocities, dtype=np.float64).reshape(self.vel_dim)
        return np.concatenate([pos, vel], axis=0)

    def vector_to_state(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(self.state_dim)
        pos = x[: self.pos_dim].reshape(self.n, self.dim)
        vel = x[self.pos_dim :].reshape(self.n, self.dim)
        return pos, vel

    def _clip_action(self, u: np.ndarray) -> np.ndarray:
        mat = np.asarray(u, dtype=np.float64).reshape(self.n, self.dim).copy()
        max_norm = float(self.env_cfg.max_action_norm)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        factors = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
        mat *= factors
        return mat.reshape(self.action_dim)

    def _gravity_acceleration(self, pos: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(pos, dtype=np.float64)
        g = float(self.env_cfg.gravitational_constant)
        m = float(self.env_cfg.mass_each)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                r = pos[j] - pos[i]
                d = max(float(np.linalg.norm(r)), 1e-9)
                acc[i] += g * m * r / (d**3)
        return acc

    def _rk4_substep(self, pos: np.ndarray, vel: np.ndarray, dt: float, acc_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        def derivs(px: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return pv, self._gravity_acceleration(px) + acc_control

        k1_x, k1_v = derivs(pos, vel)
        k2_x, k2_v = derivs(pos + 0.5 * dt * k1_x, vel + 0.5 * dt * k1_v)
        k3_x, k3_v = derivs(pos + 0.5 * dt * k2_x, vel + 0.5 * dt * k2_v)
        k4_x, k4_v = derivs(pos + dt * k3_x, vel + dt * k3_v)

        pos_n = pos + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        vel_n = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return pos_n, vel_n

    def dynamics_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pos, vel = self.vector_to_state(x)
        u_clipped = self._clip_action(u).reshape(self.n, self.dim)

        substeps = max(1, int(self.ilqr_cfg.model_substeps))
        dt = float(self.env_cfg.action_dt) / float(substeps)
        for _ in range(substeps):
            pos, vel = self._rk4_substep(pos, vel, dt, u_clipped)
        return self.state_to_vector(pos, vel)

    def _targets_for_phase(self, phase_idx: int) -> tuple[np.ndarray, np.ndarray]:
        n_samples = self.reference.num_samples
        pos = np.zeros((self.n, self.dim), dtype=np.float64)
        vel = np.zeros((self.n, self.dim), dtype=np.float64)
        for slot, offset in enumerate(self.phase_offsets):
            idx = (int(phase_idx) + int(offset)) % n_samples
            pos[slot] = self.reference.positions[idx]
            vel[slot] = self.reference.velocities[idx]
        return pos, vel

    def _reference_state(self, phase_idx: int, assignment: Sequence[int]) -> np.ndarray:
        target_pos, target_vel = self._targets_for_phase(phase_idx)
        idx = list(int(i) for i in assignment)
        body_pos = target_pos[idx]
        body_vel = target_vel[idx]
        return self.state_to_vector(body_pos, body_vel)

    def build_reference_trajectory(self, phase_idx: int, assignment: Sequence[int], horizon: int) -> np.ndarray:
        ref = np.zeros((horizon + 1, self.state_dim), dtype=np.float64)
        n_samples = self.reference.num_samples
        phase = int(phase_idx)
        step = int(self.expected_phase_stride)
        for t in range(horizon + 1):
            ref_phase = (phase + t * step) % n_samples
            ref[t] = self._reference_state(ref_phase, assignment)
        return ref

    def _near_collision_cost_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        pos, _ = self.vector_to_state(x)
        grad = np.zeros(self.state_dim, dtype=np.float64)
        cost = 0.0

        d_safe = max(float(self.ilqr_cfg.near_collision_distance), 2.0 * float(self.env_cfg.collision_radius))
        w = float(self.ilqr_cfg.near_collision_weight)
        if w <= 0.0 or d_safe <= 0.0:
            return cost, grad

        g_pos = grad[: self.pos_dim].reshape(self.n, self.dim)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r = pos[i] - pos[j]
                d = max(float(np.linalg.norm(r)), 1e-9)
                if d >= d_safe:
                    continue
                gap = d_safe - d
                cost += 0.5 * w * (gap**2)
                direction = r / d
                # d/dx (0.5*w*(d_safe - d)^2) = -w*(d_safe - d) * d(d)/dx
                c = -w * gap
                g_pos[i] += c * direction
                g_pos[j] -= c * direction
        return cost, grad

    def _stage_cost(self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray) -> float:
        dx = x - x_ref
        quad = 0.5 * float(dx @ (self.Q @ dx)) + 0.5 * float(u @ (self.R @ u))
        c_coll, _ = self._near_collision_cost_and_grad(x)
        return quad + c_coll

    def _terminal_cost(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        dx = x - x_ref
        quad = 0.5 * float(dx @ (self.Q_terminal @ dx))
        c_coll, _ = self._near_collision_cost_and_grad(x)
        return quad + float(self.ilqr_cfg.terminal_scale) * c_coll

    def _stage_derivatives(
        self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dx = x - x_ref
        c_coll, g_coll = self._near_collision_cost_and_grad(x)
        _ = c_coll  # keep structure explicit
        lx = self.Q @ dx + g_coll
        lu = self.R @ u
        lxx = self.Q.copy()
        luu = self.R.copy()
        lux = np.zeros((self.action_dim, self.state_dim), dtype=np.float64)
        return lx, lu, lxx, luu, lux

    def _terminal_derivatives(self, x: np.ndarray, x_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = x - x_ref
        c_coll, g_coll = self._near_collision_cost_and_grad(x)
        _ = c_coll
        vx = self.Q_terminal @ dx + float(self.ilqr_cfg.terminal_scale) * g_coll
        vxx = self.Q_terminal.copy()
        return vx, vxx

    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray, ref_seq: np.ndarray) -> tuple[np.ndarray, float]:
        h = u_seq.shape[0]
        x_seq = np.zeros((h + 1, self.state_dim), dtype=np.float64)
        x_seq[0] = np.asarray(x0, dtype=np.float64).reshape(self.state_dim)
        total = 0.0

        for t in range(h):
            x_t = x_seq[t]
            u_t = self._clip_action(u_seq[t])
            total += self._stage_cost(x_t, u_t, ref_seq[t])
            x_seq[t + 1] = self.dynamics_step(x_t, u_t)

        total += self._terminal_cost(x_seq[h], ref_seq[h])
        return x_seq, float(total)

    def _linearize(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        b = np.zeros((self.state_dim, self.action_dim), dtype=np.float64)

        eps_x = float(self.ilqr_cfg.fd_eps_state)
        eps_u = float(self.ilqr_cfg.fd_eps_action)

        for i in range(self.state_dim):
            dx = self._identity_state[i] * eps_x
            f_p = self.dynamics_step(x + dx, u)
            f_m = self.dynamics_step(x - dx, u)
            a[:, i] = (f_p - f_m) / (2.0 * eps_x)

        for j in range(self.action_dim):
            du = self._identity_action[j] * eps_u
            f_p = self.dynamics_step(x, u + du)
            f_m = self.dynamics_step(x, u - du)
            b[:, j] = (f_p - f_m) / (2.0 * eps_u)

        return a, b

    def shift_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        out = np.zeros_like(actions)
        if actions.shape[0] > 1:
            out[:-1] = actions[1:]
        out[-1] = actions[-1]
        for t in range(out.shape[0]):
            out[t] = self._clip_action(out[t])
        return out

    def optimize_open_loop_sequence(
        self,
        x0: np.ndarray,
        phase_idx: int,
        assignment: Sequence[int],
        horizon: int | None = None,
        init_actions: np.ndarray | None = None,
    ) -> ILQRPlanResult:
        """
        Cross-entropy method (CEM) open-loop optimizer.

        Returns a direct action sequence for the fixed initial state. This is
        used as a warm-start for receding-horizon iLQR MPC.
        """
        h = int(horizon) if horizon is not None else int(self.ilqr_cfg.horizon)
        ref_seq = self.build_reference_trajectory(phase_idx=phase_idx, assignment=assignment, horizon=h)

        if init_actions is not None and tuple(np.asarray(init_actions).shape) == (h, self.action_dim):
            mean = np.asarray(init_actions, dtype=np.float64).copy()
            for t in range(h):
                mean[t] = self._clip_action(mean[t])
        else:
            mean = np.zeros((h, self.action_dim), dtype=np.float64)

        max_action = float(self.env_cfg.max_action_norm)
        init_std = max_action * float(self.ilqr_cfg.cem_init_std_frac)
        min_std = max_action * float(self.ilqr_cfg.cem_min_std_frac)
        std = np.full((h, self.action_dim), init_std, dtype=np.float64)

        pop = max(8, int(self.ilqr_cfg.cem_population))
        elite_k = max(2, int(round(pop * float(self.ilqr_cfg.cem_elite_frac))))
        alpha = float(self.ilqr_cfg.cem_smoothing)
        alpha = min(0.95, max(0.0, alpha))

        best_u = mean.copy()
        best_x, best_cost = self._rollout(x0=x0, u_seq=best_u, ref_seq=ref_seq)

        for _ in range(max(1, int(self.ilqr_cfg.cem_iters))):
            samples = self._rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(pop, h, self.action_dim),
            )
            costs = np.zeros(pop, dtype=np.float64)
            for i in range(pop):
                for t in range(h):
                    samples[i, t] = self._clip_action(samples[i, t])
                _, c = self._rollout(x0=x0, u_seq=samples[i], ref_seq=ref_seq)
                costs[i] = c

            elite_idx = np.argsort(costs)[:elite_k]
            elite = samples[elite_idx]
            elite_cost = float(costs[elite_idx[0]])

            new_mean = np.mean(elite, axis=0)
            new_std = np.std(elite, axis=0)
            mean = (1.0 - alpha) * mean + alpha * new_mean
            std = (1.0 - alpha) * std + alpha * new_std
            std = np.maximum(std, min_std)

            if elite_cost < best_cost:
                best_cost = elite_cost
                best_u = elite[0].copy()
                best_x, _ = self._rollout(x0=x0, u_seq=best_u, ref_seq=ref_seq)

        for t in range(h):
            best_u[t] = self._clip_action(best_u[t])

        return ILQRPlanResult(
            actions=best_u,
            states=best_x,
            total_cost=float(best_cost),
            iterations=max(1, int(self.ilqr_cfg.cem_iters)),
            converged=True,
            regularization=float(self.regularization),
        )

    def plan(
        self,
        x0: np.ndarray,
        phase_idx: int,
        assignment: Sequence[int],
        warm_start: np.ndarray | None = None,
    ) -> ILQRPlanResult:
        h = int(self.ilqr_cfg.horizon)
        ref_seq = self.build_reference_trajectory(phase_idx=phase_idx, assignment=assignment, horizon=h)

        if warm_start is None or tuple(np.asarray(warm_start).shape) != (h, self.action_dim):
            u_seq = np.zeros((h, self.action_dim), dtype=np.float64)
        else:
            u_seq = np.asarray(warm_start, dtype=np.float64).copy()
            for t in range(h):
                u_seq[t] = self._clip_action(u_seq[t])

        x_seq, best_cost = self._rollout(x0=x0, u_seq=u_seq, ref_seq=ref_seq)
        converged = False
        reg = float(self.regularization)
        it_used = 0

        for it in range(int(self.ilqr_cfg.iterations)):
            it_used = it + 1
            a_list = np.zeros((h, self.state_dim, self.state_dim), dtype=np.float64)
            b_list = np.zeros((h, self.state_dim, self.action_dim), dtype=np.float64)
            for t in range(h):
                a_t, b_t = self._linearize(x_seq[t], u_seq[t])
                a_list[t] = a_t
                b_list[t] = b_t

            k_list = np.zeros((h, self.action_dim), dtype=np.float64)
            k_gain = np.zeros((h, self.action_dim, self.state_dim), dtype=np.float64)

            vx, vxx = self._terminal_derivatives(x_seq[h], ref_seq[h])
            backward_ok = True

            for t in range(h - 1, -1, -1):
                lx, lu, lxx, luu, lux = self._stage_derivatives(x_seq[t], u_seq[t], ref_seq[t])
                a_t = a_list[t]
                b_t = b_list[t]

                qx = lx + a_t.T @ vx
                qu = lu + b_t.T @ vx
                qxx = lxx + a_t.T @ vxx @ a_t
                quu = luu + b_t.T @ vxx @ b_t
                qux = lux + b_t.T @ vxx @ a_t

                quu = 0.5 * (quu + quu.T)
                quu_reg = quu + reg * np.eye(self.action_dim, dtype=np.float64)

                try:
                    k = -np.linalg.solve(quu_reg, qu)
                    k_mat = -np.linalg.solve(quu_reg, qux)
                except np.linalg.LinAlgError:
                    backward_ok = False
                    break

                k_list[t] = k
                k_gain[t] = k_mat

                vx = qx + k_mat.T @ quu @ k + k_mat.T @ qu + qux.T @ k
                vxx = qxx + k_mat.T @ quu @ k_mat + k_mat.T @ qux + qux.T @ k_mat
                vxx = 0.5 * (vxx + vxx.T)

            if not backward_ok:
                reg = min(float(self.ilqr_cfg.reg_max), reg * 10.0)
                if reg >= float(self.ilqr_cfg.reg_max):
                    break
                continue

            accepted = False
            old_cost = best_cost
            best_trial_u = u_seq
            best_trial_x = x_seq
            best_trial_cost = best_cost

            for alpha in self.ilqr_cfg.line_search_alphas:
                x_trial = np.zeros_like(x_seq)
                u_trial = np.zeros_like(u_seq)
                x_trial[0] = x0

                for t in range(h):
                    dx = x_trial[t] - x_seq[t]
                    u_nom = u_seq[t] + float(alpha) * k_list[t] + k_gain[t] @ dx
                    u_trial[t] = self._clip_action(u_nom)
                    x_trial[t + 1] = self.dynamics_step(x_trial[t], u_trial[t])

                _, trial_cost = self._rollout(x0=x0, u_seq=u_trial, ref_seq=ref_seq)
                if trial_cost < best_trial_cost:
                    best_trial_u = u_trial
                    best_trial_x = x_trial
                    best_trial_cost = trial_cost
                    accepted = True
                    break

            if accepted:
                u_seq = best_trial_u
                x_seq = best_trial_x
                best_cost = best_trial_cost
                reg = max(float(self.ilqr_cfg.reg_min), reg * 0.5)
                rel_improve = (old_cost - best_cost) / max(1.0, abs(old_cost))
                if rel_improve < float(self.ilqr_cfg.improvement_tol):
                    converged = True
                    break
            else:
                reg = min(float(self.ilqr_cfg.reg_max), reg * 10.0)
                if reg >= float(self.ilqr_cfg.reg_max):
                    break

        self.regularization = reg
        return ILQRPlanResult(
            actions=u_seq,
            states=x_seq,
            total_cost=float(best_cost),
            iterations=int(it_used),
            converged=bool(converged),
            regularization=float(reg),
        )
