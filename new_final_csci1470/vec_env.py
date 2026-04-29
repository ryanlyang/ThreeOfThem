from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from typing import Any, Iterable

import numpy as np

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights


def _worker(
    remote: Any,
    parent_remote: Any,
    cfg_dict: dict[str, Any],
    rw_dict: dict[str, Any],
    seed: int,
) -> None:
    parent_remote.close()
    cfg = EnvConfig(**cfg_dict)
    rw = RewardWeights(**rw_dict)
    env = Figure8ChoreographyEnv(config=cfg, weights=rw)
    rng = np.random.default_rng(seed)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                reset_seed = data
                if reset_seed is None:
                    reset_seed = int(rng.integers(1, 2**31 - 1))
                obs, info = env.reset(seed=reset_seed)
                remote.send((obs, info))
            elif cmd == "step":
                action = data
                obs, reward, done, info = env.step(action)
                if done:
                    obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
                remote.send((obs, reward, done, info))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass


class SerialVecEnv:
    def __init__(self, cfg: EnvConfig, rw: RewardWeights, num_envs: int, seed: int):
        self.envs = [Figure8ChoreographyEnv(config=cfg, weights=rw) for _ in range(num_envs)]
        self.rng = np.random.default_rng(seed)

    def reset(self, seeds: Iterable[int] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if seeds is None:
            seeds = [int(self.rng.integers(1, 2**31 - 1)) for _ in self.envs]
        obs_list = []
        info_list = []
        for env, seed in zip(self.envs, seeds):
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            if done:
                obs, _ = env.reset(seed=int(self.rng.integers(1, 2**31 - 1)))
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(done_list, dtype=np.float32),
            info_list,
        )

    def close(self) -> None:
        return None


class SubprocVecEnv:
    def __init__(
        self,
        cfg: EnvConfig,
        rw: RewardWeights,
        num_envs: int,
        seed: int,
        start_method: str = "spawn",
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes: list[mp.Process] = []
        cfg_dict = asdict(cfg)
        rw_dict = asdict(rw)

        for idx, (remote, work_remote) in enumerate(zip(self.remotes, self.work_remotes)):
            worker_seed = seed + idx * 1000
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, cfg_dict, rw_dict, worker_seed),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.processes.append(proc)

        self._closed = False

    def reset(self, seeds: Iterable[int] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if seeds is None:
            seeds = [None for _ in self.remotes]
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=5)
        self._closed = True


__all__ = ["SerialVecEnv", "SubprocVecEnv"]
