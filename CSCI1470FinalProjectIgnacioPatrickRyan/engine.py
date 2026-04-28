"""
Engine compatibility layer for CSCI1470FinalProjectIgnacioPatrickRyan.

This file now exposes the simulator/engine API used by the migrated
figure-8 choreography implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from config import EnvConfig, RewardWeights
from choreography_env import Figure8ChoreographyEnv
from simulator import (
    SimulationState,
    NumpyThreeBodySimulator,
    AmuseThreeBodySimulator,
    build_simulator,
)


@dataclass
class EngineStepResult:
    state: SimulationState
    collided: bool
    info: dict[str, Any]


class PhysicsEngine:
    """
    Lightweight engine wrapper for direct simulator stepping.

    This preserves an `engine.py`-centric interface while delegating
    integration details to the migrated simulator backends.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config if config is not None else EnvConfig()
        self.sim = build_simulator(self.cfg)

    def reset(self, seed: Optional[int] = None) -> SimulationState:
        return self.sim.reset(seed=seed)

    def step(self, acceleration: np.ndarray) -> EngineStepResult:
        state, collided, info = self.sim.step(acceleration)
        return EngineStepResult(state=state, collided=collided, info=info)

    def get_state(self) -> SimulationState:
        return self.sim.get_state()


def build_choreography_env(
    config: Optional[EnvConfig] = None,
    reward_weights: Optional[RewardWeights] = None,
) -> Figure8ChoreographyEnv:
    return Figure8ChoreographyEnv(config=config, weights=reward_weights)


__all__ = [
    "EnvConfig",
    "RewardWeights",
    "SimulationState",
    "NumpyThreeBodySimulator",
    "AmuseThreeBodySimulator",
    "Figure8ChoreographyEnv",
    "PhysicsEngine",
    "EngineStepResult",
    "build_simulator",
    "build_choreography_env",
]
