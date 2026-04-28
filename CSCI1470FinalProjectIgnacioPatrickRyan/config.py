from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RewardWeights:
    position: float = 1.0
    velocity_direction: float = 0.35
    fuel: float = 0.03
    collision: float = 60.0
    permutation_switch: float = 0.15
    phase_jump: float = 0.01
    out_of_plane: float = 0.0


@dataclass
class EnvConfig:
    num_bodies: int = 3
    dimensions: int = 2
    gravitational_constant: float = 1.0
    mass_each: float = 1.0

    action_dt: float = 0.05
    integrator_dt: float = 0.001
    horizon_steps: int = 600

    max_action_norm: float = 0.30
    collision_radius: float = 0.06

    init_radius_min: float = 0.15
    init_radius_max: float = 1.30
    init_speed_scale: float = 0.35

    reference_period: float = 6.32591398
    reference_samples: int = 900
    phase_offsets_fraction: tuple[float, float, float] = (0.0, 1.0 / 3.0, 2.0 / 3.0)

    phase_search_radius: int = 35
    expected_phase_stride_override: Optional[int] = None

    backend: Literal["numpy", "amuse"] = "numpy"
    integrator_name: Literal["Hermite", "Ph4", "Huayno", "Symple"] = "Hermite"

    seed: int = 7
