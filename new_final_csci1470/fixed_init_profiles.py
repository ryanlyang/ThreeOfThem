from __future__ import annotations

from typing import Optional


# A deterministic, asymmetric start condition built from a perturbed figure-8
# phase-aligned state. This is intentionally "weird", but still stable enough
# for a short proof-of-concept training run.
WEIRD_FIXED_POSITIONS_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (0.504745, -0.379373),
    (-1.216997, 0.068041),
    (0.842252, 0.131331),
)

WEIRD_FIXED_VELOCITIES_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.995967, 0.588732),
    (0.094886, -0.407093),
    (0.951081, -0.231639),
)


# Directed near-reference profile for easier fixed-init convergence. These
# values are an exact phase-aligned Figure-8 state from reference_orbit.py.
NEAR_REF_FIXED_POSITIONS_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.97956791, 0.23390607),
    (0.01966620, 0.01822763),
    (0.95990170, -0.25213373),
)

NEAR_REF_FIXED_VELOCITIES_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.44099955, -0.43833147),
    (0.93313223, 0.86381656),
    (-0.49213266, -0.42548510),
)


# Fixed recovery test profile: deliberately off the exact Figure-8, but not on
# an immediate collision trajectory. With zero thrust this state survives the
# episode but drifts away from the target choreography.
OFFSET_REF_FIXED_POSITIONS_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.92956791, 0.33390607),
    (-0.08033380, -0.01177237),
    (1.00990170, -0.32213373),
)

OFFSET_REF_FIXED_VELOCITIES_2D: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.39099955, -0.47833147),
    (0.85313223, 0.90381656),
    (-0.46213266, -0.42548510),
)


def parse_points_2d(spec: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Parse 'x1,y1;x2,y2;x3,y3' into a 3x2 tuple.
    """
    chunks = [s.strip() for s in spec.split(";") if s.strip()]
    if len(chunks) != 3:
        raise ValueError(
            "Expected exactly 3 points in 'x1,y1;x2,y2;x3,y3' format "
            f"(got {len(chunks)} from: {spec!r})"
        )

    out: list[tuple[float, float]] = []
    for c in chunks:
        xy = [s.strip() for s in c.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Expected point 'x,y', got: {c!r}")
        out.append((float(xy[0]), float(xy[1])))

    return (out[0], out[1], out[2])


def resolve_fixed_init(
    profile: str,
    positions_spec: str,
    velocities_spec: str,
) -> tuple[
    Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
]:
    """
    Resolve fixed-init settings from profile + optional explicit specs.
    """
    profile = profile.lower().strip()
    pos_spec = positions_spec.strip()
    vel_spec = velocities_spec.strip()

    if profile == "none":
        if bool(pos_spec) != bool(vel_spec):
            raise ValueError("Provide both --fixed-init-positions and --fixed-init-velocities, or neither.")
        if pos_spec and vel_spec:
            return parse_points_2d(pos_spec), parse_points_2d(vel_spec)
        return None, None

    if profile == "weird":
        pos = WEIRD_FIXED_POSITIONS_2D
        vel = WEIRD_FIXED_VELOCITIES_2D
        if pos_spec:
            pos = parse_points_2d(pos_spec)
        if vel_spec:
            vel = parse_points_2d(vel_spec)
        return pos, vel

    if profile == "near_ref":
        pos = NEAR_REF_FIXED_POSITIONS_2D
        vel = NEAR_REF_FIXED_VELOCITIES_2D
        if pos_spec:
            pos = parse_points_2d(pos_spec)
        if vel_spec:
            vel = parse_points_2d(vel_spec)
        return pos, vel

    if profile == "offset_ref":
        pos = OFFSET_REF_FIXED_POSITIONS_2D
        vel = OFFSET_REF_FIXED_VELOCITIES_2D
        if pos_spec:
            pos = parse_points_2d(pos_spec)
        if vel_spec:
            vel = parse_points_2d(vel_spec)
        return pos, vel

    raise ValueError(f"Unknown fixed init profile: {profile!r}")
