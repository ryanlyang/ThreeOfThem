import numpy as np

from choreography_env import Figure8ChoreographyEnv
from config import EnvConfig, RewardWeights


def run_episode(env: Figure8ChoreographyEnv, policy: str, seed: int) -> dict:
    obs, info = env.reset(seed=seed)

    reward_sum = 0.0
    step = 0
    done = False

    while not done:
        if policy == "zero":
            action = np.zeros(env.action_shape, dtype=np.float64)
        elif policy == "random":
            action = np.random.uniform(
                low=-env.cfg.max_action_norm,
                high=env.cfg.max_action_norm,
                size=env.action_shape,
            )
        else:
            raise ValueError(policy)

        obs, reward, done, info = env.step(action)
        reward_sum += reward
        step += 1

    return {
        "steps": step,
        "reward_sum": reward_sum,
        "final_pos_err": info["position_error"],
        "final_vel_dir_err": info["velocity_direction_error"],
        "collided": info["collided"],
        "min_pair_distance": info["min_pair_distance"],
    }


def main() -> None:
    cfg = EnvConfig(
        backend="numpy",
        seed=7,
        horizon_steps=300,
        action_dt=0.05,
        integrator_dt=0.001,
        reference_samples=900,
        phase_search_radius=35,
    )

    w = RewardWeights(
        position=1.0,
        velocity_direction=0.35,
        fuel=0.03,
        collision=60.0,
        permutation_switch=0.15,
        phase_jump=0.01,
    )

    env = Figure8ChoreographyEnv(config=cfg, weights=w)

    print("=== Smoke Test: Figure8ChoreographyEnv (numpy backend) ===")
    for policy in ["zero", "random"]:
        rows = []
        for ep in range(3):
            rows.append(run_episode(env, policy=policy, seed=ep))

        avg_reward = np.mean([r["reward_sum"] for r in rows])
        avg_steps = np.mean([r["steps"] for r in rows])
        avg_pos = np.mean([r["final_pos_err"] for r in rows])
        avg_vel = np.mean([r["final_vel_dir_err"] for r in rows])
        col_rate = np.mean([float(r["collided"]) for r in rows])

        print(f"policy={policy:6s}  avg_steps={avg_steps:6.1f}  avg_reward={avg_reward:9.3f}  "
              f"avg_final_pos_err={avg_pos:7.4f}  avg_final_vel_err={avg_vel:7.4f}  "
              f"collision_rate={col_rate:4.2f}")


if __name__ == "__main__":
    main()
