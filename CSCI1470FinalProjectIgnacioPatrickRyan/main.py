"""
Main entrypoint wrapper.

Default behavior runs PPO training with the migrated figure-8 pipeline.
Use the helper scripts directly for smoke/eval/inference workflows.
"""

from train_ppo_figure8 import main


if __name__ == "__main__":
    main()
