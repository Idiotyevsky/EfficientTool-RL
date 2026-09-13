from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def _container(name):
    return OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "grpo" / name), resolve=False
    )


def test_reward_v2_config_matches_vanilla_except_reward_and_run_identity():
    vanilla = _container("qwen8b_hotpot_mt_strict.yaml")
    reward_v2 = _container("qwen8b_hotpot_reward_v2.yaml")

    vanilla_reward = vanilla.pop("custom_reward_function")
    reward_v2_reward = reward_v2.pop("custom_reward_function")
    assert vanilla_reward["path"].endswith("/task_reward.py")
    assert reward_v2_reward["path"].endswith("/reward_v2.py")
    assert reward_v2_reward["reward_kwargs"] == {
        "beta_start": 0.10,
        "beta_end": 0.0,
        "wasted_search_lambda": 0.02,
        "total_training_steps": "${trainer.total_training_steps}",
        "require_supporting_titles": True,
    }

    for key in (
        "experiment_name",
        "default_local_dir",
        "rollout_data_dir",
        "validation_data_dir",
    ):
        vanilla["trainer"].pop(key)
        reward_v2["trainer"].pop(key)

    assert reward_v2 == vanilla
