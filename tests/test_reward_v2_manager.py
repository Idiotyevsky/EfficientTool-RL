from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("verl")
import torch
from verl import DataProto

from efficienttool_rl.verl.reward_managers.reward_v2 import RewardV2RewardManager


class FixedTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return (
            '<tool_response>{"ok":true,"tool":"search","query":"q",'
            '"results":[{"title":"A"}]}</tool_response>'
            '<answer>London</answer>'
        )


def batch_with_observation_tokens():
    values = {
        "prompts": torch.tensor([[101, 102]]),
        "responses": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        # Positions 2 and 3 are injected observation tokens.
        "response_mask": torch.tensor([[1, 1, 0, 0, 1, 1]]),
        # A deliberately wrong prefilled score proves the manager recomputes.
        "rm_scores": torch.tensor([[0.0, 0.0, 0.0, 99.0, 0.0, 0.0]]),
        "data_source": np.array(["hotpotqa_distractor"], dtype=object),
        "reward_model": np.array(
            [{"style": "rule", "ground_truth": "London"}], dtype=object
        ),
        "extra_info": np.array(
            [{"supporting_titles": ["A", "B"]}], dtype=object
        ),
    }
    batch = DataProto.from_single_dict(values)
    batch.meta_info["global_steps"] = 7
    return batch


def test_manager_uses_global_step_and_never_rewards_observation_tokens():
    captured = {}

    def compute_score(**kwargs):
        captured.update(kwargs)
        return {
            "score": 0.75,
            "answer_reward": 0.70,
            "marginal_evidence_reward": 0.5,
            "current_beta": 0.08,
            "wasted_search_count": 0,
            "wasted_search_penalty": 0.0,
            "total_reward": 0.75,
            "search_count": 1,
            "useful_search_count": 1,
            "unique_support_count": 1,
            "multi_search": 0.0,
            "evidence_gain_per_search": 0.5,
            "repeated_search_count": 0,
        }

    manager = RewardV2RewardManager(
        tokenizer=FixedTokenizer(),
        num_examine=0,
        compute_score=compute_score,
    )
    result = manager(batch_with_observation_tokens(), return_dict=True)
    reward_tensor = result["reward_tensor"]

    assert captured["global_step"] == 7
    assert float(reward_tensor.sum()) == pytest.approx(0.75)
    assert float(reward_tensor[0, 2]) == 0.0
    assert float(reward_tensor[0, 3]) == 0.0
    assert float(reward_tensor[0, 5]) == pytest.approx(0.75)
    assert result["reward_extra_info"]["current_beta"] == [0.08]


def test_manager_requires_response_mask():
    data = batch_with_observation_tokens()
    del data.batch["response_mask"]
    manager = RewardV2RewardManager(
        tokenizer=FixedTokenizer(),
        num_examine=0,
        compute_score=lambda **_: {"score": 0.0},
    )
    with pytest.raises(KeyError, match="response_mask"):
        manager(data)
