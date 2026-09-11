"""Focused tests for assistant-only DAPO overlong accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("verl")
import torch
from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls

import efficienttool_rl.verl.reward_managers  # noqa: F401


class DummyTokenizer:
    eos_token = None

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


def _manager(*, enabled=True):
    cls = get_reward_manager_cls("efficienttool_dapo")
    return cls(
        DummyTokenizer(),
        num_examine=0,
        compute_score=lambda **_: 1.0,
        max_resp_len=8,
        overlong_buffer_cfg=SimpleNamespace(
            enable=enabled,
            len=2,
            penalty_factor=1.0,
            log=True,
        ),
    )


def _batch(response_mask, *, task_score=1.0, tool_rewards=(0.0, 0.0), search_count=None):
    response_length = len(response_mask)
    prompt_ids = [101, 102]
    rm_scores = [0.0] * response_length
    rm_scores[-1] = task_score
    values = {
        "prompts": torch.tensor([prompt_ids]),
        "responses": torch.tensor([list(range(1, response_length + 1))]),
        "attention_mask": torch.tensor([[1] * (len(prompt_ids) + response_length)]),
        "response_mask": torch.tensor([response_mask]),
        "rm_scores": torch.tensor([rm_scores]),
        "data_source": np.array(["hotpotqa_distractor"], dtype=object),
        "reward_model": np.array(
            [{"style": "rule", "ground_truth": "answer"}], dtype=object
        ),
        "extra_info": np.array([{}], dtype=object),
        "tool_rewards": np.array([list(tool_rewards)], dtype=object),
        "task_component": np.array([task_score]),
    }
    if search_count is not None:
        values["executed_search_calls"] = np.array([search_count])
    batch = DataProto.from_single_dict(values)
    batch.meta_info["reward_extra_keys"] = ["task_component"]
    return batch


def test_tool_observations_do_not_trigger_overlong_penalty():
    # Four assistant tokens plus eight tool-observation tokens. The complete
    # trajectory is over max_resp_len=8, but assistant output is below the
    # expected assistant length of 6 and therefore receives no penalty.
    response_mask = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    result = _manager()(_batch(response_mask), return_dict=True)

    assert float(result["reward_tensor"].sum()) == pytest.approx(1.0)
    assert result["reward_extra_info"]["assistant_token_length"] == [4]
    assert result["reward_extra_info"]["total_trajectory_length"] == [12]
    assert result["reward_extra_info"]["overlong_penalty"] == [0.0]
    assert result["reward_extra_info"]["search_count"] == [2]
    assert result["reward_extra_info"]["task_component"] == [pytest.approx(1.0)]
    assert float(result["reward_tensor"][0, 11]) == pytest.approx(1.0)


def test_assistant_tokens_still_trigger_overlong_penalty():
    # expected_len=6; seven assistant tokens produce a -0.5 penalty even
    # though two interleaved observation tokens are ignored by the length term.
    response_mask = [1, 1, 1, 0, 0, 1, 1, 1, 1]
    result = _manager()(_batch(response_mask), return_dict=True)

    assert result["reward_extra_info"]["assistant_token_length"] == [7]
    assert result["reward_extra_info"]["total_trajectory_length"] == [9]
    assert result["reward_extra_info"]["overlong_penalty"] == [pytest.approx(-0.5)]
    assert float(result["reward_tensor"].sum()) == pytest.approx(0.5)
    assert float(result["reward_tensor"][0, 8]) == pytest.approx(0.5)


def test_explicit_executed_search_count_takes_precedence():
    result = _manager()(
        _batch([1, 1], tool_rewards=(0.0,), search_count=3), return_dict=True
    )
    assert result["reward_extra_info"]["search_count"] == [3]


def test_async_preprocessing_computes_task_reward_before_response_mask_exists():
    batch = _batch([1, 1])
    del batch.batch["response_mask"]
    del batch.batch["rm_scores"]
    result = _manager()(batch)
    assert float(result.sum()) == pytest.approx(1.0)


def test_prefilled_scores_require_response_mask_for_overlong_shaping():
    batch = _batch([1, 1])
    del batch.batch["response_mask"]
    with pytest.raises(KeyError, match="prefilled rm_scores without response_mask"):
        _manager()(batch)


def test_async_validation_batch_with_prefilled_scores_needs_no_dataset_fields():
    """AgentLoop validation may drop fields already consumed by task scoring."""
    batch = _batch([1, 1, 0, 0, 1, 1], task_score=0.75)
    del batch.non_tensor_batch["data_source"]
    del batch.non_tensor_batch["reward_model"]

    result = _manager()(batch, return_dict=True)

    assert float(result["reward_tensor"].sum()) == pytest.approx(0.75)
    assert result["reward_extra_info"]["assistant_token_length"] == [4]
    assert result["reward_extra_info"]["total_trajectory_length"] == [6]
    assert result["reward_extra_info"]["overlong_penalty"] == [0.0]
