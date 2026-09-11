"""Unit tests for the project-local DAPO reward-manager wiring.

These tests exercise the exact reward path used by ``scripts/train_dapo.py``:
verl's DAPO reward manager calling the project custom reward function, plus
the soft overlong punishment from ``reward_model.overlong_buffer``.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("verl")
import torch
from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls

import efficienttool_rl.verl.reward_managers  # noqa: F401

_ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "efficienttool_rl"
    / "verl"
    / "reward_adapters"
    / "task_reward.py"
)

QWEN3_8B = os.environ.get("ETRL_MODEL", "/home/nfs05/model/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(QWEN3_8B, "tokenizer_config.json")),
    reason="local Qwen3 tokenizer not available",
)


def _adapter_compute_score():
    """Load the reward adapter exactly the way verl loads custom functions."""
    spec = importlib.util.spec_from_file_location("etrl_task_reward_adapter", _ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def _overlong_cfg(enable: bool) -> SimpleNamespace:
    return SimpleNamespace(enable=enable, len=8, penalty_factor=1.0, log=True)


def _manager(tokenizer, enable: bool):
    cls = get_reward_manager_cls("efficienttool_dapo")
    return cls(
        tokenizer,
        num_examine=0,
        compute_score=_adapter_compute_score(),
        max_resp_len=64,
        overlong_buffer_cfg=_overlong_cfg(enable),
    )


def _batch(response_text: str, tokenizer) -> DataProto:
    prompt_ids = tokenizer.encode("user question", add_special_tokens=True)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    return DataProto.from_single_dict(
        {
            "prompts": torch.tensor([prompt_ids]),
            "responses": torch.tensor([response_ids]),
            "attention_mask": torch.tensor([[1] * (len(prompt_ids) + len(response_ids))]),
            "response_mask": torch.tensor([[1] * len(response_ids)]),
            "data_source": np.array(["hotpotqa_distractor"], dtype=object),
            "reward_model": np.array(
                [{"style": "rule", "ground_truth": "Paris"}], dtype=object
            ),
            "extra_info": np.array([{"supporting_titles": ["Paris"]}], dtype=object),
        }
    )


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        QWEN3_8B, local_files_only=True, trust_remote_code=True
    )


def test_dapo_manager_applies_task_reward_without_overlong_penalty():
    tok = _tokenizer()
    manager = _manager(tok, enable=False)
    tensor = manager(_batch("<answer>Paris</answer>", tok))
    assert float(tensor.sum()) == pytest.approx(1.0)


def test_dapo_manager_applies_overlong_penalty():
    tok = _tokenizer()
    # A long but protocol-valid answer: same task score for both managers, so
    # any difference comes from the overlong buffer alone.
    long_response = "<answer>" + ("Paris and nearby towns " * 20) + "</answer>"
    base = _manager(tok, enable=False)(_batch(long_response, tok))
    penalized = _manager(tok, enable=True)(_batch(long_response, tok))
    assert float(base.sum()) > 0.0
    assert float(penalized.sum()) < float(base.sum())
