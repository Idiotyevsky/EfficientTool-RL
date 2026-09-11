# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAPO reward manager with multi-turn-aware overlong accounting.

verl's stock DAPO manager measures the entire post-prompt trajectory with
``attention_mask``. In a tool-agent rollout that includes both assistant
tokens and injected tool observations, so useful searches consume the
overlong budget. This project-local manager keeps the stock reward semantics
but measures the overlong term with ``response_mask`` (assistant tokens only).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.naive import NaiveRewardManager


def _search_count(data_item: Any) -> int:
    """Count dispatched tool calls; strict Hotpot-MT exposes search only."""
    explicit = data_item.non_tensor_batch.get("executed_search_calls")
    if explicit is not None:
        return int(explicit)
    tool_rewards = data_item.non_tensor_batch.get("tool_rewards", [])
    try:
        return len(tool_rewards)
    except TypeError:
        return 0


@register("efficienttool_dapo")
class AssistantLengthDAPORewardManager(AbstractRewardManager):
    """Stock DAPO scoring with assistant-only overlong punishment."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Async AgentLoop first computes task reward before its postprocessor
        # creates response_mask. Defer shaping until rm_scores returns with it.
        if "response_mask" not in data.batch:
            if "rm_scores" in data.batch:
                raise KeyError(
                    "efficienttool_dapo cannot apply overlong shaping to "
                    "prefilled rm_scores without response_mask"
                )
            task_only_manager = NaiveRewardManager(
                tokenizer=self.tokenizer,
                num_examine=self.num_examine,
                compute_score=self.compute_score,
                reward_fn_key=self.reward_fn_key,
            )
            return task_only_manager(data, return_dict=return_dict)

        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info = defaultdict(list)
        already_print_data_sources: dict[Any, int] = {}
        has_prefilled_scores = "rm_scores" in data.batch

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(
                data_item.batch["attention_mask"][:prompt_length].sum().item()
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            trajectory_mask = data_item.batch["attention_mask"][prompt_length:].bool()
            assistant_mask = data_item.batch["response_mask"].bool() & trajectory_mask
            total_trajectory_length = int(trajectory_mask.sum().item())
            assistant_token_length = int(assistant_mask.sum().item())
            valid_response_ids = response_ids[:total_trajectory_length]

            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # Async AgentLoop postprocessing may retain only rm_scores and
            # trajectory metadata. Dataset fields are required only when the
            # task score still needs to be computed locally.
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth")
            result = None
            if has_prefilled_scores:
                # Async AgentLoop computes task reward before batching and
                # stores it in rm_scores. Stock DAPO returns here, which skips
                # overlong shaping entirely. Recover the scalar task score,
                # then apply assistant-only shaping below.
                score = float(data_item.batch["rm_scores"].sum().item())
                for key in data.meta_info.get("reward_extra_keys", []):
                    if key in data_item.non_tensor_batch:
                        reward_extra_info[key].append(
                            data_item.non_tensor_batch[key]
                        )
            else:
                if data_source == "unknown" or ground_truth is None:
                    raise KeyError(
                        "task reward computation requires data_source and "
                        "reward_model.ground_truth when rm_scores is absent"
                    )
                extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
                extra_info["rollout_reward_scores"] = (
                    data_item.non_tensor_batch.get("reward_scores", {})
                )
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                if isinstance(result, dict):
                    score = float(result["score"])
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = float(result)
                    reward_extra_info["acc"].append(score)

            overlong_penalty = 0.0
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - buffer_len
                exceed_len = assistant_token_length - expected_len
                penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_penalty = min(
                    -exceed_len / buffer_len * penalty_factor,
                    0.0,
                )

            reward = score + overlong_penalty
            assistant_positions = torch.nonzero(assistant_mask, as_tuple=False).flatten()
            if assistant_positions.numel() > 0:
                reward_position = int(assistant_positions[-1].item())
            else:
                reward_position = max(total_trajectory_length - 1, 0)
            reward_tensor[i, reward_position] = reward

            reward_extra_info["assistant_token_length"].append(
                assistant_token_length
            )
            reward_extra_info["total_trajectory_length"].append(
                total_trajectory_length
            )
            reward_extra_info["overlong_penalty"].append(overlong_penalty)
            reward_extra_info["search_count"].append(_search_count(data_item))
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"].append(overlong_penalty)
                reward_extra_info["overlong"].append(overlong_penalty < 0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[assistant_token_length]", assistant_token_length)
                print("[total_trajectory_length]", total_trajectory_length)
                print("[overlong_penalty]", overlong_penalty)
                print("[search_count]", _search_count(data_item))
                if result is None:
                    print("[score]", score)
                elif isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor


__all__ = ["AssistantLengthDAPORewardManager"]
