"""Project-local verl manager for step-aware, assistant-masked Reward v2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


class RewardV2RewardManager(AbstractRewardManager):
    """Recompute Reward v2 after rollout with the true optimizer step.

    Async AgentLoop pre-scoring does not receive ``global_steps``.  This
    manager deliberately ignores prefilled ``rm_scores``, decodes the finished
    trajectory, and calls Reward v2 with ``data.meta_info['global_steps']``.
    The scalar outcome reward is placed on the final assistant-generated token;
    observation tokens (``response_mask == 0``) never receive policy reward.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "response_mask" not in data.batch:
            raise KeyError(
                "Reward v2 requires response_mask so tool observations cannot "
                "receive policy reward"
            )

        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info: defaultdict[str, list[Any]] = defaultdict(list)
        global_step = int(data.meta_info.get("global_steps", 0))
        already_printed: set[Any] = set()

        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            trajectory_mask = item.batch["attention_mask"][prompt_length:].bool()
            total_trajectory_length = int(trajectory_mask.sum().item())
            valid_response_ids = item.batch["responses"][:total_trajectory_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            reward_model = item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth")
            if ground_truth is None:
                raise KeyError("Reward v2 requires reward_model.ground_truth")
            data_source = item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = dict(item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
            extra_info["rollout_reward_scores"] = item.non_tensor_batch.get(
                "reward_scores", {}
            )

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                global_step=global_step,
            )
            if not isinstance(result, dict) or "score" not in result:
                raise TypeError("Reward v2 compute_score must return a score dictionary")
            reward = float(result["score"])
            for key, value in result.items():
                reward_extra_info[key].append(value)

            assistant_mask = item.batch["response_mask"].bool() & trajectory_mask
            assistant_positions = torch.nonzero(
                assistant_mask, as_tuple=False
            ).flatten()
            if assistant_positions.numel() == 0:
                if reward != 0:
                    raise ValueError(
                        "non-zero Reward v2 score has no assistant token destination"
                    )
            else:
                reward_position = int(assistant_positions[-1].item())
                reward_tensor[i, reward_position] = reward

            if data_source not in already_printed and len(already_printed) < self.num_examine:
                already_printed.add(data_source)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in result.items():
                    print(f"[{key}]", value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor


__all__ = ["RewardV2RewardManager"]
