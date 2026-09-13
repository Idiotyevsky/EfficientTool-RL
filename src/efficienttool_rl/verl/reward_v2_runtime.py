"""Scoped runtime wiring for step-aware Reward v2 on stock verl GRPO.

The patch is activated only by the Reward v2 config.  It does not modify
upstream verl files, Agent behavior, the policy loss, or existing reward
configs.  It carries the optimizer step through async rollout metadata,
selects the project-local Reward v2 manager, and exposes scalar component
means in the normal training logger.
"""

from __future__ import annotations

from typing import Any

_REWARD_V2_METRICS = (
    "answer_reward",
    "marginal_evidence_reward",
    "current_beta",
    "wasted_search_count",
    "wasted_search_penalty",
    "total_reward",
    "search_count",
    "useful_search_count",
    "unique_support_count",
    "multi_search",
    "evidence_gain_per_search",
    "repeated_search_count",
)


def is_reward_v2_config(config: Any) -> bool:
    custom = config.get("custom_reward_function") or {}
    path = str(custom.get("path", ""))
    return path.endswith("/reward_v2.py") or path == "reward_v2.py"


def _patch_agent_loop_step_passthrough() -> None:
    from verl.experimental.agent_loop import AgentLoopManager

    if getattr(AgentLoopManager, "_toolagentlab_reward_v2_step_patched", False):
        return
    original_generate = AgentLoopManager.generate_sequences

    def generate_with_step(self, prompts):
        global_step = prompts.meta_info.get("global_steps")
        validate = prompts.meta_info.get("validate")
        output = original_generate(self, prompts)
        if global_step is not None:
            output.meta_info["global_steps"] = int(global_step)
        if validate is not None:
            output.meta_info["validate"] = bool(validate)
        return output

    AgentLoopManager.generate_sequences = generate_with_step
    AgentLoopManager._toolagentlab_reward_v2_step_patched = True


def _patch_main_ppo_reward_loader() -> None:
    import verl.trainer.main_ppo as main_ppo
    from verl.trainer.ppo.reward import get_custom_reward_fn

    from efficienttool_rl.verl.reward_managers.reward_v2 import (
        RewardV2RewardManager,
    )

    if getattr(main_ppo, "_toolagentlab_reward_v2_loader_patched", False):
        return
    original_loader = main_ppo.load_reward_manager

    def load_reward_v2_manager(config, tokenizer, num_examine, **reward_kwargs):
        if not is_reward_v2_config(config):
            return original_loader(
                config, tokenizer, num_examine, **reward_kwargs
            )
        compute_score = get_custom_reward_fn(config)
        if compute_score is None:
            raise ValueError("Reward v2 requires a custom reward function")
        return RewardV2RewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

    main_ppo.load_reward_manager = load_reward_v2_manager
    main_ppo._toolagentlab_reward_v2_loader_patched = True


def _patch_reward_v2_metric_logging() -> None:
    import numpy as np
    import verl.trainer.ppo.ray_trainer as ray_trainer

    if getattr(ray_trainer, "_toolagentlab_reward_v2_metrics_patched", False):
        return
    original_compute = ray_trainer.compute_data_metrics

    def compute_data_metrics_with_reward_v2(batch, use_critic=True):
        metrics = original_compute(batch, use_critic=use_critic)
        for key in _REWARD_V2_METRICS:
            values = batch.non_tensor_batch.get(key)
            if values is None:
                continue
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if numeric.size:
                metrics[f"reward_v2/{key}/mean"] = float(numeric.mean())
        return metrics

    ray_trainer.compute_data_metrics = compute_data_metrics_with_reward_v2
    ray_trainer._toolagentlab_reward_v2_metrics_patched = True


def patch_reward_v2_runtime() -> None:
    """Install Reward v2 wiring inside the Ray TaskRunner process only."""
    _patch_agent_loop_step_passthrough()
    _patch_main_ppo_reward_loader()
    _patch_reward_v2_metric_logging()


__all__ = ["is_reward_v2_config", "patch_reward_v2_runtime"]
