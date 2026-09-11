#!/usr/bin/env python3
"""Run DAPO-style training on the multi-turn tool-agent pipeline.

The trainer is verl's own ``RayDAPOTrainer`` (recipe/dapo), which adds:
- dynamic sampling: groups whose rollout rewards have zero variance are
  filtered and generation continues until the batch is filled
  (``algorithm.filter_groups``);
- token-level loss aggregation via ``actor.loss_agg_mode``;
- asymmetric clipping via ``actor.clip_ratio_low`` / ``clip_ratio_high``;
- soft overlong punishment via the DAPO reward manager and
  ``reward_model.overlong_buffer``.

The entry point reuses verl's standard ``TaskRunner`` workflow (async
rollout workers, dataset injection, config validation) and only changes the
trainer class and reward-manager construction. Validation always uses the
pure task reward (no overlong penalty) so results stay comparable across
GRPO and DAPO runs.
"""

from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunner, run_ppo

from efficienttool_rl.verl.dapo_async_compat import patch_dapo_trainer_async_rollout
from efficienttool_rl.verl.json_compat import (
    patch_ray_trainer_json_dump,
    patch_tool_agent_chat_template_defaults,
)


def _load_dapo_trainer_cls():
    """Import verl's recipe DAPO trainer from the verl source checkout."""
    import sys
    from pathlib import Path

    import verl

    recipe_root = str(Path(verl.__file__).resolve().parent.parent)
    if recipe_root not in sys.path:
        sys.path.insert(0, recipe_root)
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    return RayDAPOTrainer


def _load_reward_manager(config, tokenizer, num_examine: int, *, with_overlong: bool):
    # Import registers the project-local multi-turn-aware DAPO manager before
    # verl resolves ``reward_model.reward_manager`` from its registry.
    from verl.trainer.ppo.reward import load_reward_manager

    import efficienttool_rl.verl.reward_managers  # noqa: F401

    manager_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    manager_config.reward_model.reward_manager = "efficienttool_dapo"

    reward_kwargs = {}
    if with_overlong and config.reward_model.overlong_buffer.enable:
        reward_kwargs["max_resp_len"] = config.data.max_response_length
        reward_kwargs["overlong_buffer_cfg"] = config.reward_model.overlong_buffer
    return load_reward_manager(
        manager_config, tokenizer, num_examine, **reward_kwargs
    )


class EfficientToolDAPOTaskRunner(TaskRunner):
    """Standard verl task runner wired to RayDAPOTrainer with project patches."""

    def run(self, config):
        patch_ray_trainer_json_dump()
        patch_tool_agent_chat_template_defaults()
        patch_dapo_trainer_async_rollout()

        from pprint import pprint

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        ray_dapo_trainer_cls = _load_dapo_trainer_cls()

        # Training reward carries the DAPO soft overlong punishment; validation
        # uses the pure task reward for cross-algorithm comparability.
        reward_fn = _load_reward_manager(config, tokenizer, 0, with_overlong=True)
        val_reward_fn = _load_reward_manager(config, tokenizer, 1, with_overlong=False)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ray_dapo_trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../configs/dapo", version_base=None)
def main(config):
    remote_task_runner = ray.remote(num_cpus=1)(EfficientToolDAPOTaskRunner)
    run_ppo(config, task_runner_class=remote_task_runner)


if __name__ == "__main__":
    main()
