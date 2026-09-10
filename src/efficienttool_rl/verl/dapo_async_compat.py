"""Minimal async-rollout compatibility for verl's RayDAPOTrainer.

verl's ``recipe/dapo`` fit loop has two incompatibilities with
``rollout.mode=async`` multi-turn agent training:

1. It calls ``self.actor_rollout_wg.generate_sequences`` (the synchronous
   FSDP worker path). That path enters ``run_until_complete`` on a worker
   whose event loop is already owned by the vLLM async server and crashes
   with ``RuntimeError: this event loop is already running``.
2. It pops only ``raw_prompt_ids`` from the dataset batch and drops
   ``raw_prompt`` (the chat messages the agent loop needs to build contexts),
   causing ``KeyError: 'raw_prompt'`` inside ``ToolAgentLoop.run``.

Upstream ``RayPPOTrainer`` solves both: its ``_get_gen_batch`` keeps
reward-model keys (``data_source`` / ``reward_model`` / ``extra_info`` /
``uid``, which preserves ``raw_prompt`` from the dataset), and async mode
routes through ``AgentLoopManager.generate_sequences``. This module repairs
the DAPO fit loop at runtime, inside the Ray task only; the shared verl
checkout is not modified.

Repairs installed when the patch is applied:

- ``DataProto.pop`` (process-wide) keeps ``raw_prompt`` in the popped gen
  batch when the dataset provides it, matching the main trainer's
  ``_get_gen_batch`` contract;
- per trainer instance, fit-loop generation is routed through
  ``AgentLoopManager.generate_sequences`` (validation batches untouched;
  ``_validate`` already uses the manager).
"""

from __future__ import annotations


def _patch_dataproto_pop_keep_raw_prompt() -> None:
    """Ensure recipe/dapo's inline pop keeps raw_prompt for the agent loop."""
    from verl import DataProto

    if getattr(DataProto, "_efficienttool_raw_prompt_pop_patched", False):
        return

    original_pop = DataProto.pop

    # Keys RayPPOTrainer._get_gen_batch keeps for the async agent loop:
    # reward-model keys for scoring, raw_prompt for context building, and the
    # dataset-promoted tool/interaction kwargs for per-trajectory tool setup.
    _AGENT_KEYS = (
        "data_source",
        "reward_model",
        "extra_info",
        "uid",
        "raw_prompt",
        "tools_kwargs",
        "interaction_kwargs",
        "index",
        "agent_name",
    )

    def pop_keep_agent_keys(self, *, batch_keys, non_tensor_batch_keys):
        keys = list(non_tensor_batch_keys)
        for key in _AGENT_KEYS:
            if key in self.non_tensor_batch and key not in keys:
                keys = keys + [key]
        return original_pop(self, batch_keys=batch_keys, non_tensor_batch_keys=keys)

    DataProto.pop = pop_keep_agent_keys
    DataProto._efficienttool_raw_prompt_pop_patched = True


def _install_instance_route(trainer) -> None:
    """Bind manager-routed generation onto one trainer instance."""
    if not getattr(trainer, "async_rollout_mode", False):
        return
    if getattr(trainer, "_efficienttool_route_installed", False):
        return
    manager = trainer.async_rollout_manager
    worker_group = trainer.actor_rollout_wg
    original_generate = worker_group.generate_sequences

    def routed_generate(gen_batch):
        # Validation batches are already routed through the manager by
        # RayPPOTrainer._validate; only intercept the fit-loop call.
        if gen_batch.meta_info.get("validate", False):
            return original_generate(gen_batch)
        return manager.generate_sequences(gen_batch)

    worker_group.generate_sequences = routed_generate
    trainer._efficienttool_route_installed = True


def _patch_dapo_true_reward_metric() -> None:
    """Default true_reward_score before compute_data_metrics (DAPO fit gap).

    The current metric_utils requires ``batch.batch['true_reward_score']``;
    the main trainer's fit sets it to the reward tensor when the reward
    function does not emit it. The recipe's fit loop lacks that block, so
    add the same default around its compute_data_metrics reference.
    """
    import sys
    from pathlib import Path

    import verl

    recipe_root = str(Path(verl.__file__).resolve().parent.parent)
    if recipe_root not in sys.path:
        sys.path.insert(0, recipe_root)
    import recipe.dapo.dapo_ray_trainer as dapo_module

    if getattr(dapo_module, "_efficienttool_true_reward_patched", False):
        return

    original_compute = dapo_module.compute_data_metrics

    def compute_data_metrics_with_true_reward(batch, use_critic=True):
        import torch

        if "true_reward_score" not in batch.batch:
            batch.batch["true_reward_score"] = batch.batch["token_level_scores"]
        return original_compute(batch, use_critic=use_critic)

    dapo_module.compute_data_metrics = compute_data_metrics_with_true_reward
    dapo_module._efficienttool_true_reward_patched = True


def patch_dapo_trainer_async_rollout() -> None:
    """Make RayDAPOTrainer.fit async-agent compatible (routing + raw_prompt)."""
    import sys
    from pathlib import Path

    import verl

    recipe_root = str(Path(verl.__file__).resolve().parent.parent)
    if recipe_root not in sys.path:
        sys.path.insert(0, recipe_root)
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    if getattr(RayDAPOTrainer, "_efficienttool_async_rollout_patched", False):
        return

    _patch_dapo_true_reward_metric()

    original_fit = RayDAPOTrainer.fit

    def fit_wrapper(self):
        if getattr(self, "async_rollout_mode", False) and self.async_rollout_manager is not None:
            _patch_dataproto_pop_keep_raw_prompt()
            _install_instance_route(self)
        return original_fit(self)

    RayDAPOTrainer.fit = fit_wrapper
    RayDAPOTrainer._efficienttool_async_rollout_patched = True
