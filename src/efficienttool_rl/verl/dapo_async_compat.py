"""Minimal async-rollout compatibility for verl's RayDAPOTrainer.

verl's ``recipe/dapo`` fit loop calls ``self.actor_rollout_wg.generate_sequences``
(the synchronous FSDP worker path). With ``rollout.mode=async`` and a
multi-turn agent loop, that path enters ``run_until_complete`` on a worker
whose event loop is already owned by the vLLM async server and crashes with
``RuntimeError: this event loop is already running``.

Validation never hits this because ``RayPPOTrainer._validate`` routes through
``AgentLoopManager.generate_sequences`` (the async agent-loop path). This
module routes the trainer's training-generation call through the same
manager, keeping every DAPO component (dynamic sampling, asymmetric
clipping, token-level loss, overlong handling) untouched.

The patch is installed inside the Ray task by ``scripts/train_dapo.py``;
the shared verl checkout is not modified.
"""

from __future__ import annotations

import inspect


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


def patch_dapo_trainer_async_rollout() -> None:
    """Make RayDAPOTrainer.fit use the async agent-loop manager for rollouts."""
    import sys
    from pathlib import Path

    import verl

    recipe_root = str(Path(verl.__file__).resolve().parent.parent)
    if recipe_root not in sys.path:
        sys.path.insert(0, recipe_root)
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    if getattr(RayDAPOTrainer, "_efficienttool_async_rollout_patched", False):
        return

    original_fit = RayDAPOTrainer.fit

    if inspect.iscoroutinefunction(original_fit):

        async def fit_wrapper(self):
            _install_instance_route(self)
            return await original_fit(self)

    else:

        def fit_wrapper(self):
            _install_instance_route(self)
            return original_fit(self)

    RayDAPOTrainer.fit = fit_wrapper
    RayDAPOTrainer._efficienttool_async_rollout_patched = True
