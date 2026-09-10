"""Tests for the RayDAPOTrainer async-rollout routing patch."""

import pytest

pytest.importorskip("verl")

from efficienttool_rl.verl.dapo_async_compat import (  # noqa: E402
    _install_instance_route,
    patch_dapo_trainer_async_rollout,
)


class _Manager:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, gen_batch):
        self.calls.append(("manager", gen_batch))
        return "from-manager"


class _WorkerGroup:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, gen_batch):
        return "from-worker"


class _Trainer:
    """Mimics the attributes the patch touches."""

    def __init__(self, *, async_mode: bool, validate: bool):
        self.async_rollout_mode = async_mode
        self.async_rollout_manager = _Manager() if async_mode else None
        self.actor_rollout_wg = _WorkerGroup()
        self._validate_flag = validate


def _gen_batch(validate: bool):
    from types import SimpleNamespace

    return SimpleNamespace(meta_info={"validate": validate})


def test_patch_is_idempotent():
    patch_dapo_trainer_async_rollout()
    patch_dapo_trainer_async_rollout()
    import sys
    from pathlib import Path

    import verl

    sys.path.insert(0, str(Path(verl.__file__).resolve().parent.parent))
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

    assert RayDAPOTrainer._efficienttool_async_rollout_patched is True


def test_route_sends_training_batches_to_manager():
    trainer = _Trainer(async_mode=True, validate=False)
    _install_instance_route(trainer)
    gen = _gen_batch(validate=False)
    assert trainer.actor_rollout_wg.generate_sequences(gen) == "from-manager"
    assert trainer.async_rollout_manager.calls == [("manager", gen)]


def test_route_leaves_validation_batches_untouched():
    trainer = _Trainer(async_mode=True, validate=False)
    _install_instance_route(trainer)
    gen = _gen_batch(validate=True)
    assert trainer.actor_rollout_wg.generate_sequences(gen) == "from-worker"
    assert trainer.async_rollout_manager.calls == []


def test_route_noop_without_async_mode():
    trainer = _Trainer(async_mode=False, validate=False)
    _install_instance_route(trainer)
    gen = _gen_batch(validate=False)
    assert trainer.actor_rollout_wg.generate_sequences(gen) == "from-worker"
