#!/usr/bin/env python3
"""Run multi-turn GRPO training with verl and project-local adapters.

Example:
    python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict

Hydra overrides are passed through verbatim, e.g.::

    python scripts/train_grpo.py --config-name qwen8b_hotpot_mt_strict \
        trainer.total_training_steps=2

The runner installs the project-local JSON-dump and chat-template compat
patches inside the Ray task; the shared verl checkout itself is not modified.
"""

from __future__ import annotations

import hydra
import ray

from efficienttool_rl.verl.json_compat import (
    patch_ray_trainer_json_dump,
    patch_tool_agent_chat_template_defaults,
)
from verl.trainer.main_ppo import TaskRunner, run_ppo


class EfficientToolTaskRunner(TaskRunner):
    """Install project-local compatibility hooks inside the Ray task."""

    def run(self, config):
        patch_ray_trainer_json_dump()
        patch_tool_agent_chat_template_defaults()
        return super().run(config)


@hydra.main(config_path="../configs/grpo", version_base=None)
def main(config):
    remote_task_runner = ray.remote(num_cpus=1)(EfficientToolTaskRunner)
    run_ppo(config, task_runner_class=remote_task_runner)


if __name__ == "__main__":
    main()
