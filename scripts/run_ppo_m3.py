"""Run the M3 PPO entry point with the local JSON-dump compatibility shim."""

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


@hydra.main(config_path="../configs", config_name="m3_sanity", version_base=None)
def main(config):
    remote_task_runner = ray.remote(num_cpus=1)(EfficientToolTaskRunner)
    run_ppo(config, task_runner_class=remote_task_runner)


if __name__ == "__main__":
    main()
