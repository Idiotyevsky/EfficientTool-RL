"""Control experiment: stock verl TaskRunner, no project patches, no custom trainer."""
import hydra
import ray
from verl.trainer.main_ppo import TaskRunner, run_ppo


@hydra.main(config_path="../configs/grpo", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TaskRunner))


if __name__ == "__main__":
    main()
