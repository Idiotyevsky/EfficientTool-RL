"""Show grouped rewards and relative advantages without training a model."""

from __future__ import annotations

from statistics import fmean, pstdev

from efficienttool_rl.rewards import task_reward


def main() -> None:
    reference = "Paris"
    responses = (
        "<answer>Paris</answer>",
        "<answer>Paris</answer>",
        "<answer>Paris, France</answer>",
        "<answer>London</answer>",
    )
    rewards = [
        task_reward(response, reference, alpha=0.5)["score"]
        for response in responses
    ]
    mean_reward = fmean(rewards)
    reward_std = pstdev(rewards)
    scale = reward_std or 1.0
    advantages = [(reward - mean_reward) / scale for reward in rewards]

    print("Educational GRPO group demonstration")
    print("Prompt: What is the capital of France?")
    print("\nRollouts:")
    for index, (response, reward, advantage) in enumerate(
        zip(responses, rewards, advantages, strict=True),
        start=1,
    ):
        print(
            f"  rollout {index}: reward={reward:.3f}, "
            f"relative_advantage={advantage:+.3f}, response={response}"
        )

    print(f"\nGroup mean reward: {mean_reward:.3f}")
    print(f"Group reward std:  {reward_std:.3f}")
    print("\nThis reuses the project's task reward implementation.")
    print("It explains the group-relative signal; it does not run GRPO or update weights.")


if __name__ == "__main__":
    main()
