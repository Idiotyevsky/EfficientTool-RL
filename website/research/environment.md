---
title: 环境与复现
description: MiniAgentRL 学习与研究环境边界。
---

# 环境与复现

课程不要求所有人先复制 formal 8B 集群环境。根据目标选择最小 tier。

| Tier | 已验证用途 | 资源边界 |
| --- | --- | --- |
| CPU concepts | parser、BM25、多轮 loop、Reward、GRPO intuition、efficiency analyzer | Python 3.10+ |
| Local model | Qwen3-1.7B Tool Calling 与 bounded ReAct | 本地 checkpoint + 可用 CUDA 显存 |
| Real smoke | 8 prompts × 4 rollouts × 1 update | 1 GPU + compatible verl/Ray/vLLM |
| Formal research | Strict Qwen3-8B multi-turn GRPO | 当前实验使用 4× RTX A6000-class GPU |

## 长任务前的固定检查

```bash
nvidia-smi
df -h
```

还必须确认 PID、user、command、checkpoint 体积与唯一输出目录。绝不能只根据 GPU utilization 终止进程。

## Reproducibility record

每个 formal run 保存或记录：

```text
git commit · timestamp · seed · model
dataset split + fingerprint · resolved config
reward config · GPU topology · framework versions
rollouts · validation · logs · checkpoints
```

大模型、数据集、checkpoint 与 run artifact 位于 Git checkout 外部；公开页面不暴露机器路径或内部基础设施细节。

详细已验证版本见 [raw environment report](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/environment_report.md)。运行 GRPO 前同时阅读 [Chapter 07](/learn/07-grpo-smoke) 的 bounded wrapper 与安全边界。
