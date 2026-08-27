---
title: 环境与复现
description: MiniAgentRL 学习与研究环境边界。
---

# 环境与复现

课程不要求所有人先复制完整的 8B 集群环境。根据目标选择最小 tier。

| Tier | 已验证用途 | 资源边界 |
| --- | --- | --- |
| CPU concepts | parser、BM25、多轮 loop、Reward、GRPO intuition、efficiency analyzer | Python 3.10+ |
| Local model | Qwen3-1.7B Tool Calling 与 bounded ReAct | 本地 checkpoint + 可用 CUDA 显存 |
| Real smoke | 8 prompts × 4 rollouts × 1 update | 1 GPU + compatible verl/Ray/vLLM |
| Full-scale research | Strict Qwen3-8B multi-turn GRPO | 当前实验使用 4× RTX A6000-class GPU |

## 长任务前的固定检查

```bash
nvidia-smi
df -h
```

在共享服务器上运行前，请确认 GPU 上已有进程的归属，不要终止他人的任务；同时为每次运行使用唯一输出目录。

## Reproducibility record

每次完整实验保存或记录：

```text
git commit · timestamp · seed · model
dataset split + fingerprint · resolved config
reward config · GPU topology · framework versions
rollouts · validation · logs · checkpoints
```

大模型、数据集、checkpoint 与 run artifact 建议存放在 Git checkout 外，避免误提交大文件和本地路径。

详细已验证版本见 [raw environment report](https://github.com/Idiotyevsky/EfficientTool-RL/blob/main/docs/environment_report.md)。运行 GRPO 前同时阅读 [Chapter 07](/learn/07-grpo-smoke) 的 wrapper 与安全边界。
