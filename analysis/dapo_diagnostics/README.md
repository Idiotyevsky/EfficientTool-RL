# DAPO Diagnosis: One-search Collapse

## Observation

On the same Natural Bridge-Hard 200-example evaluation, Fresh DAPO produced
cleaner syntax but much weaker multi-hop behavior than vanilla GRPO:

| Metric | Vanilla GRPO | Fresh DAPO |
|---|---:|---:|
| EM | 51.5% | 33.0% |
| F1 | 62.53% | 41.83% |
| Multi-search | 86.0% | 10.0% |
| Searches | 1.960 | 1.100 |
| Useful | 1.445 | 0.880 |
| Wasted | 0.515 | 0.220 |
| Completion | 97.5% | 100% |
| Invalid action | 0.17% | 0% |

The dominant learned pattern was:

```text
search once → answer
```

Lower wasted search is not considered an improvement because answer quality
regressed sharply.

## What the Result Establishes

- The policy is not failing because of malformed action syntax.
- The DAPO recipe changes behavior, not only aggregate reward.
- Fresh DAPO is substantially more conservative about second-hop retrieval.
- Recipe-level DAPO underperforms vanilla GRPO in this completed comparison.

## What It Does Not Establish

This run does not identify which DAPO component caused the collapse. Dynamic
sampling, asymmetric clipping, token-level aggregation, KL removal, and
length-related behavior changed together.

The completed run also did **not** exercise effective overlong reward shaping.
Native async AgentLoop supplied prefilled `rm_scores`; the stock manager
returned those scores before applying its overlong term.

## Working Hypothesis: Agent-unaware Length

In a multi-turn tool trajectory, the serialized response sequence includes both
assistant-generated tokens and environment-generated tool observations. A
length penalty based on the full post-prompt attention mask can therefore
charge the policy for text it did not generate. This could discourage second
and third searches because each search introduces another observation.

That mechanism is plausible but not yet causally established for the observed
Fresh DAPO collapse, because its overlong term was bypassed.

## Corrective Experiment

The project-local manager now:

1. reads assistant-generated positions from `response_mask`;
2. computes overlong shaping from assistant tokens only;
3. leaves the policy-loss mask unchanged;
4. logs assistant length, total trajectory length, overlong penalty, and search
   count;
5. keeps all other DAPO recipe settings unchanged.

A first end-to-end batch verified that tool observations increased total
trajectory length but were excluded from assistant length. The corrected full
run and held-out evaluation are still required before any causal conclusion.

Relevant implementation:

- [DAPO entry point](../../scripts/train_dapo.py)
- [assistant-only reward manager](../../src/efficienttool_rl/verl/reward_managers/dapo_assistant_length.py)
- [DAPO config](../../configs/dapo/qwen8b_hotpot_mt_strict.yaml)
