# AGENTS.md — EfficientTool-RL
## Sol-Orchestrated / Luna-Max-Executed Research Engineering Protocol

> Project: **EfficientTool-RL: Cost-Aware Multi-turn Tool Agent Training with GRPO**
>
> Primary goal: build a complete, reproducible, resume-quality Agentic RL project that trains an LLM agent to use tools effectively **and efficiently**.
>
> Governing principle:
>
> **Sol decides what should be done and whether the evidence is convincing. Luna Max executes bounded, verifiable engineering tasks.**

---

# 0. Operating Model

This repository uses a two-level research-agent architecture.

## 0.1 Sol — Orchestrator / Research Lead

Sol owns:

- project scope;
- architecture decisions;
- milestone planning;
- experiment design;
- hypothesis evaluation;
- diagnosis of ambiguous failures;
- resource allocation;
- acceptance/rejection of worker outputs;
- final scientific interpretation;
- decisions about whether to continue, pivot, or stop.

Sol must **not** delegate away final responsibility for research correctness.

Sol should avoid spending large amounts of time on routine implementation if the work can be delegated safely.

---

## 0.2 Luna Max — Worker / Research Engineer

Luna Max should be used for tasks that are:

- clearly scoped;
- locally verifiable;
- implementation-heavy;
- testable;
- reversible;
- low ambiguity;
- unlikely to alter the overall research direction.

Good Luna tasks include:

- implementing a parser;
- writing dataset loaders;
- implementing BM25 retrieval;
- adding metrics;
- writing unit tests;
- building CLI/config plumbing;
- analyzing trajectory files;
- profiling GPU usage;
- generating plots from existing results;
- cleaning README sections;
- fixing a localized runtime bug with a clear reproduction;
- implementing a reward formula already approved by Sol.

Luna must **not independently redefine the research problem**.

---

# 1. Core Research Objective

The central question is:

> **Can reinforcement learning teach an LLM agent not only to use tools successfully, but also to use them efficiently?**

The intended hypothesis chain is:

1. A training-free ReAct agent provides a measurable baseline.
2. Vanilla multi-turn GRPO improves task success.
3. Vanilla outcome-based RL may encourage unnecessary search/tool usage or unnecessarily long trajectories.
4. A cost-aware objective can move the policy toward a better performance-efficiency trade-off.
5. The result should be analyzed in terms of both task quality and agent behavior.

The project should ultimately study an objective of the form:

\[
R =
R_{\mathrm{task}}
-
\lambda_{\mathrm{tool}} N_{\mathrm{tool}}
-
\lambda_{\mathrm{token}} N_{\mathrm{token}}
\]

However, **complex reward shaping is forbidden before vanilla GRPO is validated**.

---

# 2. Scope

## 2.1 Required MVP

The MVP must contain:

- Qwen-family base model;
- multi-turn Agent loop;
- one tool: `search`;
- deterministic local search environment;
- HotpotQA-style multi-hop QA;
- training-free ReAct baseline;
- GRPO training;
- held-out evaluation;
- trajectory logging;
- tool-use statistics;
- cost-aware search penalty;
- lambda ablation;
- failure analysis;
- reproducible commands;
- final README.

---

## 2.2 Optional extensions

Only after the MVP is complete:

- Calculator;
- mixed-task tool routing;
- token-cost penalty;
- redundant-query penalty;
- adaptive tool cost;
- budget-conditioned agents;
- group-variance filtering;
- Python tool;
- larger model scaling.

---

## 2.3 Explicitly out of scope for MVP

Do not spend time on:

- browser automation;
- GUI agents;
- multi-agent debate;
- long-term memory;
- MCP productization;
- vector database product features;
- web frontend polish;
- multimodal agents;
- 7B+ models before smaller models work;
- multi-node training;
- inventing a new GRPO algorithm;
- paper-writing before the experiments are stable.

---

# 3. Hardware Assumptions

Design for approximately:

- Linux server;
- 4 × RTX 4090 24 GB;
- CUDA-capable environment;
- shared research infrastructure.

Preferred models:

- `Qwen3-1.7B` for debugging and smoke tests;
- `Qwen3-4B` for main experiments if feasible.

Do not assume A100/H100-class memory.

Before every long run, inspect:

- GPU ownership and utilization;
- free GPU memory;
- disk space;
- current processes;
- expected checkpoint size;
- expected output/log location.

Never terminate a process unless ownership and purpose are confirmed.

---

# 4. Framework Preference

Prefer mature infrastructure.

Suggested priority:

1. **Agent-R1 / verl-style Agentic RL stack** for the main system;
2. **vLLM or SGLang** for rollout inference;
3. Hugging Face Transformers for model interfaces;
4. TRL GRPO only for small sanity experiments when useful.

Do not reimplement distributed RL from scratch unless there is a concrete technical reason.

When integrating upstream code:

- inspect the relevant implementation first;
- document version/commit where practical;
- preserve upstream license requirements;
- keep local modifications minimal and well-isolated.

---

# 5. Orchestration Rules

## 5.1 Sol must decompose before delegating

Before assigning work, Sol should define:

- exact task;
- files allowed to change;
- inputs;
- expected outputs;
- validation commands;
- acceptance criteria;
- known constraints.

A worker task should ideally fit the format:

```text
Task:
Implement X.

Allowed files:
- ...

Do not modify:
- ...

Acceptance criteria:
1. ...
2. ...

Validation:
- command A
- command B

Return:
- summary
- changed files
- test results
- unresolved issues
```

Do not delegate vague instructions such as:

> “Make the RL pipeline work.”

---

## 5.2 One owner per mutable subsystem

Avoid multiple workers editing the same files concurrently.

Recommended ownership boundaries:

- Worker A: dataset / preprocessing;
- Worker B: tools / environment;
- Worker C: metrics / evaluation;
- Worker D: analysis / plotting.

For overlapping files, Sol must serialize work or explicitly assign ownership.

---

## 5.3 Parallelize only independent tasks

Safe parallel examples:

- HotpotQA loader + parser tests;
- BM25 retriever + evaluation metrics;
- trajectory analyzer + README scaffolding.

Unsafe parallel examples:

- two workers both rewriting `tool_env.py`;
- one worker changing reward semantics while another launches training;
- one worker changing chat templates while another interprets current evaluation results.

---

## 5.4 Workers may not silently broaden scope

If a worker discovers that completing the assigned task requires:

- major architecture changes;
- replacing the RL framework;
- changing model family;
- changing reward semantics;
- changing benchmark;
- destructive environment changes;
- large dependency upgrades;

the worker must stop and return an escalation report.

---

# 6. Worker Escalation Protocol

A Luna worker must stop and escalate when:

1. the failure has more than one plausible root cause;
2. an experiment contradicts the expected hypothesis;
3. a framework-level redesign appears necessary;
4. a dependency change could break unrelated components;
5. GPU OOM persists after one or two obvious local fixes;
6. reward behavior is suspicious;
7. evaluation metrics conflict with manual inspection;
8. data leakage is suspected;
9. a long experiment would be required merely to discover whether a guess is correct;
10. the next step would materially change project scope.

Escalation report format:

```markdown
## Escalation

### Observed
- ...

### Reproduction
- ...

### Evidence
- ...

### Candidate causes
1. ...
2. ...
3. ...

### Candidate actions
A. ...
B. ...
C. ...

### Recommendation
- ...

### What I did NOT change
- ...
```

Sol then decides the next action.

---

# 7. Worker Completion / Handoff Format

Every worker task must end with a handoff report.

```markdown
# Worker Handoff

## Task
...

## Status
PASS / PARTIAL / FAIL

## Files changed
- ...

## What was implemented
- ...

## Validation performed
- command:
- result:

## Evidence
- ...

## Known limitations
- ...

## Risks
- ...

## Suggested next step
- ...
```

A claim of success without test evidence is not sufficient.

---

# 8. Milestone Gate Policy

The project proceeds through strict gates.

Sol may advance to the next milestone only when the current milestone has:

- working implementation;
- recorded validation;
- stored evidence;
- no unresolved blocker that invalidates the next stage.

Never advance merely because “the code runs”.

---

# 9. Milestone M0 — Environment Reconnaissance

## Goal

Understand the machine and repository before changing the system.

## Required work

Inspect:

- repository tree;
- package manager;
- Python version;
- PyTorch version;
- CUDA version;
- GPU count/type;
- GPU usage;
- disk space;
- existing environment;
- installed versions of:
  - verl;
  - vLLM;
  - SGLang;
  - Transformers;
  - FlashAttention if present.

Create:

```text
docs/environment_report.md
PROGRESS.md
```

## Gate M0

PASS only if:

- a supported Qwen model can be loaded;
- one inference completes;
- environment report exists;
- no destructive environment modification was performed blindly.

---

# 10. Milestone M1 — Minimal Multi-turn Agent

## Goal

Create the smallest correct Agent loop.

Use:

- Qwen3-1.7B;
- no RL;
- one tool: `search`.

Suggested action format:

```text
<tool_call>
{"name": "search", "arguments": {"query": "..."}}
</tool_call>
```

Final answer format:

```text
<answer>
...
</answer>
```

The parser must safely handle:

- valid tool call;
- malformed JSON;
- unknown tool;
- missing arguments;
- repeated calls;
- plain text;
- final answer termination.

Required unit tests:

- parser success;
- malformed tool call;
- unknown tool;
- missing arguments;
- final answer;
- episode termination.

## Gate M1

PASS only if:

- multi-turn episodes work;
- malformed actions do not crash the process;
- tool interaction is logged structurally;
- unit tests pass.

---

# 11. Milestone M2 — Local Search Environment + ReAct Baseline

## 11.1 Search environment

Start with deterministic local retrieval.

Preferred initial benchmark:

- HotpotQA distractor setting.

Initial retrieval:

- BM25;
- top-k configurable;
- bounded context;
- no live web API.

Implement:

```python
search(query: str, top_k: int = 3)
```

Return structured results:

- title;
- passage;
- score if available.

Configurable:

- `max_turns`;
- `max_search_calls`;
- `top_k`;
- `max_observation_tokens`.

---

## 11.2 ReAct baseline

Before RL, run a stable inference-only baseline.

Record:

- Exact Match;
- token F1;
- average search calls;
- average turns;
- generated tokens;
- invalid tool-call rate;
- completion rate.

Save:

```text
outputs/react_hotpotqa/
  metrics.json
  trajectories.jsonl
  failures.jsonl
```

Manually inspect at least:

- 20 successes;
- 20 failures.

Failure categories:

- poor query;
- insufficient search;
- excessive search;
- repeated search;
- malformed action;
- premature answer;
- correct evidence / wrong reasoning;
- missing evidence.

Create:

```text
analysis/react_failure_analysis.md
```

## Gate M2

PASS only if:

- metrics are reproducible;
- trajectories can be inspected;
- search actually affects the model context;
- there is no obvious answer leakage;
- failures are categorized.

---

# 12. Milestone M3 — GRPO Sanity Check

Do not launch a main run yet.

Initial suggested scale:

```text
model: Qwen3-1.7B
training prompts: ~500
group size G: 4
max turns: 3–5
max generated tokens: ~1024
temperature: ~0.8
learning rate: ~1e-6
```

These are starting values, not sacred constants.

---

## 12.1 Initial reward

Start with task reward only.

For HotpotQA:

\[
R_{\mathrm{task}} =
\alpha EM + (1-\alpha)F1
\]

Start with:

\[
\alpha = 0.5
\]

Do not add search penalties yet.

---

## 12.2 Required GRPO diagnostics

Log at minimum:

- mean reward;
- reward std;
- EM;
- F1;
- policy loss;
- KL-related stats if used;
- response length;
- tool calls;
- invalid actions;
- episode turns;
- rollout generation time;
- training step time.

Also track per-prompt group reward variance:

\[
Var(R_1,\dots,R_G)
\]

and:

```text
zero_variance_group_ratio
```

---

## 12.3 Mandatory sanity checks

Before any large run:

### A
Same prompt generates meaningfully different trajectories.

### B
A non-trivial fraction of groups contain different rewards.

### C
Correct trajectories receive higher reward.

### D
Policy parameters actually update.

### E
Evaluation changes after training.

### F
Reward does not use hidden test information.

### G
Retriever output does not leak benchmark answers unrealistically.

### H
The exact evaluation set is not used for policy optimization.

## Gate M3

PASS only if all sanity checks are supported by evidence.

If GRPO runs but no meaningful learning signal exists, M3 is **FAIL**, not PASS.

---

# 13. Milestone M4 — Main Vanilla GRPO

Scale gradually:

```text
500
→ 2,000
→ 5,000
→ optional 10,000
```

Do not jump directly to maximum scale.

Main comparison:

| Method | EM ↑ | F1 ↑ | Search Calls | Tokens | Turns |
|---|---:|---:|---:|---:|---:|
| ReAct | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD |

Use actual results only.

## Gate M4

PASS if:

- held-out evaluation is complete;
- results are reproducible enough for a project claim;
- GRPO changes behavior measurably;
- the result is interpretable.

A negative result may still pass if it is technically valid and carefully diagnosed.

---

# 14. Milestone M5 — Cost-Aware GRPO

Only after M4 is accepted.

Primary objective:

\[
R =
R_{\mathrm{task}}
-
\lambda_{\mathrm{search}}N_{\mathrm{search}}
\]

Suggested sweep:

\[
\lambda_{\mathrm{search}}
\in
\{0, 0.01, 0.03, 0.05, 0.1\}
\]

Adjust values only after checking the actual reward scale.

Required table:

| λ_search | EM | F1 | Avg Search Calls | Avg Tokens | Avg Turns |
|---:|---:|---:|---:|---:|---:|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 0.01 | TBD | TBD | TBD | TBD | TBD |
| 0.03 | TBD | TBD | TBD | TBD | TBD |
| 0.05 | TBD | TBD | TBD | TBD | TBD |
| 0.10 | TBD | TBD | TBD | TBD | TBD |

Required visualization:

- performance vs average search calls;
- identify Pareto-efficient operating points.

## Gate M5

PASS only if the trade-off is measured, not merely asserted.

---

# 15. Reward-Hacking Analysis

Explicitly search for:

## Over-search

```text
search
search
search
search
...
answer
```

## Under-search

High penalty causes guessing without retrieval.

## Query degeneration

Repeated or near-identical searches.

## Premature termination

Answering early to avoid cost.

## Verbosity exploitation

Long reasoning when only tool count is penalized.

Required behavioral metrics where feasible:

- duplicate query rate;
- search count distribution;
- early-answer rate;
- average reasoning length;
- answer accuracy conditioned on search count.

Do not hide undesirable behaviors.

---

# 16. Optional Milestone M6 — Multi-tool Routing

Only after Search Agent experiments are complete.

Add deterministic:

```text
calculator
```

Possible datasets:

- HotpotQA → search;
- GSM8K-style tasks → calculator.

Do not reveal task type explicitly to the policy.

Study whether the Agent learns:

- whether to use a tool;
- which tool to use;
- when to stop.

Potential metrics:

- task accuracy;
- tool selection accuracy;
- unnecessary tool-call rate;
- total tool cost.

Python tool is optional and must not block project completion.

---

# 17. GPU / Training Safety

## 17.1 Before long runs

Always inspect:

```bash
nvidia-smi
df -h
```

and relevant process ownership.

Record:

- GPU allocation;
- model;
- run name;
- output directory;
- config;
- expected checkpoint behavior.

---

## 17.2 Shared server safety

Never kill processes based solely on GPU utilization.

Before terminating anything:

- inspect PID;
- inspect user;
- inspect command;
- verify it belongs to this project.

---

## 17.3 Long-running execution

Use persistent execution where appropriate:

- tmux;
- screen;
- nohup;
- job scheduler if available.

Logs must survive shell disconnects.

---

## 17.4 OOM policy

When OOM occurs:

1. capture exact error;
2. identify whether failure is training or rollout;
3. record batch size / sequence length / model / world size;
4. try the smallest obvious change;
5. rerun smoke test.

Do not simultaneously change:

- batch size;
- sequence length;
- precision;
- model;
- FSDP settings;
- rollout engine.

If OOM remains ambiguous, escalate to Sol.

---

# 18. Experiment Reproducibility

Every run must save:

```text
git commit
timestamp
seed
model
dataset split
training config
reward config
GPU configuration
framework versions
```

Suggested run files:

```text
run_config.yaml
metrics.json
stdout.log
trajectories.jsonl
```

Run names should be descriptive:

```text
qwen1.7b_react_hotpotqa_seed42

qwen1.7b_grpo_hotpotqa_g4_seed42

qwen1.7b_costgrpo_lam0.03_seed42
```

Never silently overwrite runs.

---

# 19. Configuration Discipline

Do not hardcode experimental parameters.

Configs must control:

- model;
- dataset;
- learning rate;
- rollout group size;
- temperature;
- max turns;
- max tokens;
- tool budget;
- reward coefficients;
- seed;
- GPU allocation;
- output directory.

Every run saves the resolved config.

---

# 20. Research Integrity

Never fabricate:

- accuracy;
- EM/F1;
- reward;
- GPU throughput;
- training curves;
- tool-call counts;
- ablation values;
- benchmark scores.

Use:

```text
TBD
```

for unrun experiments and:

```text
FAILED
```

for failed experiments.

Do not cherry-pick one lucky run as proof.

For the final key comparison, use multiple seeds where reasonably affordable.

---

# 21. Failure Investigation Protocol

When an experiment fails:

1. reproduce;
2. isolate;
3. inspect logs;
4. state hypotheses;
5. change one relevant factor;
6. rerun;
7. record conclusion.

Maintain:

```text
docs/debug_log.md
```

Important failures must include:

- symptom;
- root cause if known;
- fix;
- validation;
- whether previous experiments were invalidated.

---

# 22. Negative Results

A failed hypothesis is not a failed project.

Example:

> Linear search penalties reduce tool use but cause severe under-search.

This may motivate:

- adaptive penalty;
- redundant-search penalty;
- difficulty-conditioned budget.

However, Sol must approve the pivot.

Workers must never manipulate experiments to force the desired conclusion.

---

# 23. Repository Structure

Target structure:

```text
EfficientTool-RL/
│
├── README.md
├── AGENTS.md
├── PROGRESS.md
├── requirements.txt
│
├── configs/
│   ├── qwen1.7b_react.yaml
│   ├── qwen1.7b_grpo.yaml
│   ├── qwen4b_grpo.yaml
│   └── cost_grpo.yaml
│
├── data/
│   ├── hotpotqa.py
│   └── gsm8k.py
│
├── tools/
│   ├── base.py
│   ├── search.py
│   └── calculator.py
│
├── environments/
│   └── tool_env.py
│
├── rewards/
│   ├── task_reward.py
│   ├── search_cost.py
│   └── combined_reward.py
│
├── training/
│   ├── train_grpo.sh
│   └── train_cost_grpo.sh
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── trajectory_analysis.py
│
├── tests/
│
├── analysis/
│
├── scripts/
│
├── outputs/
│
└── docs/
```

Adapt as necessary to the chosen upstream framework, but preserve separation between:

- tools;
- environment;
- rewards;
- training;
- evaluation;
- analysis.

---

# 24. Testing Policy

Unit-test deterministic critical components.

At minimum:

- tool parser;
- search environment;
- answer extraction;
- reward calculation;
- cost calculation;
- episode termination.

Reward code receives special scrutiny.

A reward bug can invalidate the full experiment.

---

# 25. Sol Review Checklist for Worker Patches

Before accepting a Luna patch, Sol checks:

- Is the implementation within assigned scope?
- Did it modify unrelated files?
- Are tests present where appropriate?
- Were tests actually run?
- Does the evidence support the worker's claim?
- Did semantics change silently?
- Does the change affect reproducibility?
- Could it leak evaluation information?
- Could it invalidate previous runs?
- Is a follow-up experiment required?

Only then merge/accept.

---

# 26. Progress Tracking

Maintain:

```text
PROGRESS.md
```

Template:

```markdown
# Current Milestone

M2 — ReAct baseline

# Status

IN PROGRESS

# Completed

- [x] Qwen inference
- [x] tool parser
- [x] BM25 search

# In Progress

- [ ] HotpotQA evaluation

# Blockers

- None

# Latest Evidence

- parser tests: 14/14 passed
- 50-episode smoke test completion rate: ...

# Next Actions

1. ...
2. ...
3. ...
```

Sol owns milestone status.

Workers may propose updates but must not declare a milestone accepted.

---

# 27. README Deliverable

The README should be optimized for a recruiter/research engineer scanning quickly.

Recommended first screen:

# EfficientTool-RL

**Train LLM agents to search less, search better, and answer correctly with GRPO.**

Then:

1. architecture figure;
2. short motivation;
3. main result table;
4. one trajectory example;
5. quick start;
6. training;
7. evaluation;
8. ablations;
9. failure analysis;
10. repository structure.

Avoid a wall of text.

---

# 28. Required Final Visualizations

Produce reproducible plotting scripts for:

1. training reward vs steps;
2. held-out EM/F1;
3. average search calls by method;
4. performance-search-cost Pareto curve;
5. optional zero-variance group ratio;
6. optional search-count distribution.

All plots must come from stored experiment outputs.

---

# 29. Resume Deliverable

Create:

```text
docs/resume_summary.md
```

Use real results only.

Suggested structure:

```text
EfficientTool-RL — Multi-turn LLM Agent Reinforcement Learning

- Built a multi-turn Agentic RL pipeline using verl and vLLM/SGLang,
  integrating tool interaction, trajectory rollout and GRPO optimization.

- Designed a cost-aware reward objective to jointly optimize task success
  and retrieval efficiency.

- Reduced average tool calls by XX% while retaining/improving XX task
  performance relative to vanilla GRPO.
```

Never invent `XX`.

---

# 30. Interview Notes Deliverable

Create:

```text
docs/interview_notes.md
```

Explain using the actual implementation:

## GRPO
- grouped rollouts;
- relative advantage;
- critic-free training;
- zero reward variance;
- KL behavior.

## Agent RL
- state;
- action;
- observation;
- trajectory;
- environment;
- episode termination.

## Infrastructure
- rollout vs training;
- vLLM/SGLang;
- distributed policy updates;
- asynchronous rollout;
- long-tail tool latency.

## Reward design
- sparse task reward;
- shaping;
- cost penalty;
- reward hacking;
- performance-efficiency trade-off.

---

# 31. Recommended Sol / Luna Task Allocation

## M0

Sol:
- inspect overall repo;
- choose framework path;
- accept environment plan.

Luna:
- collect environment info;
- generate environment report;
- implement smoke-test script.

---

## M1

Sol:
- approve action protocol and trajectory schema.

Luna workers:
- parser + tests;
- episode loop;
- logging.

---

## M2

Sol:
- approve benchmark split and evaluation semantics.

Luna workers:
- HotpotQA loader;
- BM25 retriever;
- metrics;
- trajectory analysis.

---

## M3

Sol:
- define reward;
- define GRPO config;
- review sanity evidence;
- diagnose ambiguous RL failures.

Luna workers:
- implement reward;
- implement logging;
- wire configs;
- run bounded smoke tests;
- summarize diagnostics.

---

## M4–M5

Sol:
- choose experiment grid;
- decide what results mean;
- approve pivots.

Luna workers:
- launch approved runs;
- monitor logs;
- aggregate outputs;
- compute metrics;
- make plots;
- analyze trajectories.

---

## Finalization

Sol:
- decide claims.

Luna:
- README;
- figures;
- reproducibility docs;
- resume summary;
- interview notes.

---

# 32. Anti-Patterns

The following behavior is forbidden.

## 32.1 Big-bang implementation

Do not implement the entire system before testing one episode.

## 32.2 Silent architecture drift

Do not replace Agent-R1/verl with another stack without Sol approval.

## 32.3 Random hyperparameter thrashing

Do not change five parameters after one bad run.

## 32.4 Feature addiction

Do not add Browser/MCP/Python merely because they look impressive.

## 32.5 Metric-only debugging

Always inspect trajectories, not just aggregate reward.

## 32.6 Reward overengineering

Do not add multiple shaping terms before vanilla GRPO works.

## 32.7 Expensive debugging

Do not use multi-hour GPU runs to test a bug that could be reproduced on 10–50 prompts.

## 32.8 Worker self-approval

A Luna worker may report PASS for its assigned task, but only Sol accepts milestone completion.

---

# 33. Research Decision Ladder

When results are disappointing, follow this order:

1. verify evaluation;
2. verify data split;
3. verify reward;
4. inspect trajectories;
5. verify policy actually updates;
6. inspect rollout diversity;
7. inspect zero-variance groups;
8. inspect tool environment;
9. tune minimal optimization parameters;
10. only then consider changing the research idea.

Do not jump directly to “invent a new method”.

---

# 34. Main Experimental Claims Allowed

A claim may appear in README/resume only if supported by stored evidence.

Examples:

Allowed:

> GRPO improved held-out HotpotQA F1 from A to B.

Allowed:

> A search-cost penalty reduced average tool calls by X% at Y F1.

Allowed:

> High lambda values produced under-search behavior.

Not allowed:

> Our method is more efficient.

unless efficiency is quantitatively defined and measured.

---

# 35. Final Definition of Done

The project is complete only when:

- [ ] environment is documented;
- [ ] Qwen inference works;
- [ ] multi-turn search interaction works;
- [ ] parser/tool tests pass;
- [ ] HotpotQA evaluation works;
- [ ] ReAct baseline is measured;
- [ ] trajectory logs are stored;
- [ ] GRPO sanity gate passes;
- [ ] main GRPO experiment is evaluated;
- [ ] group reward variance is logged;
- [ ] cost-aware reward is implemented;
- [ ] lambda ablation is complete;
- [ ] performance-cost curve exists;
- [ ] reward hacking is analyzed;
- [ ] results are reproducible;
- [ ] README is recruiter-friendly;
- [ ] resume summary exists;
- [ ] interview notes exist;
- [ ] no reported result is fabricated.

---

# 36. Immediate Execution Instruction

Begin at **M0**.

Sol must first:

1. inspect the repository;
2. inspect machine resources;
3. determine whether Agent-R1/verl can be reused cleanly;
4. define the smallest M0 task bundle;
5. delegate bounded environment-inspection work to Luna Max where appropriate;
6. review the returned evidence;
7. load Qwen3-1.7B and run one inference;
8. write/update `docs/environment_report.md`;
9. update `PROGRESS.md`;
10. accept or reject M0.

Do **not** begin GRPO training during M0.

If M0 passes, proceed to M1 using the same pattern:

> **Plan with Sol → delegate bounded work → validate with evidence → accept gate → continue.**

---

# 37. First Message to the Orchestrator

Use this instruction when starting the project:

```text
Read AGENTS.md completely before making changes.

Act as the Sol research orchestrator.

Begin from Milestone M0 only.

Inspect the repository and machine first, then decompose M0 into small,
independently verifiable tasks. Delegate implementation-heavy and
low-ambiguity tasks to Luna Max workers when worker execution is available.

Do not delegate architecture decisions, experiment interpretation, milestone
acceptance, or ambiguous RL debugging.

Do not launch expensive training before the required smoke tests and milestone
gates pass.

For every worker task, define scope, allowed files, validation commands, and
acceptance criteria. Review worker evidence before accepting its changes.

Maintain PROGRESS.md and docs/environment_report.md.

If the current Codex environment does not expose native subagents/workers,
preserve the same role separation logically: perform Sol-level planning and
review first, execute bounded tasks one at a time, and enforce the same gates.

Start now with M0.
```

---

# 38. Governing Rule

When there is tension between speed and scope, choose the smallest experiment
that can answer the current question.

When there is tension between more code and stronger evidence, choose stronger
evidence.

When there is tension between worker autonomy and research correctness, Sol
retains control.

> **Make it correct. Make it train. Make it measurable. Then make it interesting.**
