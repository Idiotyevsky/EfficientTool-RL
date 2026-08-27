# 5. Efficient Tool Use

Fewer tool calls are not automatically better. A necessary second search may solve a multi-hop question, while a third search that adds no new evidence is waste.

The project reports five levels of accounting:

| Metric | Meaning |
| --- | --- |
| Attempted | A tool-call opening was emitted |
| Valid | The action parsed as a tool call |
| Executed | The environment accepted and ran it |
| Useful | The search added a previously unseen supporting title |
| Wasted | An executed search added no new supporting title |

Useful and wasted are offline analysis labels. Supporting titles are never placed in the agent observation. This prevents the search tool from revealing gold metadata while still allowing behavior analysis after the episode.

The research question is:

> Can RL reduce wasted tool use without suppressing necessary exploration?

The strict environment is a controlled multi-turn stress test. Natural Bridge-Hard is the secondary, less-filtered evaluation used to check that conclusions do not depend only on the strict candidate filter. See the [research index](../research/README.md) for artifact provenance.

Cost-aware reward is intentionally not presented as finished. The active sequence is vanilla strict GRPO, held-out behavior analysis, then a Sol-approved cost-aware objective and lambda sweep. Candidate penalties must be checked for under-search, duplicate queries, premature answers, and verbosity exploitation. Do not report an efficiency improvement until task quality and executed cost are measured together.
