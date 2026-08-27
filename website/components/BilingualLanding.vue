<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'

type Language = 'zh' | 'en'

const STORAGE_KEY = 'miniagentrl-language'
const language = ref<Language>('zh')
const hydrated = ref(false)

const zh = {
  languageLabel: '语言',
  heroBrand: '一条可观察、可运行的 Agent RL 路线',
  heroTitle: '从 Tool Calling 到真正的 Agent RL。',
  heroLead: '从一个最小 Tool Agent 开始，逐步理解 Multi-turn Interaction、Trajectory、Reward 与 GRPO，最后亲手完成一次真实的参数更新。',
  start: '开始学习',
  docs: '在线文档',
  trajectories: 'Trajectory Explorer',
  research: 'Research',
  github: '在 GitHub 查看',
  questionLabel: '问题',
  actionLabel: 'QWEN · 生成的 Action',
  observationLabel: 'Observation',
  answerLabel: '最终答案',
  whyKicker: 'WHY MINIAGENTRL',
  whyTitle: '为什么做 MiniAgentRL？',
  whyParagraphs: [
    '很多 Agent 教程讲到 Tool Calling 就结束了。',
    '给模型定义几个工具，写一个 ReAct Prompt，让模型能够搜索、调用 API 或执行函数——这当然是 Agent 的重要部分。但如果继续追问：Agent 到底是怎么训练出来的？问题就会变成另一套东西。',
    '一个真正进入强化学习阶段的 Agent，需要产生完整的交互轨迹，需要定义 Reward，需要从同一个 Prompt 中采样多条 Rollout，还需要把这些奖励信号最终变成模型参数的更新。',
    'MiniAgentRL 把这条链路完整拆开。你不需要一开始就理解 Ray、vLLM、FSDP 或复杂的分布式训练；课程先从小而可观察的例子开始，再逐步接回同一套 Agent RL 系统。',
  ],
  buildKicker: 'FOUR MODULES · ONE SYSTEM',
  buildTitle: '你会亲手搭出什么？',
  buildLead: '从“会调用”走到“会优化”，每个模块都回答一个具体问题。',
  modules: [
    { eyebrow: '01 · BUILD', title: 'Tool Calling', goal: '看清模型文本、结构化 action 与工具执行之间的边界。', level: '入门', prerequisite: 'Python · LLM inference', href: '/learn/01-tool-calling' },
    { eyebrow: '02 · INTERACT', title: 'Multi-turn Agent', goal: '让 Observation 回到上下文，并把多轮行为变成可检查的 trajectory。', level: '入门+', prerequisite: 'Tool Calling', href: '/learn/03-multiturn' },
    { eyebrow: '03 · TRAIN', title: 'GRPO Training', goal: '从 grouped rollouts、Reward 与 Advantage 走到一次真实参数更新。', level: '进阶', prerequisite: 'Rollout · Reward', href: '/learn/06-grpo' },
    { eyebrow: '04 · OPTIMIZE', title: 'Efficient Tool Use', goal: '区分必要探索与浪费调用，理解任务质量与工具成本的关系。', level: '进阶', prerequisite: 'GRPO · evaluation', href: '/learn/08-efficient-tools' },
  ],
  pathKicker: '00 → 08',
  pathTitle: '一条连续的学习路线',
  pathLead: '不是九篇互不相关的文章：先理解边界，再增加系统复杂度。',
  pathRows: [
    ['00 · Start', '环境检查与 Agent RL 全景图', 'CPU'],
    ['01 · Tool Calling', '从模型文本到真实工具执行', 'CPU'],
    ['02 · Real Qwen', '让 Qwen3 真正生成 Tool Call', 'GPU'],
    ['03 · Multi-turn', 'Observation 如何进入下一状态', 'CPU'],
    ['04 · ReAct + HotpotQA', '在真实多跳 QA 上运行 Agent', 'GPU'],
    ['05 · Rollout & Reward', '从完整 trajectory 计算训练信号', 'CPU'],
    ['06 · GRPO', '从 Reward 到 Advantage 与 Policy Update', 'CPU'],
    ['07 · Real Update', '真正完成一次 GRPO 参数更新', 'GPU'],
    ['08 · Efficient Tools', '分析 useful 与 wasted Tool Calls', 'CPU'],
  ],
  pathCta: '开始 00 → 08 课程',
  behaviorKicker: 'BEHAVIOR IS DATA',
  behaviorTitle: '看见 Agent 的完整行为，而不只是最终答案',
  behaviorLead: '只看 Final Answer，会丢掉 Agent 最重要的信息。打开 Explorer，切换成功多跳、过早作答、重复搜索和无效 action。',
  behaviorCta: '打开 Trajectory Explorer',
  multiKicker: 'A SIGNATURE LESSON',
  multiTitle: '支持 Multi-turn，不代表真的会发生 Multi-turn',
  multiParagraph: '即使代码允许 Agent 连续执行 5 个 turn，模型也可能永远只搜索一次。如果一次检索已经返回全部需要的信息，模型没有理由继续搜索。',
  capability: 'Capability',
  capabilityDetail: '系统是否允许继续交互？',
  necessity: 'Necessity',
  necessityDetail: '信息结构是否需要下一步？',
  behavior: 'Behavior',
  behaviorDetail: '模型是否选择继续使用工具？',
  multiConclusion: 'Multi-turn 不是把 max_turns 调大，而是让环境和任务真的产生后续信息需求。',
  oldStructure: '一次 Search(top_k=3)',
  oldDetail: '一个 Observation 可能同时返回全部 supporting passage。',
  newStructure: '每次只返回 top_k=1',
  newDetail: '第二次检索才有机会提供新的证据。',
  grpoKicker: 'FROM REWARD TO UPDATE',
  grpoTitle: 'Reward 是怎么变成参数更新的？',
  grpoParagraph: '理解 GRPO 时，很容易停在“同一个问题采样几条答案，然后算 Reward”。但这还不是训练。课程会把 Reward、Advantage、Policy Ratio、Clipping、Gradient 和 Optimizer Step 串成一条可运行的链路。',
  grpoCta: '学习 GRPO',
  updateCta: '运行一次真实 Update',
  efficientKicker: 'EFFICIENT TOOL USE',
  efficientTitle: '工具调用越少，Agent 就越高效吗？',
  efficientParagraph: '不一定。一个 Agent 完全不搜索、直接猜答案，工具成本确实是 0，但这显然不叫高效。更合理的问题是：哪些调用是必要的信息获取，哪些调用只是重复或浪费？',
  efficientConclusion: '目标不是简单减少工具调用，而是保留必要的信息获取，减少没有带来新信息的调用。',
  quickTitle: 'Quick Start',
  quickLead: '最快的入口不需要 GPU，也不需要下载模型。',
  quickNext: ['开始完整 00 → 08 课程', '让 Qwen3 真正生成 Tool Call', '查看 Trajectory Explorer', '运行一次 GRPO Update'],
  tracksKicker: 'TWO CONNECTED TRACKS',
  tracksTitle: '学习系统，或者研究行为',
  learnTrack: 'Learn Track',
  learnTrackLead: '面向想真正搞懂 Agent RL 的学习者。',
  learnTrackItems: ['每个概念都能单独观察', '前半部分 CPU 就能运行', '使用 Qwen3-1.7B 完成真实模型推理', '最后完成一次真实 GRPO 参数更新'],
  researchTrack: 'Research Track',
  researchTrackLead: '面向想进一步复现实验、分析 Agent 行为的读者。',
  researchTrackItems: ['Qwen3-8B', 'Hotpot-MT Strict 与 Natural Bridge-Hard', 'deterministic BM25 search', 'native multi-turn rollout · verl · vLLM · FSDP', 'EM / F1 与 attempted / valid / executed / useful / wasted'],
  researchStatus: 'Research Track 当前正在进行 Qwen3-8B 的 vanilla GRPO 对照实验。Cost-aware Tool RL 会在基线评估完成后继续推进。',
  researchQuestionKicker: 'RESEARCH QUESTION',
  researchQuestionTitle: '除了教学，项目还在研究什么？',
  researchQuestion: '强化学习能否让 Multi-turn Tool Agent 在保持任务能力的同时，减少没有信息增益的工具调用？如果工具调用减少但准确率同时下降，那并不是有意义的效率提升。',
  structureTitle: 'Project Structure',
  structureIntro: '对外项目名称使用 MiniAgentRL；为了保持现有训练配置、实验记录和 Python import 的兼容性，核心 package 仍保留 efficienttool_rl。',
  techTitle: 'Tech Stack',
  techRows: [
    ['Qwen3', 'Agent Policy'],
    ['Transformers', '本地模型推理'],
    ['BM25', '可复现的 Search Environment'],
    ['HotpotQA', 'Multi-hop QA 任务'],
    ['verl', 'GRPO Training'],
    ['vLLM', 'Rollout Generation'],
    ['Ray', '分布式执行'],
    ['PyTorch / FSDP', '模型训练'],
    ['VitePress', '在线学习网站'],
  ],
  audienceTitle: '适合谁？',
  audience: 'MiniAgentRL 更适合已经会使用 Python、知道 Transformer / LLM 基本概念、跑过 Hugging Face 模型推理，并听说过 ReAct、PPO 或 GRPO 的读者。它是一条从 LLM inference 走向 Agentic RL / LLM Post-training 的实践路线。',
  acknowledgementsTitle: 'Acknowledgements',
  acknowledgements: 'MiniAgentRL 构建在 Qwen、verl、vLLM、Hugging Face Transformers 与 HotpotQA 之上。请遵守相关上游项目与数据集的许可证和引用要求。',
  licenseTitle: 'License',
  license: '当前仓库尚未指定项目级许可证。在正式补充 LICENSE 之前，请不要默认获得对 MiniAgentRL 项目代码进行再分发、修改或衍生使用的授权。',
  footer: '从 Tool Calling 开始，真正走完一次 Agent RL。',
}

const en = {
  languageLabel: 'Language',
  heroBrand: 'An inspectable, runnable path into Agentic RL',
  heroTitle: 'Learn Agentic RL by building it.',
  heroLead: 'Start from a minimal tool-using agent, then build your way through multi-turn interaction, trajectories, rewards, and GRPO — all the way to a real parameter update.',
  start: 'Start Learning',
  docs: 'Documentation',
  trajectories: 'Trajectory Explorer',
  research: 'Research',
  github: 'View on GitHub',
  questionLabel: 'QUESTION',
  actionLabel: 'QWEN · GENERATED ACTION',
  observationLabel: 'OBSERVATION',
  answerLabel: 'FINAL ANSWER',
  whyKicker: 'WHY MINIAGENTRL',
  whyTitle: 'Why MiniAgentRL?',
  whyParagraphs: [
    'Many Agent tutorials stop at Tool Calling.',
    'You define a few tools, write a ReAct-style prompt, and let the model search, call APIs, or execute functions. That is an important part of building an agent. But the next question is much more interesting: how is the agent actually trained?',
    'Once reinforcement learning enters the picture, the agent must produce complete interaction trajectories. Those trajectories need rewards. Multiple rollouts from the same prompt need to be compared. And eventually, those signals must change the model parameters.',
    'MiniAgentRL breaks that full pipeline into understandable pieces. You do not need to understand Ray, vLLM, FSDP, or distributed training on day one; the course starts with small, inspectable examples and gradually reconnects them into the same Agentic RL system.',
  ],
  buildKicker: 'FOUR MODULES · ONE SYSTEM',
  buildTitle: 'What will you build?',
  buildLead: 'Move from “the agent can call a tool” to “the agent can learn how to use it.”',
  modules: [
    { eyebrow: '01 · BUILD', title: 'Tool Calling', goal: 'Understand the boundary between model text, structured actions, and tool execution.', level: 'Beginner', prerequisite: 'Python · LLM inference', href: '/learn/01-tool-calling' },
    { eyebrow: '02 · INTERACT', title: 'Multi-turn Agent', goal: 'Feed observations back into context and make multi-turn behavior inspectable.', level: 'Beginner+', prerequisite: 'Tool Calling', href: '/learn/03-multiturn' },
    { eyebrow: '03 · TRAIN', title: 'GRPO Training', goal: 'Connect grouped rollouts, rewards, and advantages to a real parameter update.', level: 'Advanced', prerequisite: 'Rollout · Reward', href: '/learn/06-grpo' },
    { eyebrow: '04 · OPTIMIZE', title: 'Efficient Tool Use', goal: 'Separate necessary exploration from wasted calls and measure the trade-off.', level: 'Advanced', prerequisite: 'GRPO · evaluation', href: '/learn/08-efficient-tools' },
  ],
  pathKicker: '00 → 08',
  pathTitle: 'One continuous learning path',
  pathLead: 'Not nine unrelated articles: understand the boundaries first, then add system complexity.',
  pathRows: [
    ['00 · Start', 'Environment check and the Agentic RL map', 'CPU'],
    ['01 · Tool Calling', 'From generated text to actual tool execution', 'CPU'],
    ['02 · Real Qwen', 'Let Qwen3 generate a real Tool Call', 'GPU'],
    ['03 · Multi-turn', 'How observations become the next state', 'CPU'],
    ['04 · ReAct + HotpotQA', 'Run the agent on real multi-hop QA', 'GPU'],
    ['05 · Rollout & Reward', 'Turn complete trajectories into reward', 'CPU'],
    ['06 · GRPO', 'From reward to advantage and policy update', 'CPU'],
    ['07 · Real Update', 'Run an actual GRPO parameter update', 'GPU'],
    ['08 · Efficient Tools', 'Analyze useful and wasted tool calls', 'CPU'],
  ],
  pathCta: 'Start the 00 → 08 course',
  behaviorKicker: 'BEHAVIOR IS DATA',
  behaviorTitle: 'See the agent’s behavior, not just its final answer',
  behaviorLead: 'A final answer hides most of what matters. Open the Explorer and switch between successful multi-hop search, premature answers, repeated queries, and invalid actions.',
  behaviorCta: 'Open Trajectory Explorer',
  multiKicker: 'A SIGNATURE LESSON',
  multiTitle: 'Supporting multiple turns does not mean multiple turns will happen',
  multiParagraph: 'Even if the code allows five turns, the model may still search only once. If one retrieval exposes everything it needs, there is no reason to search again.',
  capability: 'Capability',
  capabilityDetail: 'Can the system continue interacting?',
  necessity: 'Necessity',
  necessityDetail: 'Does the information structure require another step?',
  behavior: 'Behavior',
  behaviorDetail: 'Does the policy choose to keep using tools?',
  multiConclusion: 'Multi-turn is not about increasing max_turns. It is about creating real information demand for the next turn.',
  oldStructure: 'One Search(top_k=3)',
  oldDetail: 'A single observation may expose every supporting passage.',
  newStructure: 'Only top_k=1 per search',
  newDetail: 'Later searches have a chance to provide new evidence.',
  grpoKicker: 'FROM REWARD TO UPDATE',
  grpoTitle: 'How does reward become a parameter update?',
  grpoParagraph: 'It is easy to stop the GRPO explanation at “sample several answers for one prompt and compute rewards.” That is not training yet. The course connects reward, advantage, policy ratio, clipping, gradient, and optimizer step into one runnable chain.',
  grpoCta: 'Learn GRPO',
  updateCta: 'Run a Real Update',
  efficientKicker: 'EFFICIENT TOOL USE',
  efficientTitle: 'Does fewer tool usage always mean a more efficient agent?',
  efficientParagraph: 'Not necessarily. An agent that never searches and simply guesses has zero tool cost, but that does not make it efficient. The better question is: which calls gather necessary information, and which calls are redundant or wasteful?',
  efficientConclusion: 'The goal is not merely to reduce tool usage. It is to preserve necessary information gathering while reducing calls that add no new information.',
  quickTitle: 'Quick Start',
  quickLead: 'The fastest entry point requires no GPU and no model download.',
  quickNext: ['Start the full 00 → 08 course', 'Let Qwen3 generate a real Tool Call', 'Open the Trajectory Explorer', 'Run a GRPO update'],
  tracksKicker: 'TWO CONNECTED TRACKS',
  tracksTitle: 'Learn the system or study the behavior',
  learnTrack: 'Learn Track',
  learnTrackLead: 'For readers who want to understand Agentic RL end to end.',
  learnTrackItems: ['Each concept can be inspected independently', 'The early lessons run on CPU', 'Qwen3-1.7B is used for real model inference', 'The course ends with an actual GRPO parameter update'],
  researchTrack: 'Research Track',
  researchTrackLead: 'For readers who want to reproduce experiments and analyze agent behavior.',
  researchTrackItems: ['Qwen3-8B', 'Hotpot-MT Strict and Natural Bridge-Hard', 'Deterministic BM25 search', 'Native multi-turn rollout · verl · vLLM · FSDP', 'EM / F1 and attempted / valid / executed / useful / wasted'],
  researchStatus: 'The Research Track is currently evaluating vanilla GRPO on Qwen3-8B. Cost-aware Tool RL will follow after the baseline evaluation is complete.',
  researchQuestionKicker: 'RESEARCH QUESTION',
  researchQuestionTitle: 'What is the project studying beyond the course?',
  researchQuestion: 'Can reinforcement learning reduce tool calls that add no new information while preserving the capability of a multi-turn tool agent? If tool usage decreases while accuracy also drops, that is not a meaningful efficiency improvement.',
  structureTitle: 'Project Structure',
  structureIntro: 'The public project name is MiniAgentRL. To preserve compatibility with existing training configurations, experiment records, and Python imports, the core package remains efficienttool_rl.',
  techTitle: 'Tech Stack',
  techRows: [
    ['Qwen3', 'Agent policy'],
    ['Transformers', 'Local model inference'],
    ['BM25', 'Reproducible search environment'],
    ['HotpotQA', 'Multi-hop QA tasks'],
    ['verl', 'GRPO training'],
    ['vLLM', 'Rollout generation'],
    ['Ray', 'Distributed execution'],
    ['PyTorch / FSDP', 'Model training'],
    ['VitePress', 'Learning website'],
  ],
  audienceTitle: 'Who is this for?',
  audience: 'MiniAgentRL is designed for readers who know Python, understand basic Transformer / LLM concepts, have run Hugging Face inference, and have heard of ReAct, PPO, or GRPO. It is a practical path from LLM inference to Agentic RL and LLM post-training.',
  acknowledgementsTitle: 'Acknowledgements',
  acknowledgements: 'MiniAgentRL builds on Qwen, verl, vLLM, Hugging Face Transformers, and HotpotQA. Please follow the licensing and citation requirements of the corresponding upstream projects and datasets.',
  licenseTitle: 'License',
  license: 'A project-level license has not yet been specified. Until a LICENSE is added, please do not assume permission to redistribute, modify, or create derivative works from MiniAgentRL itself.',
  footer: 'Start with Tool Calling. Finish with a real Agentic RL update.',
}

const copy = computed(() => (language.value === 'zh' ? zh : en))
const locale = computed(() => (language.value === 'zh' ? 'zh-CN' : 'en'))

onMounted(() => {
  const saved = window.localStorage.getItem(STORAGE_KEY)
  if (saved === 'zh' || saved === 'en') language.value = saved
  hydrated.value = true
})

watch(language, (value) => {
  if (hydrated.value) window.localStorage.setItem(STORAGE_KEY, value)
})
</script>

<template>
  <div class="mini-home bilingual-landing" :lang="locale">
    <div class="landing-language-bar">
      <span>{{ copy.languageLabel }}</span>
      <div class="landing-language-switch" role="group" :aria-label="copy.languageLabel">
        <button type="button" :class="{ 'is-active': language === 'zh' }" :aria-pressed="language === 'zh'" @click="language = 'zh'">中文</button>
        <button type="button" :class="{ 'is-active': language === 'en' }" :aria-pressed="language === 'en'" @click="language = 'en'">English</button>
      </div>
    </div>

    <section class="mini-hero landing-hero">
      <div class="mini-hero__copy">
        <div class="mini-hero__brand">
          <img :src="withBase('/logo.svg')" alt="">
          <span>{{ copy.heroBrand }}</span>
        </div>
        <h1>MiniAgentRL <span>{{ copy.heroTitle }}</span></h1>
        <p class="mini-hero__lead">{{ copy.heroLead }}</p>
        <div class="mini-hero__actions">
          <a class="mini-button mini-button--primary" :href="withBase('/learn/')">{{ copy.start }}</a>
          <a class="mini-button" :href="withBase('/learn/01-tool-calling')">{{ copy.docs }}</a>
          <a class="mini-button" :href="withBase('/playground/trajectories')">{{ copy.trajectories }}</a>
          <a class="mini-button" :href="withBase('/research/')">{{ copy.research }}</a>
        </div>
        <p class="mini-hero__stack">Qwen3 · Tool Calling · Multi-turn · GRPO · verl · vLLM</p>
        <a class="landing-github-link" href="https://github.com/Idiotyevsky/EfficientTool-RL">{{ copy.github }} <span aria-hidden="true">↗</span></a>
      </div>
      <div class="mini-hero__terminal" :aria-label="copy.trajectories">
        <div class="terminal-bar" aria-hidden="true"><i></i><i></i><i></i></div>
        <div class="terminal-block"><span>{{ copy.questionLabel }}</span><code>What machine did Ada Lovelace write notes about?</code></div>
        <div class="terminal-block is-tool"><span>{{ copy.actionLabel }}</span><code>&lt;tool_call&gt;{"name":"search", ...}&lt;/tool_call&gt;</code></div>
        <div class="terminal-block"><span>{{ copy.observationLabel }}</span><code>Ada Lovelace wrote notes on the Analytical Engine.</code></div>
        <div class="terminal-block is-answer"><span>{{ copy.answerLabel }}</span><code>&lt;answer&gt;Analytical Engine&lt;/answer&gt;</code></div>
      </div>
    </section>

    <section class="mini-section landing-section landing-visual-section">
      <AgentLoopDemo :language="language" />
    </section>

    <section id="why" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.whyKicker }}</span>
        <h2>{{ copy.whyTitle }}</h2>
      </div>
      <div class="landing-two-column">
        <div class="landing-prose">
          <p v-for="paragraph in copy.whyParagraphs" :key="paragraph">{{ paragraph }}</p>
        </div>
        <pre class="landing-pipeline"><code>Tool Calling
     ↓
Multi-turn Agent
     ↓
Trajectory / Rollout
     ↓
Reward
     ↓
GRPO
     ↓
Updated Policy</code></pre>
      </div>
    </section>

    <section class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.buildKicker }}</span>
        <h2>{{ copy.buildTitle }}</h2>
        <p>{{ copy.buildLead }}</p>
      </div>
      <div class="landing-module-grid">
        <a v-for="module in copy.modules" :key="module.title" class="landing-module-card" :href="withBase(module.href)">
          <span class="landing-module-card__eyebrow">{{ module.eyebrow }}</span>
          <h3>{{ module.title }}</h3>
          <p>{{ module.goal }}</p>
          <dl>
            <div><dt>{{ language === 'zh' ? '难度' : 'Level' }}</dt><dd>{{ module.level }}</dd></div>
            <div><dt>{{ language === 'zh' ? '前置' : 'Prerequisite' }}</dt><dd>{{ module.prerequisite }}</dd></div>
          </dl>
          <span class="landing-module-card__link">{{ language === 'zh' ? '进入模块' : 'Open module' }} <span aria-hidden="true">→</span></span>
        </a>
      </div>
    </section>

    <section id="learning-path" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.pathKicker }}</span>
        <h2>{{ copy.pathTitle }}</h2>
        <p>{{ copy.pathLead }}</p>
      </div>
      <CourseMap :language="language" />
      <div class="landing-path-table-wrap">
        <table class="landing-path-table">
          <thead>
            <tr>
              <th>Chapter</th>
              <th>{{ language === 'zh' ? '你会学什么' : 'What you learn' }}</th>
              <th>{{ language === 'zh' ? '环境' : 'Runtime' }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in copy.pathRows" :key="row[0]">
              <td><strong>{{ row[0] }}</strong></td>
              <td>{{ row[1] }}</td>
              <td><code>{{ row[2] }}</code></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p><a class="mini-button mini-button--primary" :href="withBase('/learn/')">{{ copy.pathCta }}</a></p>
    </section>

    <section id="trajectory" class="mini-section landing-section landing-visual-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.behaviorKicker }}</span>
        <h2>{{ copy.behaviorTitle }}</h2>
        <p>{{ copy.behaviorLead }}</p>
      </div>
      <TrajectoryExplorer :language="language" />
      <p><a class="mini-button mini-button--primary" :href="withBase('/playground/trajectories')">{{ copy.behaviorCta }}</a></p>
    </section>

    <section id="multiturn" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.multiKicker }}</span>
        <h2>{{ copy.multiTitle }}</h2>
        <p>{{ copy.multiParagraph }}</p>
      </div>
      <div class="compare-panel landing-compare-panel">
        <article>
          <span class="section-kicker">EASY INFORMATION STRUCTURE</span>
          <h3>{{ copy.oldStructure }}</h3>
          <p>{{ copy.oldDetail }}</p>
          <code>Question → Search(top3) → Answer</code>
        </article>
        <article class="is-new">
          <span class="section-kicker">CONTROLLED HOTPOT-MT</span>
          <h3>{{ copy.newStructure }}</h3>
          <p>{{ copy.newDetail }}</p>
          <code>Question → Search₁ → Observation₁ → Search₂ → Answer</code>
        </article>
      </div>
      <ConceptFlow :items="[
        { label: copy.capability, detail: copy.capabilityDetail },
        { label: copy.necessity, detail: copy.necessityDetail, tone: 'tool' },
        { label: copy.behavior, detail: copy.behaviorDetail, tone: 'agent' }
      ]" />
      <p class="landing-callout">{{ copy.multiConclusion }}</p>
    </section>

    <section id="grpo" class="mini-section landing-section landing-visual-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.grpoKicker }}</span>
        <h2>{{ copy.grpoTitle }}</h2>
        <p>{{ copy.grpoParagraph }}</p>
      </div>
      <GRPOGroupDemo :language="language" />
      <div class="landing-two-column landing-grpo-explanation">
        <pre class="landing-pipeline"><code>Rollout
   ↓
Reward
   ↓
Advantage
   ↓
Policy Ratio
   ↓
Clipping / KL
   ↓
Gradient
   ↓
Optimizer Step</code></pre>
        <pre class="landing-config"><code>actor_rollout_ref:
  rollout:
    n: 4

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true</code></pre>
      </div>
      <div class="landing-action-row">
        <a class="mini-button" :href="withBase('/learn/06-grpo')">{{ copy.grpoCta }}</a>
        <a class="mini-button mini-button--primary" :href="withBase('/learn/07-grpo-smoke')">{{ copy.updateCta }}</a>
      </div>
    </section>

    <section id="efficient-tools" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.efficientKicker }}</span>
        <h2>{{ copy.efficientTitle }}</h2>
        <p>{{ copy.efficientParagraph }}</p>
      </div>
      <div class="landing-efficiency-flow">
        <span>attempted</span><b>→</b><span>valid</span><b>→</b><span>executed</span><b>→</b><span class="is-positive">useful</span><b>/</b><span class="is-negative">wasted</span>
      </div>
      <p class="landing-callout">{{ copy.efficientConclusion }}</p>
      <p><a class="mini-button mini-button--primary" :href="withBase('/learn/08-efficient-tools')">{{ language === 'zh' ? '学习 Efficient Tool Use' : 'Learn Efficient Tool Use' }}</a></p>
    </section>

    <section id="quick-start" class="mini-section landing-section landing-quick-start">
      <div class="mini-section__head">
        <span class="section-kicker">FIRST 3 MINUTES</span>
        <h2>{{ copy.quickTitle }}</h2>
        <p>{{ copy.quickLead }}</p>
      </div>
      <pre class="landing-command"><code>git clone https://github.com/Idiotyevsky/EfficientTool-RL.git
cd EfficientTool-RL
pip install -e ".[test]"
PYTHONPATH=src python examples/01_tool_calling.py</code></pre>
      <p class="landing-check-result">{{ language === 'zh' ? '你会依次看到：' : 'You should see:' }} <code>Model Output</code> → <code>Parsed Action</code> → <code>Search Observation</code></p>
      <div class="landing-next-grid">
        <a v-for="(next, index) in copy.quickNext" :key="next" :href="withBase(['/learn/', '/learn/02-real-qwen', '/playground/trajectories', '/learn/07-grpo-smoke'][index])">
          <span>0{{ index + 1 }}</span>
          <strong>{{ next }}</strong>
          <span aria-hidden="true">→</span>
        </a>
      </div>
    </section>

    <section id="research" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.tracksKicker }}</span>
        <h2>{{ copy.tracksTitle }}</h2>
      </div>
      <div class="landing-track-grid">
        <article class="landing-track-card">
          <span class="section-kicker">LEARN TRACK</span>
          <h3>{{ copy.learnTrack }}</h3>
          <p>{{ copy.learnTrackLead }}</p>
          <ul><li v-for="item in copy.learnTrackItems" :key="item">{{ item }}</li></ul>
          <a class="mini-text-link" :href="withBase('/learn/')">{{ language === 'zh' ? '进入 Learn Track' : 'Enter Learn Track' }} →</a>
        </article>
        <article class="landing-track-card landing-track-card--research">
          <span class="section-kicker">RESEARCH TRACK</span>
          <h3>{{ copy.researchTrack }}</h3>
          <p>{{ copy.researchTrackLead }}</p>
          <ul><li v-for="item in copy.researchTrackItems" :key="item">{{ item }}</li></ul>
          <p class="landing-status">{{ copy.researchStatus }}</p>
          <a class="mini-text-link" :href="withBase('/research/')">{{ language === 'zh' ? '查看 Research Track' : 'Explore Research Track' }} →</a>
        </article>
      </div>
    </section>

    <section class="mini-section landing-section landing-research-question">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.researchQuestionKicker }}</span>
        <h2>{{ copy.researchQuestionTitle }}</h2>
        <p>{{ copy.researchQuestion }}</p>
      </div>
    </section>

    <section id="structure" class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">UNDER THE SURFACE</span>
        <h2>{{ copy.structureTitle }}</h2>
        <p>{{ copy.structureIntro }}</p>
      </div>
      <pre class="landing-structure"><code>MiniAgentRL
│
├── src/efficienttool_rl/  # Agent, protocol, tools, rewards, evaluation
├── examples/              # minimal runnable examples
├── configs/               # Agent / GRPO configurations
├── scripts/               # data, training, and evaluation entry points
├── website/               # learning website
├── research/              # research design and experiment notes
├── tests/                 # unit and integration tests
└── assets/                # README and documentation visuals</code></pre>
    </section>

    <section class="mini-section landing-section">
      <div class="mini-section__head">
        <span class="section-kicker">TOOLS OF THE TRADE</span>
        <h2>{{ copy.techTitle }}</h2>
      </div>
      <div class="landing-tech-grid">
        <div v-for="row in copy.techRows" :key="row[0]" class="landing-tech-item">
          <strong>{{ row[0] }}</strong>
          <span>{{ row[1] }}</span>
        </div>
      </div>
    </section>

    <section class="mini-section landing-section landing-audience">
      <div class="mini-section__head">
        <span class="section-kicker">WHO THIS IS FOR</span>
        <h2>{{ copy.audienceTitle }}</h2>
        <p>{{ copy.audience }}</p>
      </div>
    </section>

    <section class="mini-section landing-section landing-footer-info">
      <div>
        <span class="section-kicker">{{ copy.acknowledgementsTitle }}</span>
        <p>{{ copy.acknowledgements }}</p>
      </div>
      <div>
        <span class="section-kicker">{{ copy.licenseTitle }}</span>
        <p>{{ copy.license }}</p>
      </div>
    </section>

    <footer class="landing-footer">
      <strong>{{ copy.footer }}</strong>
      <nav>
        <a :href="withBase('/learn/')">{{ copy.start }}</a>
        <a :href="withBase('/playground/trajectories')">{{ copy.trajectories }}</a>
        <a :href="withBase('/research/')">{{ copy.research }}</a>
        <a href="https://github.com/Idiotyevsky/EfficientTool-RL">GitHub</a>
      </nav>
    </footer>
  </div>
</template>
