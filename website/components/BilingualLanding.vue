<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import ShowcaseTabs from './ShowcaseTabs.vue'

type Language = 'zh' | 'en'

const STORAGE_KEY = 'miniagentrl-language'
const language = ref<Language>('zh')
const hydrated = ref(false)

const zh = {
  heroBrand: 'AGENTIC RL · 实战课程',
  heroTitle: '从会调用工具，到真的用 GRPO 训练 Agent。',
  heroLead: '从最小 Tool Call 开始，走过 Multi-turn Interaction、Trajectory 与 Reward，最后完成一次真实的模型参数更新。',
  start: '开始学习',
  github: 'View on GitHub',
  trajectories: 'Trajectory Explorer',
  research: 'Research',
  questionLabel: '问题',
  actionLabel: 'QWEN · 生成的 ACTION',
  observationLabel: 'OBSERVATION',
  answerLabel: '最终答案',
  whyKicker: 'WHY MINIAGENTRL',
  whyTitle: '很多教程教你调用工具，这里继续往下走。',
  whyLead: 'Tool Calling 只是起点。MiniAgentRL 把工具交互、完整轨迹、Reward 和 GRPO 参数更新放进同一条可以观察的学习路径。',
  whyBody: '你会先看到模型文本如何变成 action，再看到 observation 如何改变下一步决策，最后理解这些行为如何转化为训练信号。',
  commonPath: '很多 Agent 教程停在',
  commonPathValue: 'LLM + Tool + Prompt',
  miniPath: 'MiniAgentRL 继续走到',
  miniPathValue: 'Tool → Multi-turn → Reward → GRPO',
  loopNote: '支持 Multi-turn ≠ 模型真的会 Multi-turn。先看一次交互，再理解为什么环境设计会改变行为。',
  loopLink: '学习 Multi-turn →',
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
  pathLead: '从边界、状态和轨迹开始，逐步把 Agent 接回真正的 RL 训练系统。',
  pathCta: '开始完整课程',
  showcaseKicker: 'SEE IT IN ACTION',
  showcaseTitle: '看见 Agent 如何行动，也看见它如何学习。',
  showcaseLead: '切换行为轨迹与 GRPO group，直接观察工具使用、Reward 和相对 Advantage。',
  quickTitle: '三分钟跑通第一次 Tool Call',
  quickLead: '最快的入口不需要 GPU，也不需要下载模型。',
  quickResult: '你会依次看到：',
  quickCourse: '进入完整课程',
  quickQwen: '让 Qwen3 生成 Tool Call',
  quickExplore: '打开 Trajectory Explorer',
  tracksKicker: 'GO FURTHER',
  tracksTitle: '学习系统，或者研究行为。',
  learnTrack: 'Learn Track',
  learnTrackLead: '用小而可观察的例子，走完从 Tool Calling 到一次 GRPO 更新的完整路径。',
  learnTrackItems: ['前半部分 CPU 可运行', 'Qwen3-1.7B 真实模型推理', '实际 verl / vLLM GRPO smoke'],
  researchTrack: 'Research Track',
  researchTrackLead: '在更完整的环境中复现并分析 Agent 的任务能力与工具成本。',
  researchTrackItems: ['Qwen3-8B', 'Hotpot-MT Strict · Natural Bridge-Hard', 'vanilla GRPO → cost-aware Tool RL'],
  researchStatus: 'vanilla GRPO 基线对照已完成；在 Natural Bridge-Hard 上，任务质量和多步检索均有明显提升。下一步是 cost-aware Tool RL。',
  researchLatestKicker: 'LATEST BASELINE · NATURAL BRIDGE-HARD',
  researchLatestTitle: 'Vanilla GRPO 让 Agent 更强，也更主动。',
  researchLatestLead: 'Qwen3-8B · 200 examples · Base → Step 62',
  researchLatestMetrics: [
    ['EM', '32.5%', '51.5%'],
    ['F1', '42.03%', '62.53%'],
    ['Multi-search', '31.5%', '86.0%'],
    ['Useful search', '0.965', '1.445'],
    ['Wasted search', '0.370', '0.515'],
  ],
  researchLatestNote: '答案质量提升的同时，工具调用也增加了；下一步是保留有用探索、减少浪费调用。',
  learnLink: '进入 Learn Track',
  researchLink: '查看 Research Track',
  footer: '从 Tool Calling 开始，真正走完一次 Agent RL。',
  footerNote: '基于 Qwen、verl、vLLM、Transformers 与 HotpotQA 构建。',
}

const en = {
  heroBrand: 'AGENTIC RL · HANDS-ON COURSE',
  heroTitle: 'Build a Tool Agent. Train it with GRPO.',
  heroLead: 'Start from a minimal Tool Call, move through multi-turn interaction, trajectories, and rewards, then complete a real model parameter update.',
  start: 'Start Learning',
  github: 'View on GitHub',
  trajectories: 'Trajectory Explorer',
  research: 'Research',
  questionLabel: 'QUESTION',
  actionLabel: 'QWEN · GENERATED ACTION',
  observationLabel: 'OBSERVATION',
  answerLabel: 'FINAL ANSWER',
  whyKicker: 'WHY MINIAGENTRL',
  whyTitle: 'Many tutorials teach tool use. This one keeps going.',
  whyLead: 'Tool Calling is only the starting point. MiniAgentRL puts tool interaction, complete trajectories, rewards, and GRPO parameter updates on one inspectable learning path.',
  whyBody: 'You will trace model text into an action, see how an observation changes the next decision, and connect that behavior to a trainable signal.',
  commonPath: 'Many Agent tutorials stop at',
  commonPathValue: 'LLM + Tool + Prompt',
  miniPath: 'MiniAgentRL continues to',
  miniPathValue: 'Tool → Multi-turn → Reward → GRPO',
  loopNote: 'Multi-turn capability ≠ multi-turn behavior. See the interaction first, then learn why environment design changes what the policy does.',
  loopLink: 'Learn Multi-turn →',
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
  pathLead: 'Start with boundaries, state, and trajectories. Then reconnect the agent to a real RL training system.',
  pathCta: 'Start the full course',
  showcaseKicker: 'SEE IT IN ACTION',
  showcaseTitle: 'See how the agent acts — and how it learns.',
  showcaseLead: 'Switch between behavior trajectories and a GRPO group to inspect tool use, rewards, and relative advantage.',
  quickTitle: 'Run your first Tool Call in three minutes',
  quickLead: 'The fastest entry point requires no GPU and no model download.',
  quickResult: 'You should see:',
  quickCourse: 'Enter the full course',
  quickQwen: 'Let Qwen3 generate a Tool Call',
  quickExplore: 'Open Trajectory Explorer',
  tracksKicker: 'GO FURTHER',
  tracksTitle: 'Learn the system or study the behavior.',
  learnTrack: 'Learn Track',
  learnTrackLead: 'Use small, inspectable examples to follow the complete path from Tool Calling to a GRPO update.',
  learnTrackItems: ['The first half runs on CPU', 'Real Qwen3-1.7B inference', 'Actual verl / vLLM GRPO smoke'],
  researchTrack: 'Research Track',
  researchTrackLead: 'Reproduce and analyze task capability and tool cost in a more complete environment.',
  researchTrackItems: ['Qwen3-8B', 'Hotpot-MT Strict · Natural Bridge-Hard', 'vanilla GRPO → cost-aware Tool RL'],
  researchStatus: 'The vanilla GRPO baseline comparison is complete; Natural Bridge-Hard shows stronger task quality and more multi-step retrieval. Cost-aware Tool RL is next.',
  researchLatestKicker: 'LATEST BASELINE · NATURAL BRIDGE-HARD',
  researchLatestTitle: 'Vanilla GRPO makes the agent stronger — and more active.',
  researchLatestLead: 'Qwen3-8B · 200 examples · Base → Step 62',
  researchLatestMetrics: [
    ['EM', '32.5%', '51.5%'],
    ['F1', '42.03%', '62.53%'],
    ['Multi-search', '31.5%', '86.0%'],
    ['Useful search', '0.965', '1.445'],
    ['Wasted search', '0.370', '0.515'],
  ],
  researchLatestNote: 'Better answers came with more tool use. The next step is to preserve useful exploration while reducing wasted calls.',
  learnLink: 'Enter Learn Track',
  researchLink: 'Explore Research Track',
  footer: 'Start with Tool Calling. Finish with a real Agentic RL update.',
  footerNote: 'Built with Qwen, verl, vLLM, Transformers, and HotpotQA.',
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
    <section class="mini-hero landing-hero">
      <div class="landing-language-float">
        <div class="landing-language-switch" role="group" :aria-label="language === 'zh' ? '切换语言' : 'Switch language'">
          <button type="button" :class="{ 'is-active': language === 'zh' }" :aria-pressed="language === 'zh'" @click="language = 'zh'">中</button>
          <button type="button" :class="{ 'is-active': language === 'en' }" :aria-pressed="language === 'en'" @click="language = 'en'">EN</button>
        </div>
      </div>

      <div class="mini-hero__copy">
        <div class="mini-hero__brand">
          <img :src="withBase('/logo.svg')" alt="">
          <span>{{ copy.heroBrand }}</span>
        </div>
        <h1>MiniAgentRL <span>{{ copy.heroTitle }}</span></h1>
        <p class="mini-hero__lead">{{ copy.heroLead }}</p>
        <div class="mini-hero__actions">
          <a class="mini-button mini-button--primary" :href="withBase('/learn/')">{{ copy.start }} <span aria-hidden="true">→</span></a>
          <a class="mini-button" href="https://github.com/Idiotyevsky/EfficientTool-RL">{{ copy.github }} <span aria-hidden="true">↗</span></a>
        </div>
        <p class="mini-hero__stack">Qwen3 · Tool Calling · Multi-turn · GRPO · verl · vLLM</p>
        <nav class="landing-hero-links" :aria-label="language === 'zh' ? '更多入口' : 'More links'">
          <a :href="withBase('/playground/trajectories')">{{ copy.trajectories }}</a>
          <span aria-hidden="true">·</span>
          <a :href="withBase('/research/')">{{ copy.research }}</a>
        </nav>
      </div>

      <div class="mini-hero__terminal" :aria-label="copy.trajectories">
        <div class="terminal-bar" aria-hidden="true"><i></i><i></i><i></i></div>
        <div class="terminal-block"><span>{{ copy.questionLabel }}</span><code>What machine did Ada Lovelace write notes about?</code></div>
        <div class="terminal-block is-tool"><span>{{ copy.actionLabel }}</span><code>&lt;tool_call&gt;{"name":"search", ...}&lt;/tool_call&gt;</code></div>
        <div class="terminal-block"><span>{{ copy.observationLabel }}</span><code>Ada Lovelace wrote notes on the Analytical Engine.</code></div>
        <div class="terminal-block is-answer"><span>{{ copy.answerLabel }}</span><code>&lt;answer&gt;Analytical Engine&lt;/answer&gt;</code></div>
      </div>
    </section>

    <section id="why" class="mini-section landing-section landing-why">
      <div class="mini-section__head">
        <span class="section-kicker">{{ copy.whyKicker }}</span>
        <h2>{{ copy.whyTitle }}</h2>
      </div>
      <div class="landing-why-grid">
        <div class="landing-why-copy">
          <p class="landing-lead-copy">{{ copy.whyLead }}</p>
          <p>{{ copy.whyBody }}</p>
          <div class="landing-why-contrast">
            <article>
              <span>{{ copy.commonPath }}</span>
              <strong>{{ copy.commonPathValue }}</strong>
            </article>
            <article class="is-highlight">
              <span>{{ copy.miniPath }}</span>
              <strong>{{ copy.miniPathValue }}</strong>
            </article>
          </div>
          <pre class="landing-pipeline"><code>Tool Calling
     ↓
Multi-turn Agent
     ↓
Trajectory / Reward
     ↓
GRPO
     ↓
Updated Policy</code></pre>
        </div>
        <div class="landing-why-demo">
          <AgentLoopDemo :language="language" />
          <p class="landing-inline-note">{{ copy.loopNote }} <a :href="withBase('/learn/03-multiturn')">{{ copy.loopLink }}</a></p>
        </div>
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
      <p class="landing-section-cta"><a class="mini-button mini-button--primary" :href="withBase('/learn/')">{{ copy.pathCta }} <span aria-hidden="true">→</span></a></p>
    </section>

    <section id="showcase" class="mini-section landing-section landing-showcase">
      <ShowcaseTabs :language="language" :kicker="copy.showcaseKicker" :title="copy.showcaseTitle" :lead="copy.showcaseLead" />
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
      <p class="landing-check-result">{{ copy.quickResult }} <code>Model Output</code> → <code>Parsed Action</code> → <code>Search Observation</code></p>
      <div class="landing-quick-links">
        <a :href="withBase('/learn/')"><span>01</span><strong>{{ copy.quickCourse }}</strong><span aria-hidden="true">→</span></a>
        <a :href="withBase('/learn/02-real-qwen')"><span>02</span><strong>{{ copy.quickQwen }}</strong><span aria-hidden="true">→</span></a>
        <a :href="withBase('/playground/trajectories')"><span>03</span><strong>{{ copy.quickExplore }}</strong><span aria-hidden="true">→</span></a>
      </div>
    </section>

    <section id="go-further" class="mini-section landing-section">
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
          <a class="mini-text-link" :href="withBase('/learn/')">{{ copy.learnLink }} →</a>
        </article>
        <article class="landing-track-card landing-track-card--research">
          <span class="section-kicker">RESEARCH TRACK</span>
          <h3>{{ copy.researchTrack }}</h3>
          <p>{{ copy.researchTrackLead }}</p>
          <ul><li v-for="item in copy.researchTrackItems" :key="item">{{ item }}</li></ul>
          <p class="landing-status">{{ copy.researchStatus }}</p>
          <div class="landing-research-result">
            <span class="section-kicker">{{ copy.researchLatestKicker }}</span>
            <h4>{{ copy.researchLatestTitle }}</h4>
            <p class="landing-research-result__meta">{{ copy.researchLatestLead }}</p>
            <dl>
              <div v-for="metric in copy.researchLatestMetrics" :key="metric[0]">
                <dt>{{ metric[0] }}</dt>
                <dd><span>{{ metric[1] }}</span><b aria-hidden="true">→</b><strong>{{ metric[2] }}</strong></dd>
              </div>
            </dl>
            <p>{{ copy.researchLatestNote }}</p>
          </div>
          <a class="mini-text-link" :href="withBase('/research/')">{{ copy.researchLink }} →</a>
        </article>
      </div>
    </section>

    <footer class="landing-footer">
      <div>
        <strong>{{ copy.footer }}</strong>
        <p>{{ copy.footerNote }}</p>
      </div>
      <nav>
        <a :href="withBase('/learn/')">{{ copy.start }}</a>
        <a :href="withBase('/playground/trajectories')">{{ copy.trajectories }}</a>
        <a :href="withBase('/research/')">{{ copy.research }}</a>
        <a href="https://github.com/Idiotyevsky/EfficientTool-RL">GitHub</a>
      </nav>
    </footer>
  </div>
</template>
