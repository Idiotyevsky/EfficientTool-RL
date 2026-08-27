---
layout: page
title: MiniAgentRL
description: Learn Agentic RL by building it — 从 Tool Calling 到真正的 GRPO 参数更新。
sidebar: false
aside: false
editLink: false
lastUpdated: false
---

<div class="mini-home">
  <section class="mini-hero">
    <div class="mini-hero__copy">
      <div class="mini-hero__brand">
        <img src="/logo.svg" alt="">
        <span>A real Agentic RL learning system</span>
      </div>
      <h1>MiniAgentRL <span>Learn Agentic RL by building it.</span></h1>
      <p class="mini-hero__lead">从 Tool Calling 到 Multi-turn Agent，再到真正的 GRPO 参数更新。入口足够轻，底层仍是 Qwen3、verl 与 vLLM 的真实系统。</p>
      <div class="mini-hero__actions">
        <a class="mini-button mini-button--primary" href="./learn/">开始学习</a>
        <a class="mini-button" href="https://github.com/Idiotyevsky/EfficientTool-RL">View on GitHub</a>
      </div>
      <p class="mini-hero__stack">Qwen3 · Tool Calling · Multi-turn · GRPO · verl · vLLM</p>
    </div>
    <div class="mini-hero__terminal" aria-label="一次工具调用的输出预览">
      <div class="terminal-bar" aria-hidden="true"><i></i><i></i><i></i></div>
      <div class="terminal-block"><span>QUESTION</span><code>What machine did Ada Lovelace write notes about?</code></div>
      <div class="terminal-block is-tool"><span>QWEN · GENERATED ACTION</span><code>&lt;tool_call&gt;{"name":"search", ...}&lt;/tool_call&gt;</code></div>
      <div class="terminal-block"><span>OBSERVATION</span><code>Ada Lovelace wrote notes on the Analytical Engine.</code></div>
      <div class="terminal-block is-answer"><span>FINAL ANSWER</span><code>&lt;answer&gt;Analytical Engine&lt;/answer&gt;</code></div>
    </div>
  </section>

  <section class="mini-section"><AgentLoopDemo /></section>

  <section class="mini-section">
    <div class="mini-section__head">
      <span class="section-kicker">FOUR MODULES · ONE SYSTEM</span>
      <h2>从“会调用”走到“会优化”</h2>
      <p>每个模块都回答一个具体问题，并最终汇入同一条真实 Agent RL pipeline。</p>
    </div>
    <div class="module-grid">
      <LessonCard eyebrow="BUILD" title="Tool Calling" goal="看清模型文本、结构化 action 与工具执行之间的边界。" href="/learn/01-tool-calling" level="入门" prerequisite="Python · LLM inference" />
      <LessonCard eyebrow="INTERACT" title="Multi-turn Agent" goal="让 Observation 回到上下文，并把多轮行为变成可检查的 trajectory。" href="/learn/03-multiturn" level="入门+" prerequisite="Tool Calling" />
      <LessonCard eyebrow="TRAIN" title="GRPO" goal="从 grouped rollouts、Reward 与 Advantage 走到一次真实参数更新。" href="/learn/06-grpo" level="进阶" prerequisite="Rollout · Reward" />
      <LessonCard eyebrow="OPTIMIZE" title="Efficient Tool Use" goal="区分必要探索与浪费调用，用行为指标约束效率叙事。" href="/learn/08-efficient-tools" level="进阶" prerequisite="GRPO · evaluation" />
    </div>
  </section>

  <section class="mini-section">
    <div class="mini-section__head">
      <span class="section-kicker">00 → 08</span>
      <h2>一条连续的学习路线</h2>
      <p>CPU 概念课、真实 Qwen 推理和 GPU GRPO smoke 各自标注，先理解边界，再增加系统复杂度。</p>
    </div>
    <CourseMap />
  </section>

  <section class="mini-section">
    <div class="mini-section__head">
      <span class="section-kicker">BEHAVIOR IS DATA</span>
      <h2>不要只看一个 Reward</h2>
      <p>比较成功多跳、过早作答、重复搜索、无效 action 与浪费的第三次调用。</p>
      <p><a class="mini-button mini-button--primary" href="./playground/trajectories">打开 Trajectory Explorer</a></p>
    </div>
  </section>
</div>
