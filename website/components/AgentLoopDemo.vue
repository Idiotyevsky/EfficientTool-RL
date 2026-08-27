<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

const nodes = [
  { key: 'question', label: 'Question', detail: '用户提出多跳问题', tone: 'neutral' },
  { key: 'agent', label: 'Agent', detail: 'Qwen 生成一个 action', tone: 'agent' },
  { key: 'tool', label: 'Tool Call', detail: '结构化 search 请求', tone: 'tool' },
  { key: 'search', label: 'Search', detail: 'BM25 执行检索', tone: 'tool' },
  { key: 'observation', label: 'Observation', detail: '证据回到上下文', tone: 'neutral' },
  { key: 'answer', label: 'Answer', detail: '输出最小答案 span', tone: 'agent' },
  { key: 'reward', label: 'Reward', detail: 'EM / F1 评分', tone: 'reward' },
  { key: 'grpo', label: 'GRPO', detail: '相对优势与参数更新', tone: 'agent' },
]

const active = ref(0)
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | undefined

const status = computed(() => `${active.value + 1} / ${nodes.length} · ${nodes[active.value].detail}`)

function next() {
  active.value = (active.value + 1) % nodes.length
}

function toggle() {
  playing.value = !playing.value
  if (playing.value) timer = setInterval(next, 1200)
  else if (timer) clearInterval(timer)
}

onBeforeUnmount(() => timer && clearInterval(timer))
</script>

<template>
  <section class="agent-loop" aria-label="Agentic RL 交互与训练流程">
    <header class="agent-loop__header">
      <div>
        <span class="section-kicker">ONE COMPLETE LOOP</span>
        <h2>Agent 不只是调用工具，它还会被训练。</h2>
      </div>
      <button class="quiet-button" type="button" :aria-pressed="playing" @click="toggle">
        {{ playing ? '暂停演示' : '播放流程' }}
      </button>
    </header>

    <div class="agent-loop__rail">
      <template v-for="(node, index) in nodes" :key="node.key">
        <button
          type="button"
          class="agent-loop__node"
          :class="[`is-${node.tone}`, { 'is-active': active === index }]"
          :aria-current="active === index ? 'step' : undefined"
          @click="active = index"
        >
          <span class="agent-loop__index">{{ String(index + 1).padStart(2, '0') }}</span>
          <strong>{{ node.label }}</strong>
          <small>{{ node.detail }}</small>
        </button>
        <span v-if="index < nodes.length - 1" class="agent-loop__edge" aria-hidden="true">→</span>
      </template>
    </div>
    <p class="agent-loop__status" aria-live="polite">{{ status }}</p>
    <p class="agent-loop__note">Search 与 Observation 可以循环多次；动画只是索引，理解流程不依赖动画。</p>
  </section>
</template>
