<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

type Metrics = { attempted: number; valid: number; executed: number; useful: number; wasted: number }
type Turn = { role: string; title: string; content: string; useful?: boolean }
type Trajectory = {
  id: string
  label: string
  kind: string
  description: string
  question: string
  finalAnswer: string | null
  terminationReason: string
  reward: number
  metrics: Metrics
  turns: Turn[]
}

const trajectories = ref<Trajectory[]>([])
const selected = ref(0)
const provenance = ref('')
const error = ref('')

onMounted(async () => {
  try {
    const response = await fetch(withBase('/data/teaching-trajectories.json'))
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    trajectories.value = payload.trajectories
    provenance.value = payload.provenance
  } catch (reason) {
    error.value = `轨迹数据加载失败：${reason instanceof Error ? reason.message : String(reason)}`
  }
})
</script>

<template>
  <section class="trajectory-explorer" aria-labelledby="trajectory-title">
    <header class="trajectory-explorer__header">
      <div>
        <span class="section-kicker">BEHAVIOR, NOT JUST SCORES</span>
        <h2 id="trajectory-title">Trajectory Explorer</h2>
        <p>切换行为模式，观察一次 Agent episode 在哪里成功、失效或浪费工具。</p>
      </div>
    </header>

    <p v-if="error" class="trajectory-error" role="alert">{{ error }}</p>
    <template v-else-if="trajectories.length">
      <div class="trajectory-tabs" role="tablist" aria-label="选择轨迹类型">
        <button
          v-for="(trajectory, index) in trajectories"
          :id="`trajectory-tab-${index}`"
          :key="trajectory.id"
          type="button"
          role="tab"
          :aria-selected="selected === index"
          :aria-controls="`trajectory-panel-${index}`"
          :tabindex="selected === index ? 0 : -1"
          @click="selected = index"
        >
          {{ trajectory.label }}
        </button>
      </div>

      <div
        :id="`trajectory-panel-${selected}`"
        class="trajectory-layout"
        role="tabpanel"
        :aria-labelledby="`trajectory-tab-${selected}`"
      >
        <main class="trajectory-main">
          <div class="trajectory-question">
            <span>QUESTION</span>
            <strong>{{ trajectories[selected].question }}</strong>
          </div>
          <ol class="trajectory-timeline">
            <li v-for="(turn, index) in trajectories[selected].turns" :key="`${turn.title}-${index}`" :class="`is-${turn.role}`">
              <span class="trajectory-dot" aria-hidden="true" />
              <article>
                <div class="trajectory-turn-title">
                  <strong>{{ turn.title }}</strong>
                  <span v-if="turn.useful === true" class="evidence-tag is-useful">新增证据</span>
                  <span v-else-if="turn.useful === false" class="evidence-tag">无新增证据</span>
                </div>
                <code>{{ turn.content }}</code>
              </article>
            </li>
          </ol>
        </main>

        <aside class="trajectory-summary" aria-label="轨迹摘要">
          <span class="teaching-label">{{ trajectories[selected].kind }}</span>
          <h3>{{ trajectories[selected].label }}</h3>
          <p>{{ trajectories[selected].description }}</p>
          <div class="trajectory-metrics">
            <MetricPill label="attempted" :value="trajectories[selected].metrics.attempted" />
            <MetricPill label="valid" :value="trajectories[selected].metrics.valid" />
            <MetricPill label="executed" :value="trajectories[selected].metrics.executed" tone="tool" />
            <MetricPill label="useful" :value="trajectories[selected].metrics.useful" tone="positive" />
            <MetricPill label="wasted" :value="trajectories[selected].metrics.wasted" tone="negative" />
            <MetricPill label="reward" :value="trajectories[selected].reward.toFixed(2)" tone="agent" />
          </div>
          <dl class="trajectory-result">
            <div><dt>Final answer</dt><dd>{{ trajectories[selected].finalAnswer ?? 'None' }}</dd></div>
            <div><dt>Termination</dt><dd>{{ trajectories[selected].terminationReason }}</dd></div>
          </dl>
        </aside>
      </div>
      <p class="trajectory-provenance"><strong>Provenance:</strong> {{ provenance }}</p>
    </template>
    <p v-else class="trajectory-loading" aria-live="polite">正在加载教学轨迹…</p>
  </section>
</template>
