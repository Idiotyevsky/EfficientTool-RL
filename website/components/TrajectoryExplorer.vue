<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

type Language = 'zh' | 'en'
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

const props = withDefaults(defineProps<{ language?: Language }>(), { language: 'zh' })
const trajectories = ref<Trajectory[]>([])
const selected = ref(0)
const provenance = ref('')
const errorMessage = ref('')

const ui = computed(() => props.language === 'en'
  ? {
      title: 'Trajectory Explorer',
      subtitle: 'Switch behavior patterns to see where an agent succeeds, fails, or wastes a tool call.',
      tabAria: 'Choose a trajectory type',
      question: 'QUESTION',
      summary: 'Trajectory summary',
      newEvidence: 'new evidence',
      noNewEvidence: 'no new evidence',
      finalAnswer: 'Final answer',
      termination: 'Termination',
      none: 'None',
      loading: 'Loading teaching trajectories…',
      error: 'Could not load trajectory data: ',
      provenance: 'Provenance',
    }
  : {
      title: 'Trajectory Explorer',
      subtitle: '切换行为模式，观察一次 Agent episode 在哪里成功、失效或浪费工具。',
      tabAria: '选择轨迹类型',
      question: 'QUESTION',
      summary: '轨迹摘要',
      newEvidence: '新增证据',
      noNewEvidence: '无新增证据',
      finalAnswer: 'Final answer',
      termination: 'Termination',
      none: '无',
      loading: '正在加载教学轨迹…',
      error: '轨迹数据加载失败：',
      provenance: '数据来源',
    })

const zhLabels: Record<string, string> = {
  'good-two-hop': '成功多跳',
  'under-search': '过早作答',
  'repeated-query': '重复查询',
  'invalid-tool-call': '无效 Tool Call',
  'wasted-third-search': '浪费的第三次调用',
}

const zhDescriptions: Record<string, string> = {
  'good-two-hop': '两次搜索分别补齐人物与机器证据，随后正确作答。',
  'under-search': '没有执行搜索便猜测答案；成本低，但任务失败。',
  'repeated-query': '第二次执行了合法搜索，却没有带来新 supporting title。',
  'invalid-tool-call': '模型尝试调用工具，但 JSON 未闭合，parser 将其记录为 InvalidAction。',
  'wasted-third-search': '前两次搜索已经补齐证据，第三次搜索没有信息增益。',
}

const enDescriptions: Record<string, string> = {
  'good-two-hop': 'Two searches gather the person and machine evidence before the agent answers correctly.',
  'under-search': 'The agent guesses without searching; the cost is low, but the task fails.',
  'repeated-query': 'The second valid search executes but adds no new supporting title.',
  'invalid-tool-call': 'The model attempts a tool call with unclosed JSON, which the parser records as InvalidAction.',
  'wasted-third-search': 'The first two searches complete the evidence; the third search adds nothing new.',
}

const turnTitlesZh: Record<string, string> = {
  'Turn 1 · Tool Call': '第 1 轮 · Tool Call',
  'Observation 1': 'Observation 1',
  'Turn 2 · Tool Call': '第 2 轮 · Tool Call',
  'Observation 2': 'Observation 2',
  'Turn 3 · Final Answer': '第 3 轮 · Final Answer',
  'Turn 1 · Premature Answer': '第 1 轮 · 过早作答',
  'Turn 2 · Repeated Tool Call': '第 2 轮 · 重复 Tool Call',
  'Turn 1 · Malformed Action': '第 1 轮 · 格式错误 Action',
  'Recovery Observation': 'Recovery Observation',
}

function displayLabel(trajectory: Trajectory) {
  return props.language === 'en' ? trajectory.label : (zhLabels[trajectory.id] || trajectory.label)
}

function displayDescription(trajectory: Trajectory) {
  return props.language === 'en' ? (enDescriptions[trajectory.id] || trajectory.description) : (zhDescriptions[trajectory.id] || trajectory.description)
}

function displayTurnTitle(title: string) {
  return props.language === 'en' ? title : (turnTitlesZh[title] || title)
}

function displayKind() {
  return props.language === 'en' ? 'Teaching example' : '教学示例'
}

const provenanceText = computed(() => {
  if (props.language === 'en') return provenance.value
  return '这些是根据 tiny corpus、示例脚本和项目 AgentRunner 整理的教学示例，不代表模型预测或 benchmark 结果。'
})

onMounted(async () => {
  try {
    const response = await fetch(withBase('/data/teaching-trajectories.json'))
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    trajectories.value = payload.trajectories
    provenance.value = payload.provenance
  } catch (reason) {
    errorMessage.value = reason instanceof Error ? reason.message : String(reason)
  }
})
</script>

<template>
  <section class="trajectory-explorer" aria-labelledby="trajectory-title">
    <header class="trajectory-explorer__header">
      <div>
        <span class="section-kicker">BEHAVIOR, NOT JUST SCORES</span>
        <h2 id="trajectory-title">{{ ui.title }}</h2>
        <p>{{ ui.subtitle }}</p>
      </div>
    </header>

    <p v-if="errorMessage" class="trajectory-error" role="alert">{{ ui.error }}{{ errorMessage }}</p>
    <template v-else-if="trajectories.length">
      <div class="trajectory-tabs" role="tablist" :aria-label="ui.tabAria">
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
          {{ displayLabel(trajectory) }}
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
            <span>{{ ui.question }}</span>
            <strong>{{ trajectories[selected].question }}</strong>
          </div>
          <ol class="trajectory-timeline">
            <li v-for="(turn, index) in trajectories[selected].turns" :key="`${turn.title}-${index}`" :class="`is-${turn.role}`">
              <span class="trajectory-dot" aria-hidden="true" />
              <article>
                <div class="trajectory-turn-title">
                  <strong>{{ displayTurnTitle(turn.title) }}</strong>
                  <span v-if="turn.useful === true" class="evidence-tag is-useful">{{ ui.newEvidence }}</span>
                  <span v-else-if="turn.useful === false" class="evidence-tag">{{ ui.noNewEvidence }}</span>
                </div>
                <code>{{ turn.content }}</code>
              </article>
            </li>
          </ol>
        </main>

        <aside class="trajectory-summary" :aria-label="ui.summary">
          <span class="teaching-label">{{ displayKind() }}</span>
          <h3>{{ displayLabel(trajectories[selected]) }}</h3>
          <p>{{ displayDescription(trajectories[selected]) }}</p>
          <div class="trajectory-metrics">
            <MetricPill label="attempted" :value="trajectories[selected].metrics.attempted" />
            <MetricPill label="valid" :value="trajectories[selected].metrics.valid" />
            <MetricPill label="executed" :value="trajectories[selected].metrics.executed" tone="tool" />
            <MetricPill label="useful" :value="trajectories[selected].metrics.useful" tone="positive" />
            <MetricPill label="wasted" :value="trajectories[selected].metrics.wasted" tone="negative" />
            <MetricPill label="reward" :value="trajectories[selected].reward.toFixed(2)" tone="agent" />
          </div>
          <dl class="trajectory-result">
            <div><dt>{{ ui.finalAnswer }}</dt><dd>{{ trajectories[selected].finalAnswer ?? ui.none }}</dd></div>
            <div><dt>{{ ui.termination }}</dt><dd>{{ trajectories[selected].terminationReason }}</dd></div>
          </dl>
        </aside>
      </div>
      <p class="trajectory-provenance"><strong>{{ ui.provenance }}:</strong> {{ provenanceText }}</p>
    </template>
    <p v-else class="trajectory-loading" aria-live="polite">{{ ui.loading }}</p>
  </section>
</template>
