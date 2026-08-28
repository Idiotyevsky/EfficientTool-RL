<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import GRPOGroupDemo from './GRPOGroupDemo.vue'
import TrajectoryExplorer from './TrajectoryExplorer.vue'

type Language = 'zh' | 'en'
type Panel = 'trajectory' | 'grpo'

const props = withDefaults(defineProps<{
  language?: Language
  kicker?: string
  title?: string
  lead?: string
}>(), {
  language: 'zh',
  kicker: 'SEE IT IN ACTION',
  title: 'See how the agent acts — and how it learns.',
  lead: 'Switch between behavior trajectories and a GRPO group to inspect tool use, rewards, and relative advantage.',
})

const active = ref<Panel>('trajectory')

const ui = computed(() => props.language === 'en'
  ? {
      behavior: 'Agent Behavior',
      learning: 'GRPO Learning',
      behaviorLead: 'Inspect a complete trajectory: what the model attempted, what the environment executed, and which searches added evidence.',
      learningLead: 'Change the rewards and watch the group-relative signal move.',
      behaviorLink: 'Explore all trajectories',
      learningLink: 'Learn the GRPO update',
    }
  : {
      behavior: 'Agent Behavior',
      learning: 'GRPO Learning',
      behaviorLead: '观察一条完整 trajectory：模型尝试了什么、环境执行了什么，以及哪一次搜索真正带来了证据。',
      learningLead: '拖动 Reward，观察组内相对信号如何变化。',
      behaviorLink: '查看全部轨迹',
      learningLink: '学习 GRPO 更新',
    })
</script>

<template>
  <section class="showcase-tabs" aria-labelledby="showcase-title">
    <header class="showcase-tabs__header">
      <div>
        <span class="section-kicker">{{ props.kicker }}</span>
        <h2 id="showcase-title">{{ props.title }}</h2>
        <p>{{ props.lead }}</p>
      </div>
      <div class="showcase-tabs__switch" role="tablist" :aria-label="language === 'en' ? 'Choose a showcase' : '选择展示内容'">
        <button
          id="showcase-tab-trajectory"
          type="button"
          role="tab"
          :aria-selected="active === 'trajectory'"
          aria-controls="showcase-panel-trajectory"
          @click="active = 'trajectory'"
        >
          {{ ui.behavior }}
        </button>
        <button
          id="showcase-tab-grpo"
          type="button"
          role="tab"
          :aria-selected="active === 'grpo'"
          aria-controls="showcase-panel-grpo"
          @click="active = 'grpo'"
        >
          {{ ui.learning }}
        </button>
      </div>
    </header>

    <div
      id="showcase-panel-trajectory"
      class="showcase-tabs__panel"
      role="tabpanel"
      aria-labelledby="showcase-tab-trajectory"
      :aria-hidden="active !== 'trajectory'"
      v-show="active === 'trajectory'"
    >
      <p class="showcase-tabs__description">{{ ui.behaviorLead }}</p>
      <TrajectoryExplorer :language="language" />
      <a class="mini-text-link showcase-tabs__link" :href="withBase('/playground/trajectories')">{{ ui.behaviorLink }} →</a>
    </div>

    <div
      id="showcase-panel-grpo"
      class="showcase-tabs__panel"
      role="tabpanel"
      aria-labelledby="showcase-tab-grpo"
      :aria-hidden="active !== 'grpo'"
      v-show="active === 'grpo'"
    >
      <p class="showcase-tabs__description">{{ ui.learningLead }}</p>
      <GRPOGroupDemo :language="language" />
      <a class="mini-text-link showcase-tabs__link" :href="withBase('/learn/06-grpo')">{{ ui.learningLink }} →</a>
    </div>
  </section>
</template>
