<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

type Language = 'zh' | 'en'

const props = withDefaults(defineProps<{ language?: Language }>(), { language: 'zh' })

const lessons = computed(() => props.language === 'en'
  ? [
      { id: '00', title: 'Start', meta: 'CPU · runnable', href: '/learn/00-start' },
      { id: '01', title: 'Tool Calling', meta: 'CPU · runnable', href: '/learn/01-tool-calling' },
      { id: '02', title: 'Real Qwen', meta: 'GPU · runnable', href: '/learn/02-real-qwen' },
      { id: '03', title: 'Multi-turn', meta: 'CPU · runnable', href: '/learn/03-multiturn' },
      { id: '04', title: 'ReAct', meta: 'GPU · runnable', href: '/learn/04-react-hotpot' },
      { id: '05', title: 'Rollout & Reward', meta: 'CPU · runnable', href: '/learn/05-rollout-reward' },
      { id: '06', title: 'GRPO', meta: 'CPU · interactive concept', href: '/learn/06-grpo' },
      { id: '07', title: 'Real Update', meta: 'GPU · actual training', href: '/learn/07-grpo-smoke' },
      { id: '08', title: 'Efficient Tools', meta: 'CPU · runnable', href: '/learn/08-efficient-tools' },
    ]
  : [
      { id: '00', title: 'Start', meta: 'CPU · 可执行', href: '/learn/00-start' },
      { id: '01', title: 'Tool Calling', meta: 'CPU · 可执行', href: '/learn/01-tool-calling' },
      { id: '02', title: 'Real Qwen', meta: 'GPU · 可执行', href: '/learn/02-real-qwen' },
      { id: '03', title: 'Multi-turn', meta: 'CPU · 可执行', href: '/learn/03-multiturn' },
      { id: '04', title: 'ReAct', meta: 'GPU · 可执行', href: '/learn/04-react-hotpot' },
      { id: '05', title: 'Rollout & Reward', meta: 'CPU · 可执行', href: '/learn/05-rollout-reward' },
      { id: '06', title: 'GRPO', meta: 'CPU · 交互概念', href: '/learn/06-grpo' },
      { id: '07', title: 'Real Update', meta: 'GPU · 真训练', href: '/learn/07-grpo-smoke' },
      { id: '08', title: 'Efficient Tools', meta: 'CPU · 可执行', href: '/learn/08-efficient-tools' },
    ])

const ariaLabel = computed(() => props.language === 'en' ? 'MiniAgentRL learning path' : 'MiniAgentRL 课程路线')
</script>

<template>
  <nav class="course-map" :aria-label="ariaLabel">
    <a v-for="(lesson, index) in lessons" :key="lesson.id" class="course-node" :href="withBase(lesson.href)">
      <span class="course-node__number">{{ lesson.id }}</span>
      <span class="course-node__content">
        <strong>{{ lesson.title }}</strong>
        <small>{{ lesson.meta }}</small>
      </span>
      <span v-if="index < lessons.length - 1" class="course-node__connector" aria-hidden="true" />
    </a>
  </nav>
</template>
