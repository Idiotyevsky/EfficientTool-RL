<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{
  eyebrow: string
  title: string
  goal: string
  href: string
  level: string
  prerequisite: string
}>()
</script>

<template>
  <a class="lesson-card" :href="withBase(href)">
    <span class="lesson-card__eyebrow">{{ eyebrow }}</span>
    <h3>{{ title }}</h3>
    <p>{{ goal }}</p>
    <dl>
      <div><dt>难度</dt><dd>{{ level }}</dd></div>
      <div><dt>前置</dt><dd>{{ prerequisite }}</dd></div>
    </dl>
    <span class="lesson-card__link">进入模块 <span aria-hidden="true">→</span></span>
  </a>
</template>
