<script setup lang="ts">
withDefaults(defineProps<{
  label: string
  value: string | number
  tone?: 'neutral' | 'agent' | 'tool' | 'positive' | 'negative'
}>(), { tone: 'neutral' })
</script>

<template>
  <span class="metric-pill" :class="`metric-pill--${tone}`">
    <span class="metric-pill__label">{{ label }}</span>
    <strong>{{ value }}</strong>
  </span>
</template>
