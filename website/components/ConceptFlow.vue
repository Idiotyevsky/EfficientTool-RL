<script setup lang="ts">
defineProps<{
  items: Array<{ label: string; detail?: string; tone?: string }>
  ariaLabel?: string
}>()
</script>

<template>
  <div class="concept-flow" role="list" :aria-label="ariaLabel || '概念流程'">
    <template v-for="(item, index) in items" :key="item.label">
      <div class="concept-flow__item" :class="item.tone ? `is-${item.tone}` : ''" role="listitem">
        <strong>{{ item.label }}</strong>
        <span v-if="item.detail">{{ item.detail }}</span>
      </div>
      <span v-if="index < items.length - 1" class="concept-flow__arrow" aria-hidden="true">→</span>
    </template>
  </div>
</template>
