<script setup lang="ts">
withDefaults(defineProps<{ title?: string }>(), { title: 'Checkpoint' })
</script>

<template>
  <section class="learning-checkpoint" aria-labelledby="checkpoint-title">
    <div class="learning-checkpoint__mark" aria-hidden="true">✓</div>
    <div>
      <h2 id="checkpoint-title">{{ title }}</h2>
      <slot />
    </div>
  </section>
</template>
