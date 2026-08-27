<script setup lang="ts">
import { computed, ref } from 'vue'

type Language = 'zh' | 'en'

const props = withDefaults(defineProps<{ language?: Language }>(), { language: 'zh' })

const rewards = ref([1, 0.72, 0, 0])
const labels = ['A', 'B', 'C', 'D']
const mean = computed(() => rewards.value.reduce((sum, value) => sum + value, 0) / rewards.value.length)
const std = computed(() => {
  const variance = rewards.value.reduce((sum, value) => sum + (value - mean.value) ** 2, 0) / rewards.value.length
  return Math.sqrt(variance)
})
const advantages = computed(() => rewards.value.map(value => std.value ? (value - mean.value) / std.value : 0))
const ui = computed(() => props.language === 'en'
  ? {
      ariaLabel: 'Interactive GRPO rollout group',
      kicker: 'INTERACTIVE GROUP',
      title: 'One prompt, four rollouts, relative comparison',
      reward: 'Reward',
      positive: 'Increase the relative probability of this sampled action',
      negative: 'Decrease the relative probability of this sampled action',
      neutral: 'No relative direction within the group',
      warning: 'All four rewards are identical: this is a zero-variance group, so every normalized advantage is 0.',
      footnote: 'This is the group-relative signal; a real update still needs policy ratio, clipping, KL, backpropagation, and an optimizer step.',
    }
  : {
      ariaLabel: '交互式 GRPO rollout group',
      kicker: 'INTERACTIVE GROUP',
      title: '同一个 Prompt，四条 Rollout 互相比较',
      reward: 'Reward',
      positive: '提高该采样动作的相对概率',
      negative: '降低该采样动作的相对概率',
      neutral: '组内没有相对方向',
      warning: '四个 Reward 完全相同：这是 zero-variance group，归一化 Advantage 全为 0。',
      footnote: '这里计算的是组内相对信号；真正更新还需要 policy ratio、clipping、KL、反向传播和 optimizer step。',
    })
</script>

<template>
  <section class="grpo-demo" :aria-label="ui.ariaLabel">
    <header>
      <div>
        <span class="section-kicker">{{ ui.kicker }}</span>
        <h2 id="grpo-demo-title">{{ ui.title }}</h2>
      </div>
      <div class="grpo-demo__summary" aria-live="polite">
        <MetricPill label="mean" :value="mean.toFixed(3)" tone="agent" />
        <MetricPill label="std" :value="std.toFixed(3)" tone="tool" />
      </div>
    </header>

    <div class="grpo-demo__grid">
      <article v-for="(reward, index) in rewards" :key="labels[index]" class="rollout-card">
        <div class="rollout-card__title">
          <strong>Rollout {{ labels[index] }}</strong>
          <span :class="advantages[index] >= 0 ? 'adv-positive' : 'adv-negative'">
            A = {{ advantages[index] >= 0 ? '+' : '' }}{{ advantages[index].toFixed(2) }}
          </span>
        </div>
        <label :for="`reward-${index}`">{{ ui.reward }} {{ reward.toFixed(2) }}</label>
        <input
          :id="`reward-${index}`"
          v-model.number="rewards[index]"
          :aria-label="`${ui.reward} ${labels[index]}`"
          type="range"
          min="0"
          max="1"
          step="0.01"
        >
        <p>{{ advantages[index] > 0 ? ui.positive : advantages[index] < 0 ? ui.negative : ui.neutral }}</p>
      </article>
    </div>

    <div v-if="std === 0" class="grpo-demo__warning" role="status">
      {{ ui.warning }}
    </div>
    <p class="grpo-demo__footnote">{{ ui.footnote }}</p>
  </section>
</template>
