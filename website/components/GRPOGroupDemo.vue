<script setup lang="ts">
import { computed, ref } from 'vue'

const rewards = ref([1, 0.72, 0, 0])
const labels = ['A', 'B', 'C', 'D']
const mean = computed(() => rewards.value.reduce((sum, value) => sum + value, 0) / rewards.value.length)
const std = computed(() => {
  const variance = rewards.value.reduce((sum, value) => sum + (value - mean.value) ** 2, 0) / rewards.value.length
  return Math.sqrt(variance)
})
const advantages = computed(() => rewards.value.map(value => std.value ? (value - mean.value) / std.value : 0))
</script>

<template>
  <section class="grpo-demo" aria-labelledby="grpo-demo-title">
    <header>
      <div>
        <span class="section-kicker">INTERACTIVE GROUP</span>
        <h2 id="grpo-demo-title">同一个 Prompt，四条 Rollout 互相比较</h2>
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
        <label :for="`reward-${index}`">Reward {{ reward.toFixed(2) }}</label>
        <input
          :id="`reward-${index}`"
          v-model.number="rewards[index]"
          type="range"
          min="0"
          max="1"
          step="0.01"
        >
        <p>{{ advantages[index] > 0 ? '提高该采样动作的相对概率' : advantages[index] < 0 ? '降低该采样动作的相对概率' : '组内没有相对方向' }}</p>
      </article>
    </div>

    <div v-if="std === 0" class="grpo-demo__warning" role="status">
      四个 Reward 完全相同：这是 zero-variance group，归一化 Advantage 全为 0。
    </div>
    <p class="grpo-demo__footnote">这里计算的是组内相对信号；真正更新还需要 policy ratio、clipping、KL、反向传播和 optimizer step。</p>
  </section>
</template>
