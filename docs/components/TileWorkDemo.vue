<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const n = ref(10)
const block = ref(4)
const completed = ref(0)
const count = computed(() => Math.ceil(n.value / block.value))
const groups = computed(() => Array.from({ length: count.value }, (_, bx) => ({
  bx,
  indices: Array.from({ length: block.value }, (_, i) => bx * block.value + i),
})))
const skipped = computed(() => count.value * block.value - n.value)
const status = computed(() => completed.value === count.value
  ? `已完成全部 ${count.value} 块，得到 ${n.value} 个输出；跳过 ${skipped.value} 个越界位置。`
  : `已演示 ${completed.value} / ${count.value} 块；候选位置 ${count.value * block.value}，有效元素 ${n.value}。`)
watch([n, block], () => { completed.value = 0 })
</script>

<template>
  <section class="tile-work-demo" aria-label="Tile 分块与尾部保护演示">
    <p><strong>把每个元素加上 10：先看分工，再看结果。</strong></p>
    <div class="tile-work-controls">
      <label>数组长度
        <select v-model.number="n" aria-label="数组长度">
          <option :value="10">10</option>
          <option :value="16">16</option>
          <option :value="33">33</option>
        </select>
      </label>
      <label>每块元素数
        <select v-model.number="block" aria-label="每块元素数">
          <option :value="4">4</option>
          <option :value="8">8</option>
          <option :value="16">16</option>
        </select>
      </label>
      <button type="button" :disabled="completed === count" @click="completed++">演示下一块</button>
      <button type="button" @click="completed = 0">重置</button>
    </div>
    <div class="tile-work-groups">
      <div v-for="group in groups" :key="group.bx" class="tile-work-group"
        :class="{ active: group.bx === completed }">
        <span>线程组 {{ group.bx }}</span>
        <div class="tile-work-cells">
          <span v-for="idx in group.indices" :key="idx" class="tile-work-cell"
            :class="{ invalid: idx >= n, done: idx < n && group.bx < completed }"
            :aria-label="idx >= n ? `索引 ${idx} 越界，跳过` : group.bx < completed ? `C[${idx}] = ${idx + 10}` : `待计算索引 ${idx}`">
            {{ idx >= n ? '跳过' : group.bx < completed ? idx + 10 : `[${idx}]` }}
          </span>
        </div>
      </div>
    </div>
    <p class="tile-work-status" role="status" aria-live="polite">{{ status }}</p>
    <p class="tile-work-note">方括号表示输入索引，完成后显示输出；虚线格不访问内存。此图只演示逻辑分工，不在浏览器中运行 GPU；实际线程组不保证按按钮演示的顺序执行。</p>
  </section>
</template>

<style scoped>
.tile-work-demo { margin: 24px 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 8px; }
.tile-work-controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; }
.tile-work-controls label { display: grid; gap: 4px; font-size: 14px; }
.tile-work-controls select, .tile-work-controls button { min-height: 44px; padding: 6px 12px; border: 1px solid var(--vp-c-divider); border-radius: 6px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); font: inherit; }
.tile-work-controls button { cursor: pointer; }
.tile-work-controls button:disabled { cursor: default; opacity: 0.5; }
.tile-work-controls :focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 3px; }
.tile-work-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(156px, 100%), 1fr)); gap: 12px; margin-top: 16px; }
.tile-work-group { min-width: 0; border: 1px solid var(--vp-c-divider); padding: 8px; border-radius: 4px; font-size: 13px; }
.tile-work-group.active { border-color: var(--vp-c-brand-1); }
.tile-work-cells { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; margin-top: 8px; }
.tile-work-cell { padding: 4px 0; text-align: center; background: var(--vp-c-bg-soft); border-bottom: 2px solid transparent; }
.tile-work-cell.done { border-bottom-color: var(--vp-c-brand-1); }
.tile-work-cell.invalid { border: 1px dashed var(--vp-c-divider); color: var(--vp-c-text-2); background: none; }
.tile-work-status { font-size: 14px; }
.tile-work-note { font-size: 13px; color: var(--vp-c-text-2); }
</style>
