import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getHealth, getMetaFeatures } from '@/api'
import type { HealthStatus, MetaFeatures } from '@/types'

export const useAppStore = defineStore('app', () => {
  const health = ref<HealthStatus | null>(null)
  const metaFeatures = ref<MetaFeatures | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function init() {
    loading.value = true
    error.value = null
    try {
      const [h, m] = await Promise.all([getHealth(), getMetaFeatures()])
      health.value = h
      metaFeatures.value = m
    } catch (e: any) {
      error.value = e?.message || '初始化失败'
    } finally {
      loading.value = false
    }
  }

  return { health, metaFeatures, loading, error, init }
})
