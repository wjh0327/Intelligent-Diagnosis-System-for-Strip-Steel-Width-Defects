import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { queryLLM, uploadCSV } from '@/api'
import type { ChatStep } from '@/types'

export interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMsg[]>([])
  const sessionId = ref<string>('')
  const loading = ref(false)
  const error = ref<string | null>(null)
  // 计时器：开始等待的时间戳 + 每秒更新的响应式 now（驱动 elapsedSeconds 重算）
  const startedAt = ref<number | null>(null)
  const now = ref(Date.now())
  let timer: ReturnType<typeof setInterval> | null = null
  // 与 Streamlit 原版保持一致：跨轮次传递批量诊断状态
  const batchDone = ref(false)
  const batchSummary = ref('')
  let abortCtrl: AbortController | null = null
  // CSV 附件（上传到服务器后可在后续提问中做整批诊断）
  const csvPath = ref('')
  const csvName = ref('')
  const csvRows = ref(0)
  const csvMissing = ref<string[]>([])
  const csvUploading = ref(false)

  function startTicker() {
    now.value = Date.now()
    if (timer) return
    timer = setInterval(() => { now.value = Date.now() }, 1000)
  }
  function stopTicker() {
    if (timer) { clearInterval(timer); timer = null }
  }

  /** 取消当前请求（长任务/卡死时可一键中止） */
  function cancel() {
    abortCtrl?.abort()
    abortCtrl = null
    loading.value = false
    startedAt.value = null
    stopTicker()
  }

  /** 上传 CSV 并作为后续提问的附件 */
  async function attachCSV(file: File) {
    csvUploading.value = true
    csvMissing.value = []
    try {
      const res = await uploadCSV(file)
      csvPath.value = res.csv_path
      csvName.value = res.file_name
      csvRows.value = res.rows
      csvMissing.value = res.missing_features || []
    } finally {
      csvUploading.value = false
    }
  }

  function detachCSV() {
    csvPath.value = ''
    csvName.value = ''
    csvRows.value = 0
    csvMissing.value = []
  }

  /** 当前已等待秒数（loading 期间每秒更新） */
  const elapsedSeconds = computed(() => {
    if (!loading.value || startedAt.value === null) return 0
    return Math.max(0, Math.floor((now.value - startedAt.value) / 1000))
  })

  async function sendMessage(question: string, features?: string) {
    error.value = null
    // ★ 关键：立即将用户提问加入列表，不必等 API 返回
    const userContent = csvPath.value
      ? `${question}\n\n[附件 CSV：${csvName.value}（${csvRows.value} 行）]`
      : question
    messages.value.push({ role: 'user', content: userContent })
    loading.value = true
    startedAt.value = Date.now()
    startTicker()

    abortCtrl = new AbortController()
    const isAborted = () => abortCtrl?.signal.aborted === true

    try {
      const MAX_HISTORY_ROUNDS = 4
      const recentHistory = messages.value.slice(0, -1)  // 去掉刚加入的本条
      const history = recentHistory.slice(-MAX_HISTORY_ROUNDS * 2)
        .map(m => ({ role: m.role, content: m.content }))

      const res = await queryLLM({
        question,
        features,
        session_id: sessionId.value || undefined,
        conversation_history: history,
        batch_done: batchDone.value || undefined,
        batch_summary: batchSummary.value || undefined,
        csv_path: csvPath.value || undefined,
      }, abortCtrl.signal)

      // 用户已取消 → 不写入结果
      if (isAborted()) return

      if (res.session_id && !sessionId.value) {
        sessionId.value = res.session_id
      }
      batchDone.value = res.batch_done ?? false
      batchSummary.value = res.batch_summary ?? ''

      // 拼接 Agent 推理步骤（与 Streamlit 原版一致）
      const steps = (res.steps ?? []) as ChatStep[]
      let answer = res.answer
      if (steps.length > 0) {
        const stepText = '### 🧩 推理过程\n' +
          steps.map((s, i) => {
            const obs = s.observation.length > 800
              ? s.observation.slice(0, 800) + '...'
              : s.observation
            return `\n**步骤 ${i + 1}：调用 \`${s.tool}\`**\n\n\`\`\`\n${obs}\n\`\`\``
          }).join('')
        answer = stepText + '\n### 📋 最终诊断结论\n' + answer
      }
      messages.value.push({ role: 'assistant', content: answer })
    } catch (e: any) {
      // 主动取消不算错误
      if (isAborted()) return
      const msg = e?.message || '请求失败'
      error.value = msg
      messages.value.push({
        role: 'assistant',
        content: `⚠️ 请求失败：${msg}`,
      })
    } finally {
      if (!isAborted()) {
        loading.value = false
        startedAt.value = null
        stopTicker()
      }
      abortCtrl = null
    }
  }

  function clear() {
    cancel()
    messages.value = []
    sessionId.value = ''
    batchDone.value = false
    batchSummary.value = ''
    error.value = null
    detachCSV()
  }

  return {
    messages, sessionId, loading, error,
    batchDone, batchSummary,
    startedAt, elapsedSeconds,
    csvPath, csvName, csvRows, csvMissing, csvUploading,
    attachCSV, detachCSV,
    sendMessage, clear, cancel,
  }
})
