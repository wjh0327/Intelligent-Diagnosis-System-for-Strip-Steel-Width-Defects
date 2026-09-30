<template>
  <div class="chat-view">
    <el-card shadow="never" class="chat-card">
      <template #header>
        <span class="card-title">知识问答 / Agent 诊断</span>
        <div class="header-right">
          <el-tag v-if="chatStore.messages.length > 0" size="small" type="info">
            多轮对话（{{ chatStore.messages.length / 2 }} 轮）
          </el-tag>
          <el-button size="small" @click="chatStore.clear">清除对话</el-button>
        </div>
      </template>

      <div class="chat-messages" ref="messagesRef">
        <div v-if="chatStore.messages.length === 0 && !chatStore.loading" class="empty-hint">
          请输入问题，AI 专家将自主调用工具进行分析（支持多轮对话）
        </div>

        <div
          v-for="(msg, i) in chatStore.messages"
          :key="i"
          :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-ai']"
        >
          <div class="msg-role">
            <span>{{ msg.role === 'user' ? '👤 你' : '🤖 专家' }}</span>
            <span class="msg-time" v-if="msg.role === 'user'">刚刚</span>
          </div>
          <div class="msg-body" v-html="formatContent(msg.content)"></div>
        </div>

        <!-- 等待中：显示计时器 + 取消按钮 -->
        <div v-if="chatStore.loading" class="msg msg-ai loading-msg">
          <div class="msg-role">🤖 专家</div>
          <div class="msg-body loading-body">
            <div class="loading-row">
              <el-icon class="spinner"><Loading /></el-icon>
              <span>正在分析中，请稍候...</span>
            </div>
            <div class="elapsed">
              已等待 <strong>{{ formatElapsed(chatStore.elapsedSeconds) }}</strong>
              <span class="elapsed-hint" v-if="chatStore.elapsedSeconds > 120">
                （批量诊断含重排序，复杂问题可能需要 10 分钟，可点下方"取消"）
              </span>
            </div>
            <el-button
              type="warning"
              size="small"
              plain
              style="margin-top:4px"
              @click="chatStore.cancel"
            >
              取消
            </el-button>
          </div>
        </div>
      </div>

      <div class="chat-attach">
        <input
          ref="fileInputRef"
          type="file"
          accept=".csv"
          style="display:none"
          @change="onFilePick"
        />
        <el-button
          size="small"
          :loading="chatStore.csvUploading"
          :disabled="chatStore.loading"
          @click="() => fileInputRef?.click()"
        >
          <el-icon><Paperclip /></el-icon> 选择 CSV（整批诊断）
        </el-button>

        <template v-if="chatStore.csvPath">
          <el-tag type="success" size="small" closable @close="chatStore.detachCSV">
            <el-icon><Document /></el-icon>
            {{ chatStore.csvName }}（{{ chatStore.csvRows }} 行）
          </el-tag>
          <span v-if="chatStore.csvMissing.length" class="attach-warn">
            缺少特征列：{{ chatStore.csvMissing.join('、') }}
          </span>
        </template>
      </div>

      <div class="chat-input">
        <el-input
          v-model="input"
          type="textarea"
          :rows="2"
          placeholder="输入问题，例如：'带钢头部拉窄是什么原因？' 或粘贴 12 个特征值"
          @keydown.ctrl.enter="send"
          :disabled="chatStore.loading"
        />
        <div class="input-actions">
          <el-button type="primary" :loading="chatStore.loading" @click="send">
            {{ chatStore.loading ? '分析中...' : '发送' }}
          </el-button>
          <el-button
            v-if="chatStore.messages.length > 0 && !chatStore.loading"
            @click="chatStore.clear"
            size="small"
          >
            清空
          </el-button>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import { Loading, Paperclip, Document } from '@element-plus/icons-vue'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const input = ref('')
const messagesRef = ref<HTMLElement | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)

async function onFilePick(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  await chatStore.attachCSV(file)
  target.value = ''  // 允许重复选择同一文件
}

// 每次新增消息或 loading 状态变化时滚动到底部
watch(
  () => [chatStore.messages.length, chatStore.loading, chatStore.elapsedSeconds],
  () => {
    nextTick(() => {
      if (messagesRef.value) {
        messagesRef.value.scrollTop = messagesRef.value.scrollHeight
      }
    })
  }
)

function send() {
  const q = input.value.trim()
  if (!q || chatStore.loading) return
  chatStore.sendMessage(q)
  input.value = ''
}

function formatContent(raw: string): string {
  if (raw == null) return ''
  let s = String(raw)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
  s = s.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>')
  s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/\n/g, '<br/>')
  return s
}

function formatElapsed(sec: number): string {
  if (sec < 60) return `${sec} 秒`
  return `${Math.floor(sec / 60)} 分 ${sec % 60} 秒`
}
</script>

<style scoped>
.chat-view { display: flex; flex-direction: column; gap: 0; }
.chat-card {
  width: 100%;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 500px;
}
.chat-card :deep(.el-card__header) {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.header-right { display: flex; align-items: center; gap: 8px; }
.card-title { font-size: 15px; font-weight: 600; }

.chat-messages {
  flex: 1;
  min-height: 420px;
  max-height: 600px;
  overflow-y: auto;
  padding: 16px;
  background: #f7f8fa;
  border-radius: 8px;
  margin-bottom: 12px;
}
.empty-hint {
  color: #909399;
  text-align: center;
  padding: 60px 0;
  font-size: 14px;
}

.msg {
  display: flex;
  flex-direction: column;
  margin-bottom: 16px;
  max-width: 88%;
}
.msg-user { align-self: flex-end; align-items: flex-end; }
.msg-ai   { align-self: flex-start; align-items: flex-start; }
.msg-role {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.msg-time { font-size: 11px; color: #c0c4cc; }
.msg-body {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
}
.msg-user .msg-body {
  background: #1e3a5f;
  color: #fff;
}
.msg-ai .msg-body {
  background: #fff;
  border: 1px solid #e4e7ed;
}
.msg-body pre {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 8px 0;
  font-size: 12px;
}
.msg-body code {
  background: #f0f0f0;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 13px;
}
.msg-body pre code { background: none; padding: 0; }

/* 等待中 */
.loading-msg .loading-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px 20px;
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 12px;
}
.loading-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #606266;
}
.spinner {
  font-size: 18px;
  color: #409eff;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.elapsed {
  font-size: 12px;
  color: #909399;
}
.elapsed strong { color: #409eff; font-size: 14px; }
.elapsed-hint { color: #e6a23c; }

.chat-input { display: flex; gap: 10px; align-items: flex-end; }
.chat-input .el-input { flex: 1; }
.chat-input .el-input :deep(.el-textarea__inner) { border-radius: 8px; }
.input-actions { display: flex; flex-direction: column; gap: 6px; }

/* CSV 附件栏 */
.chat-attach {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  min-height: 24px;
}
.attach-warn {
  font-size: 12px;
  color: #e6a23c;
}
.chat-attach .el-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
</style>
