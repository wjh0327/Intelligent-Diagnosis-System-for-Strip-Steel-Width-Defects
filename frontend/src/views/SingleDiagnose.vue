<template>
  <div class="single-diagnose">
    <el-card shadow="never" class="upload-card">
      <template #header>
        <span class="card-title">单条 / 少量诊断</span>
        <span class="hint">上传 CSV 文件（每行一条带钢）</span>
      </template>
      <el-upload
        drag
        action=""
        :auto-upload="false"
        :on-change="handleFile"
        :on-remove="handleRemove"
        accept=".csv"
        :limit="1"
        :disabled="submitting"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">拖拽文件到此处，或 <em>点击上传</em></div>
        <template #tip>
          <div class="el-upload__tip">
            CSV 需包含列: STRIPNO(可选)、FMWTARGETHOT、RDWTARGETTOTAL、PDIWIDTHTOL、
            FMWIDTHACTHOT、RMWIDTHACTHOT、FMWTARGETCOL、RMDATASEQ、FMDATASEQ、DCDATASEQ
            （6个 is_* 标志位由服务按机理逻辑自动计算，无需提供）
          </div>
        </template>
      </el-upload>

      <div v-if="file" class="file-info">
        <el-icon><document /></el-icon>
        <span>{{ file.name }}</span>
        <span class="file-size">{{ formatSize(file.size) }}</span>
      </div>

      <el-button
        v-if="file && !submitting"
        type="primary"
        style="margin-top:16px; width: 100%"
        @click="runDiagnose"
      >
        开始诊断
      </el-button>
      <el-button
        v-if="submitting"
        type="primary"
        :loading="submitting"
        style="margin-top:16px; width: 100%"
        disabled
      >
        正在诊断中...
      </el-button>
    </el-card>

    <el-alert v-if="error" :title="error" type="error" show-icon :closable="false" style="margin:16px 0" />

    <el-card v-if="result" shadow="never" class="result-card">
      <template #header>
        <span class="card-title">诊断结果</span>
        <el-button size="small" @click="exportCSV">导出 CSV</el-button>
      </template>

      <div class="dist-section">
        <div class="section-label">缺陷类型分布</div>
        <el-tag
          v-for="[k, v] in Object.entries(result.distribution).sort((a,b) => (b[1] as number) - (a[1] as number))"
          :key="k"
          type="info"
          size="large"
          style="margin-right:8px;margin-bottom:8px"
        >
          {{ k }}：{{ v }} 条
        </el-tag>
        <el-tag v-if="result.needs_review_count" type="warning" size="large">
          需人工复核：{{ result.needs_review_count }} 条
        </el-tag>
      </div>

      <el-table :data="result.results" stripe size="small" max-height="420" style="margin-top:16px">
        <el-table-column prop="index" label="#" width="60" />
        <el-table-column prop="stripno" label="带钢编号" width="130" />
        <el-table-column prop="fault_cn" label="缺陷类型" width="130" />
        <el-table-column prop="confidence" label="置信度" width="90" :formatter="formatConfidence" />
        <el-table-column label="需复核" width="90">
          <template #default="{ row }">
            <el-tag :type="row.needs_review ? 'warning' : 'success'" size="small">
              {{ row.needs_review ? '是' : '否' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="evidence" label="归因证据" min-width="240" show-overflow-tooltip />
        <el-table-column prop="fault_desc" label="描述" min-width="160" show-overflow-tooltip />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Document } from '@element-plus/icons-vue'
import axios from 'axios'
import type { BatchResult } from '@/types'

const file = ref<File | null>(null)
const submitting = ref(false)
const result = ref<BatchResult | null>(null)
const error = ref<string | null>(null)

// 上传文件时获取原始 File 对象（Element Plus 的 on-change 给的是 UploadFile 包装）
function handleFile(uploadFile: any) {
  const rawFile = uploadFile?.raw ?? uploadFile?.file ?? null
  if (rawFile instanceof File) {
    file.value = rawFile
    error.value = null
    result.value = null
  } else {
    error.value = '无法识别上传的文件，请重新选择 CSV 文件'
    file.value = null
  }
}

function handleRemove() {
  file.value = null
  result.value = null
  error.value = null
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

async function runDiagnose() {
  if (!file.value) {
    ElMessage.warning('请先选择 CSV 文件')
    return
  }
  submitting.value = true
  result.value = null
  error.value = null
  try {
    const form = new FormData()
    form.append('file', file.value)
    const resp = await axios.post('/api/v1/diagnose/batch/file', form, {
      timeout: 120000,
    })
    result.value = resp.data as BatchResult
  } catch (e: any) {
    if (e.code === 'ERR_NETWORK' || e.code === 'ERR_CANCELED') {
      error.value = '网络连接失败，请检查后端服务是否运行'
    } else {
      const detail = e?.response?.data?.detail
      error.value = detail && typeof detail === 'string' ? detail : (e?.message ?? '诊断失败，请重试')
    }
  } finally {
    submitting.value = false
  }
}

function formatConfidence(_row: { confidence: number | null }, _col: string, row: { confidence: number | null }) {
  return (row.confidence ?? 0).toFixed(4)
}

function exportCSV() {
  if (!result.value) return
  const lines = ['index,stripno,fault_cn,confidence,needs_review,evidence,fault_desc']
  for (const r of result.value.results) {
    lines.push(`${r.index},"${r.stripno ?? ''}","${r.fault_cn}",${r.confidence ?? ''},${r.needs_review ? '是' : '否'},"${(r.evidence ?? '').replace(/"/g, '""')}","${(r.fault_desc ?? '').replace(/"/g, '""')}"`)
  }
  const blob = new Blob(['\uFEFF' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `诊断结果_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
}
</script>

<style scoped>
.single-diagnose { display: flex; flex-direction: column; gap: 16px; }
.upload-card :deep(.el-card__header) { display: flex; align-items: center; justify-content: space-between; }
.card-title { font-size: 16px; font-weight: 600; }
.hint { font-size: 12px; color: #909399; }
.file-info {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-radius: 6px;
  font-size: 13px;
  color: #606266;
}
.file-size { color: #909399; margin-left: auto; }
.result-card :deep(.el-card__header) { display: flex; align-items: center; justify-content: space-between; }
.dist-section { margin-bottom: 16px; }
.section-label { font-size: 13px; color: #909399; margin-bottom: 8px; }
</style>
