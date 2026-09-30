<template>
  <div class="records-view">
    <el-card shadow="never">
      <template #header>
        <span class="card-title">历史诊断记录</span>
        <el-tag v-if="!mysqlEnabled" type="danger">MySQL 未启用</el-tag>
        <el-tag v-else type="success">MySQL 已连接</el-tag>
      </template>

      <el-alert v-if="!mysqlEnabled && mysqlHint" :title="mysqlHint" type="warning" :closable="false" style="margin-bottom:16px" />

      <div class="filter-bar" v-if="mysqlEnabled">
        <el-input v-model="faultFilter" placeholder="故障类型关键词" clearable style="width:180px" @keyup.enter="loadRecords" />
        <el-input-number v-model="daysFilter" :min="1" :max="365" style="width:120px" />
        <span style="font-size:12px;color:#909399">天内</span>
        <el-button type="primary" @click="loadRecords">查询</el-button>
        <el-button @click="resetFilter">重置</el-button>
        <span class="total-hint">共 {{ total }} 条</span>
      </div>

      <el-table :data="records" stripe size="medium" v-loading="loading" style="margin-top:12px">
        <el-table-column prop="updated_at" label="时间" width="170" />
        <el-table-column prop="fault_cn" label="故障类型" width="160">
          <template #default="{ row }">
            <el-tag size="small">{{ row.fault_cn }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="90" :formatter="formatConfidence" />
        <el-table-column prop="fault_desc" label="诊断描述" min-width="280" show-overflow-tooltip />
        <el-table-column label="特征哈希" width="120" show-overflow-tooltip>
          <template #default="{ row }">{{ (row.feat_hash ?? '').slice(0, 12) }}...</template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-if="total > 0"
        v-model:current-page="page"
        :page-size="pageSize"
        :total="total"
        layout="prev, pager, next, total"
        style="margin-top:16px; justify-content: flex-end;"
        @current-change="loadRecords"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getDiagRecords } from '@/api'
import type { DiagRecord } from '@/types'

const records = ref<DiagRecord[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const faultFilter = ref('')
const daysFilter = ref(30)
const mysqlEnabled = ref(false)
const mysqlHint = ref('')

async function loadRecords() {
  loading.value = true
  try {
    const res = await getDiagRecords(faultFilter.value || undefined, daysFilter.value, page.value, pageSize.value)
    mysqlEnabled.value = res.enabled
    mysqlHint.value = res.hint ?? ''
    if (res.enabled) {
      records.value = res.items
      total.value = res.total
    }
  } finally {
    loading.value = false
  }
}

function formatConfidence(_row: {confidence:number|null}, _col: string, row: {confidence:number|null}) {
  return (row.confidence ?? 0).toFixed(4)
}

function resetFilter() {
  faultFilter.value = ''
  daysFilter.value = 30
  page.value = 1
  loadRecords()
}

onMounted(() => { loadRecords() })
</script>

<style scoped>
.records-view { display: flex; }
.filter-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.total-hint { font-size: 12px; color: #909399; margin-left: 8px; }
.card-title { font-size: 16px; font-weight: 600; }
</style>
