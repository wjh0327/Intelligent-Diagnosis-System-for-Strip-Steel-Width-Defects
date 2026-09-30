<template>
  <div class="kb-view">
    <el-card shadow="never">
      <template #header>
        <div class="header-row">
          <span class="card-title">知识库浏览</span>
          <el-radio-group v-model="tab" size="small">
            <el-radio-button value="docs">📄 技术文档</el-radio-button>
            <el-radio-button value="quads">🔧 四元组案例</el-radio-button>
          </el-radio-group>
        </div>
      </template>

      <!-- ========== 文档列表 ========== -->
      <template v-if="tab === 'docs'">
        <div class="search-bar">
          <el-input v-model="docKeyword" placeholder="搜索文档名..." clearable style="width:260px" @keyup.enter="loadDocs" />
          <el-button type="primary" @click="loadDocs">搜索</el-button>
          <el-button @click="loadDocs">刷新</el-button>
          <span class="total-hint">共 {{ docTotal }} 篇</span>
        </div>

        <el-table :data="docList" stripe size="medium" @row-click="openDoc" style="cursor:pointer" v-loading="docLoading">
          <el-table-column prop="source" label="文档名称" min-width="320" show-overflow-tooltip />
          <el-table-column prop="pages" label="页数" width="80" align="center" />
          <el-table-column prop="chunk_count" label="文本块数" width="100" align="center" />
          <el-table-column label="预览" min-width="200" show-overflow-tooltip>
            <template #default="{ row }">
              <span class="preview-text">{{ row.preview }}</span>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80" align="center">
            <template #default="{ row }">
              <el-button link type="primary" @click.stop="openDoc(row)">查看</el-button>
            </template>
          </el-table-column>
        </el-table>

        <el-pagination
          v-if="docTotal > 0"
          v-model:current-page="docPage"
          :page-size="docPageSize"
          :total="docTotal"
          layout="prev, pager, next"
          style="margin-top:16px; justify-content: flex-end;"
          @current-change="loadDocs"
        />
      </template>

      <!-- ========== 四元组列表 ========== -->
      <template v-else>
        <div class="search-bar">
          <el-input v-model="quadFault" placeholder="故障类型" clearable style="width:160px" />
          <el-input v-model="quadDevice" placeholder="设备" clearable style="width:160px" />
          <el-input v-model="quadKeyword" placeholder="关键词" clearable style="width:160px" />
          <el-button type="primary" @click="loadQuads">搜索</el-button>
          <span class="total-hint">共 {{ quadTotal }} 条</span>
        </div>

        <el-table :data="quadList" stripe size="medium" v-loading="quadLoading">
          <el-table-column prop="device" label="设备" width="120" />
          <el-table-column prop="fault_type" label="故障类型" width="160">
            <template #default="{ row }">{{ FAULT_TYPE_NAME[row.fault_type] ?? row.fault_type }}</template>
          </el-table-column>
          <el-table-column prop="feature_str" label="故障特征" min-width="260" show-overflow-tooltip />
          <el-table-column prop="solution" label="诊断方案" min-width="280" show-overflow-tooltip />
        </el-table>

        <el-pagination
          v-if="quadTotal > 0"
          v-model:current-page="quadPage"
          :page-size="quadPageSize"
          :total="quadTotal"
          layout="prev, pager, next"
          style="margin-top:16px; justify-content: flex-end;"
          @current-change="loadQuads"
        />
      </template>
    </el-card>

    <!-- 文档全文预览弹窗 -->
    <el-dialog v-model="docDialogVisible" :title="currentDoc?.source ?? '文档预览'" width="70%" destroy-on-close>
      <template v-if="currentDoc">
        <div class="doc-meta">
          <el-tag size="small">共 {{ currentDoc.pages }} 页，{{ currentDoc.chunk_count }} 个文本块</el-tag>
        </div>
        <el-tabs v-model="currentDocPage" style="margin-top:12px">
          <el-tab-pane
            v-for="ch in docChunks"
            :key="ch.page"
            :label="`第 ${ch.page} 页`"
            :name="ch.page"
          >
            <div class="doc-text" v-html="formatText(ch.text)"></div>
          </el-tab-pane>
        </el-tabs>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getKBDocs, getKBDocChunks, getKBQuadruples } from '@/api'
import type { DocItem, DocChunk, QuadItem } from '@/types'

// 四元组中存储的故障类型简称 → 完整名称映射
const FAULT_TYPE_NAME: Record<string, string> = {
  '粗轧头尾': '粗轧头尾拉窄',
}

const tab = ref<'docs' | 'quads'>('docs')

// --- 文档 ---
const docKeyword = ref('')
const docList = ref<DocItem[]>([])
const docTotal = ref(0)
const docPage = ref(1)
const docPageSize = ref(20)
const docLoading = ref(false)

const docDialogVisible = ref(false)
const currentDoc = ref<DocItem | null>(null)
const currentDocPage = ref(1)
const docChunks = ref<DocChunk[]>([])

async function loadDocs() {
  docLoading.value = true
  try {
    const res = await getKBDocs(docKeyword.value, docPage.value, docPageSize.value)
    docList.value = res.items
    docTotal.value = res.total
  } finally {
    docLoading.value = false
  }
}

async function openDoc(row: DocItem) {
  currentDoc.value = row
  currentDocPage.value = 1
  docDialogVisible.value = true
  await loadDocChunks(row.id, 1)
}

async function loadDocChunks(docId: string, page: number) {
  const res = await getKBDocChunks(docId, page, 20)
  docChunks.value = res.chunks
}

// --- 四元组 ---
const quadFault = ref('')
const quadDevice = ref('')
const quadKeyword = ref('')
const quadList = ref<QuadItem[]>([])
const quadTotal = ref(0)
const quadPage = ref(1)
const quadPageSize = ref(20)
const quadLoading = ref(false)

async function loadQuads() {
  quadLoading.value = true
  try {
    const res = await getKBQuadruples(quadFault.value, quadDevice.value, quadKeyword.value, quadPage.value, quadPageSize.value)
    quadList.value = res.items
    quadTotal.value = res.total
  } finally {
    quadLoading.value = false
  }
}

function formatText(raw: string): string {
  let s = raw
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  s = s.replace(/\n/g, '<br/>')
  return s
}

onMounted(() => { loadDocs(); loadQuads() })
</script>

<style scoped>
.kb-view { display: flex; flex-direction: column; gap: 0; }
.header-row { display: flex; align-items: center; justify-content: space-between; }
.card-title { font-size: 16px; font-weight: 600; }
.search-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.total-hint { font-size: 12px; color: #909399; margin-left: 8px; }
.preview-text { font-size: 12px; color: #666; }
.doc-meta { margin-bottom: 8px; }
.doc-text { line-height: 1.8; font-size: 14px; white-space: pre-wrap; max-height: 500px; overflow-y: auto; padding: 8px; background: #fafafa; border-radius: 6px; }
</style>
