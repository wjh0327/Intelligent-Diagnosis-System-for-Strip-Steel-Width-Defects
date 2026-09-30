<template>
  <div class="status-view">
    <el-row :gutter="16">
      <el-col :span="8">
        <el-card shadow="never">
          <template #header><span class="card-title">服务状态</span></template>
          <el-descriptions :column="1" border size="medium">
            <el-descriptions-item label="状态">
              <el-tag :type="store.health?.status === 'ok' ? 'success' : 'danger'">
                {{ store.health?.status ?? '未知' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="服务名">{{ store.health?.service ?? '-' }}</el-descriptions-item>
            <el-descriptions-item label="模型已加载">
              <el-tag :type="store.health?.resources_loaded ? 'success' : 'info'">
                {{ store.health?.resources_loaded ? '是' : '否（首次请求时懒加载）' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="LLM 配置">
              <el-tag :type="store.health?.llm_configured ? 'success' : 'warning'">
                {{ store.health?.llm_configured ? '已配置' : '未配置（问答功能不可用）' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="记忆后端">
              <el-tag :type="store.health?.redis_mode === 'redis' ? 'success' : store.health?.redis_mode === 'file' ? 'warning' : 'info'">
                {{ { redis: 'Redis', file: '文件', memory: '内存' }[store.health?.redis_mode ?? 'memory'] ?? '-' }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="never">
          <template #header><span class="card-title">特征字段（12 维）</span></template>
          <el-table :data="featureRows" stripe size="small">
            <el-table-column type="index" label="#" width="40" />
            <el-table-column prop="name" label="字段名" />
            <el-table-column prop="desc" label="说明" show-overflow-tooltip />
          </el-table>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="never">
          <template #header><span class="card-title">故障类型</span></template>
          <el-tag
            v-for="t in faultTypes"
            :key="t"
            type="info"
            size="large"
            style="margin:4px"
          >{{ t }}</el-tag>
        </el-card>
      </el-col>
    </el-row>

    <el-alert
      v-if="store.error"
      :title="`初始化错误：${store.error}`"
      type="error"
      :closable="false"
      style="margin-top:16px"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'

const store = useAppStore()

const featureRows = computed(() =>
  (store.metaFeatures?.feature_names ?? []).map((name: string) => ({
    name,
    desc: FEATURE_DESC[name] ?? '',
  }))
)

const faultTypes = computed(() => store.metaFeatures?.fault_types ?? [])

const FEATURE_DESC: Record<string, string> = {
  FMWIDTHACTHOT: '精轧出口实际平均宽度热值',
  FMWTARGETCOL: '精轧出口实际平均宽度冷值',
  FMWTARGETHOT: '精轧出口目标平均宽度热值',
  PDIWIDTHTOL: '带钢计划宽度余量',
  RDWTARGETTOTAL: '粗轧出口目标平均宽度热值',
  RMWIDTHACTHOT: '粗轧出口实际平均宽度热值',
  is_DM: '卷取是否有波谷',
  is_FM: '精轧是否有波谷',
  is_HT: '精轧粗轧波谷位置是否同时在头部或尾部',
  is_NA: '精轧和卷取宽度是否整体分布均匀且小于精轧目标宽度',
  is_RM: '粗轧是否有波谷',
  is_WS: '带钢中部宽度是否不均匀',
}

onMounted(() => { store.init() })
</script>

<style scoped>
.status-view { display: flex; flex-direction: column; gap: 16px; }
.card-title { font-size: 15px; font-weight: 600; }
</style>
