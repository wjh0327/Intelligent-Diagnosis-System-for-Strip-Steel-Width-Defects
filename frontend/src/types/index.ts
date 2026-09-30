// ==================== 诊断相关 ====================

export interface FeatureMeta {
  feature: string
  importance: number
}

export interface DiagnoseResult {
  request_id: string
  fault_cn: string
  fault_desc: string
  confidence: number | null
  needs_review: boolean
  evidence: string
  probabilities: Record<string, number>
  top_features: FeatureMeta[]
}

export interface BatchResult {
  request_id: string
  total: number
  distribution: Record<string, number>
  needs_review_count: number
  results: BatchItem[]
}

export interface BatchItem {
  index: number
  stripno: string
  fault_cn: string
  fault_desc: string
  confidence: number | null
  needs_review: boolean
  evidence: string
  features_named: Record<string, number>
}

// ==================== 知识库相关 ====================

export interface DocItem {
  id: string
  source: string
  pages: number
  preview: string
  chunk_count: number
}

export interface DocChunk {
  page: number
  text: string
}

export interface DocChunksResponse {
  doc_id: string
  total_pages: number
  page: number
  page_size: number
  total_chunks: number
  chunks: DocChunk[]
}

export interface QuadItem {
  id: string
  device: string
  fault_type: string
  feature_str: string
  solution: string
  page: number | null
}

// ==================== 诊断记录相关 ====================

export interface DiagRecord {
  feat_hash: string
  fault_cn: string
  fault_desc: string | null
  confidence: number | null
  features: number[] | null
  updated_at: string
}

// ==================== 问答相关 ====================

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatStep {
  tool: string
  observation: string
}

export interface QueryResponse {
  request_id: string
  session_id: string | null
  answer: string
  steps: ChatStep[]
  batch_done: boolean
  batch_summary: string
  csv_path: string
  history_length: number
}

// CSV 上传
export interface CsvUploadResponse {
  csv_path: string
  file_name: string
  rows: number
  columns: string[]
  missing_features: string[]
  feature_names: string[]
}

// ==================== 系统状态相关 ====================

export interface HealthStatus {
  status: string
  service: string
  resources_loaded: boolean
  redis_mode: string
  llm_configured: boolean
}

export interface MetaFeatures {
  n_features: number
  feature_names: string[]
  fault_types: string[]
}

// ==================== 分页通用 ====================

export interface PageResponse<T> {
  total: number
  page: number
  page_size: number
  items: T[]
}
