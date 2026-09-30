import axios from 'axios'
import type {
  DiagnoseResult,
  BatchResult,
  DocItem,
  DocChunksResponse,
  QuadItem,
  DiagRecord,
  QueryResponse,
  CsvUploadResponse,
  HealthStatus,
  MetaFeatures,
} from '@/types'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 300000, // 5分钟，LLM 多步推理可能耗时较长
})

// 全局错误拦截：网络错误统一提示
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.code === 'ERR_NETWORK' ? '网络连接失败，请确认后端服务已启动' :
      err.code === 'ERR_TIMEOUT' ? '请求超时，请稍后重试' :
      err.response?.data?.detail || err.message || '请求失败'
    return Promise.reject(new Error(message))
  }
)

// 单条诊断（带钢记录：STRIPNO + 6工艺参数 + 3段曲线）
export const diagnoseStrip = (strip: Record<string, unknown>): Promise<DiagnoseResult> =>
  api.post('/diagnose', { strip }).then((r) => r.data)

// 批量诊断（带钢记录数组）
export const diagnoseStrips = (strips: Record<string, unknown>[]): Promise<BatchResult> =>
  api.post('/diagnose/batch', { strips }).then((r) => r.data)

// 批量诊断（CSV 上传）
export const diagnoseBatchFile = (file: File): Promise<BatchResult> => {
  const form = new FormData()
  form.append('file', file)
  return api
    .post('/diagnose/batch/file', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data)
}

// LLM 问答（可传入 AbortSignal 以支持取消）
// 超时单独放大到 15 分钟：Agent 多步推理 + 重排序器 CPU 批处理，
// 实测复杂批量分析（含 CrossEncoder 重排）可达 13-14 分钟
export const QUERY_TIMEOUT = 900000

export const queryLLM = (
  payload: {
    question: string
    features?: string
    session_id?: string
    conversation_history?: { role: string; content: string }[]
    batch_done?: boolean
    batch_summary?: string
    csv_path?: string
  },
  signal?: AbortSignal
): Promise<QueryResponse> =>
  api.post('/query', payload, { signal, timeout: QUERY_TIMEOUT }).then((r) => r.data)

// 上传 CSV 供会话式批量诊断（返回服务器路径）
export const uploadCSV = (file: File): Promise<CsvUploadResponse> => {
  const form = new FormData()
  form.append('file', file)
  return api
    .post('/csv/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data)
}

// 系统健康
// 注意：健康检查挂在 /api 下（非 /api/v1），不能用上面 baseURL='/api/v1' 的实例，
// 否则请求会打到 /api/v1/health 得到 404，进而让 app store 的 Promise.all 整体失败，
// 表现为状态页全项失真、顶栏恒显"初始化中..."。
export const getHealth = (): Promise<HealthStatus> =>
  axios.get('/api/health').then((r) => r.data)

// 特征元信息
export const getMetaFeatures = (): Promise<MetaFeatures> =>
  api.get('/meta/features').then(r => r.data)

// 知识库文档列表
export const getKBDocs = (
  keyword = '',
  page = 1,
  pageSize = 20
): Promise<{ total: number; page: number; page_size: number; items: DocItem[] }> =>
  api.get('/kb/docs', { params: { keyword, page, page_size: pageSize } }).then(r => r.data)

// 知识库文档全文
export const getKBDocChunks = (
  docId: string,
  page = 1,
  pageSize = 20
): Promise<DocChunksResponse> =>
  api.get(`/kb/docs/${encodeURIComponent(docId)}/chunks`, {
    params: { page, page_size: pageSize },
  }).then(r => r.data)

// 四元组知识库
export const getKBQuadruples = (
  faultCn = '',
  device = '',
  keyword = '',
  page = 1,
  pageSize = 20
): Promise<{ total: number; page: number; page_size: number; items: QuadItem[] }> =>
  api.get('/kb/quadruples', {
    params: { fault_cn: faultCn, device, keyword, page, page_size: pageSize },
  }).then(r => r.data)

// MySQL 诊断记录
export const getDiagRecords = (
  faultCn?: string,
  days?: number,
  page = 1,
  pageSize = 20
): Promise<{
  enabled: boolean
  hint?: string
  total: number
  page: number
  page_size: number
  items: DiagRecord[]
}> =>
  api.get('/diag/records', {
    params: { fault_cn: faultCn || undefined, days: days || undefined, page, page_size: pageSize },
  }).then(r => r.data)
