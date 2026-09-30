import { createRouter, createWebHistory } from 'vue-router'
import SingleDiagnose from '@/views/SingleDiagnose.vue'
import BatchDiagnose from '@/views/BatchDiagnose.vue'
import Chat from '@/views/Chat.vue'
import KnowledgeBase from '@/views/KnowledgeBase.vue'
import DiagnosisRecords from '@/views/DiagnosisRecords.vue'
import Status from '@/views/Status.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/diagnose' },
    { path: '/diagnose', component: SingleDiagnose, meta: { title: '单条诊断' } },
    { path: '/batch', component: BatchDiagnose, meta: { title: '批量诊断' } },
    { path: '/chat', component: Chat, meta: { title: '知识问答' } },
    { path: '/kb', component: KnowledgeBase, meta: { title: '知识库' } },
    { path: '/records', component: DiagnosisRecords, meta: { title: '诊断记录' } },
    { path: '/status', component: Status, meta: { title: '系统状态' } },
  ],
})

export default router
