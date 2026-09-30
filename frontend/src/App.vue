<template>
  <el-config-provider :locale="zhCn">
    <el-container class="app-container">
      <el-header class="app-header">
        <div class="header-left">
          <span class="logo">🔍</span>
          <span class="app-title">带钢宽度缺陷智能诊断系统</span>
        </div>
        <div class="header-right">
          <el-tag :type="headerHealthType" size="small">
            {{ headerHealthText }}
          </el-tag>
        </div>
      </el-header>

      <el-container>
        <el-aside width="180px" class="app-aside">
          <el-menu
            :default-active="activeView"
            router
            class="nav-menu"
            @select="(val: string) => (activeView = val)"
          >
            <el-menu-item index="diagnose">
              <el-icon><Histogram /></el-icon>
              <span>单条诊断</span>
            </el-menu-item>
            <el-menu-item index="batch">
              <el-icon><Document /></el-icon>
              <span>批量诊断</span>
            </el-menu-item>
            <el-menu-item index="chat">
              <el-icon><ChatDotRound /></el-icon>
              <span>知识问答</span>
            </el-menu-item>
            <el-menu-item index="kb">
              <el-icon><Folder /></el-icon>
              <span>知识库</span>
            </el-menu-item>
            <el-menu-item index="records">
              <el-icon><List /></el-icon>
              <span>诊断记录</span>
            </el-menu-item>
            <el-menu-item index="status">
              <el-icon><Setting /></el-icon>
              <span>系统状态</span>
            </el-menu-item>
          </el-menu>
        </el-aside>

        <el-main class="app-main">
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </el-container>
  </el-config-provider>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import zhCn from 'element-plus/dist/locale/zh-cn'

const route = useRoute()
const store = useAppStore()
const activeView = ref<string>(route.path.slice(1) || 'diagnose')

const headerHealth = computed(() => store.health)
const headerHealthType = computed(() =>
  headerHealth.value?.status === 'ok' ? 'success' : 'danger'
)
const headerHealthText = computed(() =>
  headerHealth.value?.resources_loaded ? '已连接' : '初始化中...'
)

onMounted(() => {
  store.init()
  activeView.value = route.path.slice(1) || 'diagnose'
})
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; background: #f0f2f5; }
#app { height: 100vh; }
</style>

<style scoped>
.app-container { height: 100vh; }
.app-header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2e5a88 100%);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo { font-size: 22px; }
.app-title { color: #fff; font-size: 16px; font-weight: 600; letter-spacing: 0.5px; }
.header-right { display: flex; align-items: center; gap: 8px; }
.app-aside {
  background: #fff;
  border-right: 1px solid #e4e7ed;
  overflow-y: auto;
}
.nav-menu { border-right: none; flex: 1; }
.nav-menu .el-menu-item { height: 48px; line-height: 48px; }
.app-main { background: #f0f2f5; padding: 20px; overflow-y: auto; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.15s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
