import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 本文件由 tsconfig.node.json 参与类型检查（lib 为 ES2023，未安装 @types/node），
// 因此不能使用 __dirname / import 'path'（会分别报 TS2304 / TS2307）。
// 下方显式声明 ImportMeta.dirname 后使用 import.meta.dirname——Node 20.11+ 原生提供，
// 无需任何 @types 包。
declare global {
  interface ImportMeta {
    /** Node 20.11+ / 21.2+ 原生提供：当前模块所在目录 */
    dirname: string
  }
}

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': `${import.meta.dirname}/src`,
    },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
