declare module 'element-plus/dist/locale/zh-cn' {
  const zhCn: any
  export default zhCn
}

declare module 'element-plus/dist/index.css' {}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
