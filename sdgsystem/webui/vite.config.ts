import { resolve } from 'node:path'
import react from '@vitejs/plugin-react-swc'
import { defineConfig } from 'vite'
import svgr from 'vite-plugin-svgr'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    svgr({
      svgrOptions: {
        exportType: 'default',
      },
      include: '**/*.svg?react',
    }),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
    },
  },

  // 开发服务器代理配置
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, ''), // 如果后端不需要 /api 前缀，取消注释这行
      },
    },
  },

  // 构建优化
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // 将 Ant Design 单独打包
          if (id.includes('node_modules/antd/')) {
            return 'antd-vendor'
          }
          if (id.includes('node_modules/@ant-design/icons/')) {
            return 'antd-icons'
          }
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/')) {
            return 'react-vendor'
          }
        },
      },
    },
  },
})
