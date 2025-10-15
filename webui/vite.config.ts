import { defineConfig } from 'vite'
import viteReact from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

import tanstackRouter from '@tanstack/router-plugin/vite'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [
    tanstackRouter({ autoCodeSplitting: true }),
    viteReact(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/admin': {
        target: process.env.VITE_DF_SERVER_URL || 'http://localhost:8086/',
        changeOrigin: true,
      },
    },
  },
})
