import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:5001'
    }
  },
  base: '/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
    assetsDir: 'assets'
  }
})
