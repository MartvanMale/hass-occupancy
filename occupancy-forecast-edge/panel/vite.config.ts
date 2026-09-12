import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base: './'` is not a preference, it is the whole panel: Ingress serves this
// from a token path nothing can know at build time, so every URL must be
// relative or the panel is a blank page. Same rule in `api.ts`.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // No hashed chunk splitting beyond the default: the panel is served by the
    // add-on itself over a LAN, so a second round trip costs more than the bytes.
    chunkSizeWarningLimit: 800,
  },
})
