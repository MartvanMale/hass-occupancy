import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base: './'` is not a preference, it is the whole panel. Ingress serves this
// app from `/api/hassio_ingress/<token>/`, a path nothing here can know at build
// time, so every asset URL and every fetch has to be relative to the document.
// One absolute path and the panel is a blank page with a 404 in the console.
//
// The same rule applies in `api.ts`: `fetch('api/status')`, never `/api/status`.
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
