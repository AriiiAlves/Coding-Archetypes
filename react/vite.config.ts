/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: true },
  // Vitest reads its settings from the same file, so tests share Vite's plugins and aliases.
  test: {
    environment: "jsdom",          // a fake browser DOM for component tests
    globals: true,                 // describe / it / expect without imports
    setupFiles: "./src/test-setup.ts",
  },
});
