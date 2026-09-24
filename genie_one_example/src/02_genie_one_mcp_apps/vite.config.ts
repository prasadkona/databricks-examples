import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

const input = process.env.INPUT;
if (!input) throw new Error("INPUT is required");

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    rollupOptions: { input },
    outDir: "dist",
    emptyOutDir: false,
  },
});
