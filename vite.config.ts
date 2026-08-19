import { defineConfig } from "vite";

// GitHub Pages などサブパス配信でもそのまま置けるよう相対 base にしておく
export default defineConfig({
  base: "./",
});
