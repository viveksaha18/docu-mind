import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  if (command === "build" && !env.VITE_API_BASE_URL?.trim()) {
    throw new Error("Set VITE_API_BASE_URL to the deployed backend URL before building for production.");
  }

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": "http://127.0.0.1:8000",
      },
    },
  };
});
