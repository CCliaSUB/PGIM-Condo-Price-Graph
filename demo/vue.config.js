const { defineConfig } = require("@vue/cli-service");

module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave: false,

  devServer: {
    proxy: {
      "/api": {
        target: process.env.PGIM_API_URL || "http://localhost:8888",
        changeOrigin: true,
        pathRewrite: { "^/api": "" }, // /api/predict -> /predict
      },
    },
  },
});
