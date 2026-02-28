// Vite plugin to serve ML runtime files correctly (WASM + ONNX models)

{
  name: 'serve-ml-assets-raw',

  // DEV: Serve ML files directly, bypassing ALL Vite transforms
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url?.startsWith('/docaligner-ort/') ||
          req.url?.startsWith('/blazedoc/')) {
        const cleanUrl = req.url.split('?')[0];
        const filePath = path.join(__dirname, 'public', cleanUrl);

        if (fs.existsSync(filePath)) {
          const mimeType = {
            '.wasm': 'application/wasm',
            '.mjs':  'application/javascript; charset=utf-8',
            '.ort':  'application/octet-stream',
          }[path.extname(filePath)] || 'application/octet-stream';

          res.setHeader('Content-Type', mimeType);
          res.end(fs.readFileSync(filePath));
          return;
        }
      }
      next();
    });
  },

  // PROD: Remove WASM files from node_modules bundle
  // We use our own custom-built WASM from public/docaligner-ort/
  generateBundle(_options, bundle) {
    for (const fileName in bundle) {
      if (fileName.endsWith('.wasm')) {
        delete bundle[fileName];  // saves 11.8 MB!
      }
    }
  },
}

// Also needed: prevent Vite from pre-bundling onnxruntime-web
// optimizeDeps: { exclude: ['onnxruntime-web'] }
