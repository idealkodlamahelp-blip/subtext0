// Subtext — Node HTTP server for Render.
// Serves the built frontend (dist/) and dispatches /api/* requests to the
// Vercel-style handler modules in api/.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(__dirname, 'dist');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.pdf': 'application/pdf',
  '.txt': 'text/plain; charset=utf-8',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

function readBody(req) {
  return new Promise((resolve) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 5_000_000) req.destroy();
    });
    req.on('end', () => {
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        resolve({});
      }
    });
    req.on('error', () => resolve({}));
  });
}

/** Wrap a Node ServerResponse with the res.status()/.json() helpers the API handlers use. */
function wrapRes(res) {
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (value) => {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify(value));
  };
  return res;
}

async function callApi(pathname, req, res) {
  const modulePath = path.join(__dirname, pathname + '.js');
  let mod;
  try {
    mod = await import(pathToFileURL(modulePath).href);
  } catch (err) {
    console.error('[api] import failed', pathname, err);
    res.statusCode = 404;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ ok: false, error: `api not found: ${pathname}` }));
    return;
  }
  const handler = mod.default;
  if (typeof handler !== 'function') {
    res.statusCode = 500;
    res.end(`no handler in ${pathname}`);
    return;
  }
  const body = await readBody(req);
  req.body = body;
  wrapRes(res);
  try {
    await handler(req, res);
  } catch (err) {
    console.error('[api] handler failed', pathname, err);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ ok: false, error: String(err) }));
  }
}

async function serveStatic(req, res) {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    pathname = '/';
  }
  if (pathname === '/') pathname = '/index.html';

  let data;
  try {
    const filePath = path.join(DIST, pathname);
    if (!filePath.startsWith(DIST + path.sep) && filePath !== DIST) throw new Error('escape');
    data = await readFile(filePath);
  } catch {
    // SPA fallback: any deep link (e.g. /books/123) serves index.html
    try {
      data = await readFile(path.join(DIST, 'index.html'));
    } catch {
      data = Buffer.from('<!doctype html><title>Subtext</title>');
    }
  }
  res.setHeader('Content-Type', MIME[path.extname(pathname)] ?? 'application/octet-stream');
  res.end(data);
}

const server = http.createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/')) {
    await callApi(pathname, req, res);
  } else {
    await serveStatic(req, res);
  }
});

const port = Number(process.env.PORT || 4173);
server.listen(port, () => {
  console.log(`subtext server on http://localhost:${port}`);
});
