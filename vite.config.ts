import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'api-dev-middleware',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (!req.url) return next();
          const [path] = req.url.split('?');
          if (path === '/api/phone-code' || path === '/api/book-suggestion' || path === '/api/books' || path === '/api/users') {
            const execute = async (body: Record<string, unknown>) => {
              const reqWrapper = Object.assign(req, { body });
              const resWrapper = Object.assign(res, {
                status(code: number) {
                  res.statusCode = code;
                  return resWrapper;
                },
                json(data: unknown) {
                  res.setHeader('Content-Type', 'application/json');
                  res.end(JSON.stringify(data));
                  return resWrapper;
                },
              });
              try {
                const mod =
                  path === '/api/phone-code'
                    ? await import('./api/phone-code.js')
                    : await import('./api/book-suggestion.js');
                await mod.default(reqWrapper, resWrapper);
              } catch (err: unknown) {
                res.statusCode = 500;
                res.setHeader('Content-Type', 'application/json');
                res.end(JSON.stringify({ ok: false, error: String(err) }));
              }
            };

            if (req.method === 'GET' || req.method === 'HEAD' || req.method === 'OPTIONS') {
              void execute({});
              return;
            }

            let bodyStr = '';
            req.on('data', (chunk) => {
              bodyStr += chunk;
            });
            req.on('end', () => {
              let parsedBody: Record<string, unknown> = {};
              try {
                parsedBody = JSON.parse(bodyStr);
              } catch {
                /* ignore */
              }
              void execute(parsedBody);
            });
            return;
          }
          next();
        });
      },
    },
  ],
});
