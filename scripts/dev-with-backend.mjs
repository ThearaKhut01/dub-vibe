/**
 * Start the Vite dev server once the API is actually listening.
 *
 * `npm run dev` used to launch both at the same time, so a page that was already
 * open fired every `/api/...` request into a backend that had not finished
 * booting yet: Vite's proxy logged a wall of ECONNREFUSED, and the UI showed
 * empty panels until it was refreshed. Waiting for /health costs about a second
 * and removes the race; if the backend never answers, Vite still starts so the
 * frontend can be worked on alone.
 */
import { spawn } from 'node:child_process';

const healthUrl = process.env.DUBVIBE_HEALTH_URL ?? 'http://127.0.0.1:8001/health';
const timeoutMs = Number(process.env.DUBVIBE_HEALTH_TIMEOUT_MS ?? 90_000);
const deadline = Date.now() + timeoutMs;

async function waitForBackend() {
  while (Date.now() < deadline) {
    try {
      const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1500) });
      if (response.ok) return true;
    } catch {
      // not listening yet — keep waiting
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
}

const ready = await waitForBackend();
if (ready) {
  console.log('[dev] backend is up — starting Vite');
} else {
  console.warn(`[dev] no answer from ${healthUrl} after ${Math.round(timeoutMs / 1000)}s — starting Vite anyway`);
}

// A fixed command string (no user input): passing args alongside `shell` is
// deprecated, and Node refuses to spawn the .cmd shim directly on Windows.
const child = spawn('npm --prefix frontend run dev -- --host 0.0.0.0', {
  stdio: 'inherit',
  shell: true,
});

child.on('exit', (code) => process.exit(code ?? 0));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
