import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, CircleSlash, Loader2, Radio, Zap } from 'lucide-react';

/**
 * Live API activity: one hub ("Translate Router") and one node per credential.
 *
 * Data comes from two places only — `GET /api/api-live/snapshot` for the first
 * paint, then `api_event` frames on the existing /ws socket. No polling.
 *
 * Rendering is deliberately cheap: events are queued and applied once per
 * requestAnimationFrame, travelling dots are capped, and their motion is a single
 * CSS transform. `prefers-reduced-motion` drops the dots and keeps the colours.
 */

interface LiveNode {
  id: string;
  label: string;
  provider: string;
  model: string;
  key_hint: string;
  enabled: boolean;
  state: string;
  cooldown_remaining?: number;
  requests: number;
  successes: number;
  failures: number;
  tokens_total: number;
  last_latency_ms: number;
  last_error?: string;
}

interface ApiEvent {
  type: 'api_event';
  phase: 'start' | 'success' | 'fail' | 'retry' | 'cache_hit';
  request_id: string;
  node_id: string;
  provider: string;
  model: string;
  lines: number;
  latency_ms: number;
  prompt_tokens: number;
  output_tokens: number;
  thoughts_tokens: number;
  http_status: number;
  error_short: string;
  key_hint?: string;
}

interface LogRow {
  at: number;
  node_id: string;
  label: string;
  model: string;
  lines: number;
  latency_ms: number;
  tokens: number;
  status: string;
  error_short: string;
  http_status: number;
}

interface Snapshot {
  nodes: LiveNode[];
  totals: { requests: number; successes: number; failures: number; tokens_total: number };
  req_per_min: number;
  recent: LogRow[];
}

interface Flight {
  key: string;
  x: number;
  y: number;
  dx: number;
  dy: number;
  color: string;
}

const PROVIDER_LABELS: Record<string, string> = {
  gemini: 'Gemini', openai: 'OpenAI', codecraft: 'CodeCraft', custom: 'Custom API',
};

const API_BASE = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';
const MAX_FLIGHTS = 30;

function wsUrl(): string {
  const base = import.meta.env.DEV
    ? `${window.location.protocol === 'https:' ? 'wss://' : 'ws://'}${window.location.host}`
    : 'ws://127.0.0.1:8001';
  return `${base}/ws`;
}

const nf = new Intl.NumberFormat('en-US');

function formatTokens(value: number): string {
  if (!value) return '0';
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k` : String(value);
}

function ago(seconds: number): string {
  if (seconds <= 0) return 'now';
  if (seconds < 60) return `${seconds}s`;
  return `${Math.round(seconds / 60)}m`;
}

export default function ApiLiveView() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [nodes, setNodes] = useState<LiveNode[]>([]);
  const [totals, setTotals] = useState({ requests: 0, successes: 0, failures: 0, tokens_total: 0 });
  const [log, setLog] = useState<LogRow[]>([]);
  const [flights, setFlights] = useState<Flight[]>([]);
  const [flashes, setFlashes] = useState<Record<string, { state: string; at: number }>>({});
  const [ripples, setRipples] = useState<number[]>([]);
  const [floats, setFloats] = useState<Array<{ key: string; x: number; y: number; text: string }>>([]);
  const [live, setLive] = useState(false);
  const [cooldowns, setCooldowns] = useState<Record<string, number>>({});
  const [testBusy, setTestBusy] = useState(false);
  const [testNote, setTestNote] = useState<string>('');
  const [size, setSize] = useState({ w: 760, h: 360 });
  const [tick, setTick] = useState(0);

  const boxRef = useRef<HTMLDivElement | null>(null);
  const queueRef = useRef<ApiEvent[]>([]);
  const frameRef = useRef<number | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number>(0);
  const counterRef = useRef(0);
  const reduced = useRef(false);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    reduced.current = query.matches;
    const onChange = () => { reduced.current = query.matches; };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  // Cooldown countdowns only need a slow tick.
  useEffect(() => {
    const timer = window.setInterval(() => setTick((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const element = boxRef.current;
    if (!element) return undefined;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setSize({ w: rect.width, h: rect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const geometry = useMemo(() => {
    const cx = size.w / 2;
    const cy = size.h / 2;
    const count = nodes.length;
    const rx = Math.max(90, Math.min(size.w * 0.34, 320));
    const ry = Math.max(70, Math.min(size.h * 0.36, 150));
    const points = nodes.map((_, index) => {
      const angle = -Math.PI / 2 + (index * 2 * Math.PI) / Math.max(count, 1);
      return { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry };
    });
    return { cx, cy, points };
  }, [nodes, size.w, size.h]);

  const nodeIndex = useMemo(() => {
    const map = new Map<string, number>();
    nodes.forEach((node, index) => map.set(node.id, index));
    return map;
  }, [nodes]);

  const applyEvents = useCallback((events: ApiEvent[]) => {
    if (!events.length) return;
    const newFlights: Flight[] = [];
    const newFloats: Array<{ key: string; x: number; y: number; text: string }> = [];
    const now = Date.now();

    setNodes((current) => {
      if (!current.length) return current;
      const next = current.map((node) => ({ ...node }));
      const byId = new Map(next.map((node) => [node.id, node]));
      for (const event of events) {
        if (event.phase === 'cache_hit') continue;
        const node = byId.get(event.node_id);
        if (!node) continue;
        if (event.phase === 'success') {
          node.successes += 1;
          node.tokens_total += event.prompt_tokens + event.output_tokens + event.thoughts_tokens;
          node.last_latency_ms = event.latency_ms;
          node.state = 'ready';
          if (event.model) node.model = event.model;
          setCooldowns((value) => {
            if (!(event.node_id in value)) return value;
            const next = { ...value };
            delete next[event.node_id];
            return next;
          });
        }
        if (event.phase === 'start') node.requests += 1;
        if (event.phase === 'fail' || event.phase === 'retry') {
          node.failures += 1;
          node.last_latency_ms = event.latency_ms;
          node.state = event.http_status === 429 ? 'cooling' : 'failed';
          if (event.http_status === 429) {
            setCooldowns((value) => ({ ...value, [event.node_id]: now + 60_000 }));
          }
        }
      }
      return next;
    });

    setTotals((current) => {
      const next = { ...current };
      for (const event of events) {
        if (event.phase === 'success') {
          next.successes += 1;
          next.tokens_total += event.prompt_tokens + event.output_tokens + event.thoughts_tokens;
        }
        if (event.phase === 'fail' || event.phase === 'retry') next.failures += 1;
        if (event.phase === 'start') next.requests += 1;
      }
      return next;
    });

    setFlashes((current) => {
      const next = { ...current };
      for (const event of events) {
        if (event.phase === 'cache_hit' || !event.node_id) continue;
        next[event.node_id] = {
          state: event.phase === 'success' ? 'success' : event.phase === 'start' ? 'sending' : event.phase,
          at: now,
        };
      }
      return next;
    });

    for (const event of events) {
      if (event.phase === 'cache_hit') {
        setRipples((value) => [...value.slice(-4), now]);
        continue;
      }
      const index = nodeIndex.get(event.node_id);
      if (index === undefined) continue;
      const point = geometry.points[index];
      if (!point) continue;
      const hub = { x: geometry.cx, y: geometry.cy };
      if (!reduced.current && newFlights.length < MAX_FLIGHTS) {
        const outbound = event.phase === 'start' || event.phase === 'retry';
        const from = outbound ? hub : point;
        const to = outbound ? point : hub;
        newFlights.push({
          key: `${event.request_id}-${event.phase}-${counterRef.current += 1}`,
          x: from.x, y: from.y, dx: to.x - from.x, dy: to.y - from.y,
          color: event.phase === 'success' ? '#34d399'
            : event.phase === 'start' ? '#38bdf8'
              : event.phase === 'retry' ? '#fbbf24' : '#f87171',
        });
      }
      if (event.phase === 'success') {
        const tokens = event.prompt_tokens + event.output_tokens + event.thoughts_tokens;
        if (tokens) {
          newFloats.push({
            key: `${event.request_id}-tok`,
            x: point.x, y: point.y - 26,
            text: `+${nf.format(tokens)} tok`,
          });
        }
      }
    }

    if (newFlights.length) setFlights((value) => [...value, ...newFlights].slice(-MAX_FLIGHTS));
    if (newFloats.length) setFloats((value) => [...value, ...newFloats].slice(-8));

    setLog((current) => {
      const rows = events
        .filter((event) => event.phase !== 'start' && event.phase !== 'cache_hit')
        .map<LogRow>((event) => ({
          at: now,
          node_id: event.node_id,
          label: '',
          model: event.model,
          lines: event.lines,
          latency_ms: event.latency_ms,
          tokens: event.prompt_tokens + event.output_tokens + event.thoughts_tokens,
          status: event.phase,
          error_short: event.error_short,
          http_status: event.http_status,
        }));
      return [...rows.reverse(), ...current].slice(0, 50);
    });
  }, [geometry, nodeIndex]);

  const drain = useCallback(() => {
    frameRef.current = null;
    const events = queueRef.current;
    queueRef.current = [];
    applyEvents(events);
  }, [applyEvents]);

  const enqueue = useCallback((event: ApiEvent) => {
    queueRef.current.push(event);
    if (queueRef.current.length > 200) queueRef.current = queueRef.current.slice(-200);
    if (frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(drain);
    }
  }, [drain]);

  const loadSnapshot = useCallback(async () => {
    // A dev backend that is still starting (or reloading) answers 500 for a
    // moment; retry instead of leaving the tab empty.
    for (let attempt = 0; attempt < 4; attempt += 1) {
      try {
        const response = await fetch(`${API_BASE}/api/api-live/snapshot`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = (await response.json()) as Snapshot;
        setSnapshot(data);
        setNodes(data.nodes.map((node) => ({ ...node })));
        setTotals({ ...data.totals });
        setLog(data.recent ?? []);
        const until: Record<string, number> = {};
        for (const node of data.nodes) {
          if (node.state === 'cooling' && node.cooldown_remaining) {
            until[node.id] = Date.now() + node.cooldown_remaining * 1000;
          }
        }
        setCooldowns(until);
        return;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    }
  }, []);

  useEffect(() => {
    void loadSnapshot();
  }, [loadSnapshot]);

  // The socket must connect once: a changing dependency here (geometry) would
  // tear it down and reopen it on every node move, resetting the counters.
  const enqueueRef = useRef(enqueue);
  useEffect(() => {
    enqueueRef.current = enqueue;
  }, [enqueue]);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      if (closed) return;
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl());
      } catch {
        retryRef.current = window.setTimeout(connect, 2000);
        return;
      }
      socketRef.current = socket;
      socket.onopen = () => {
        setLive(true);
        void loadSnapshot();
      };
      socket.onmessage = (message) => {
        try {
          const data = JSON.parse(String(message.data)) as ApiEvent | { type?: string };
          if (data && (data as ApiEvent).type === 'api_event') enqueueRef.current(data as ApiEvent);
        } catch {
          // Progress frames from the workflow socket are ignored on purpose.
        }
      };
      socket.onclose = () => {
        setLive(false);
        if (!closed) retryRef.current = window.setTimeout(connect, 2000);
      };
      socket.onerror = () => socket.close();
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retryRef.current);
      socketRef.current?.close();
    };
  }, [loadSnapshot]);

  const test = async () => {
    setTestBusy(true);
    setTestNote('');
    try {
      const response = await fetch(`${API_BASE}/api/api-live/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: nodes.find((node) => node.provider === 'gemini')?.provider
            ?? nodes[0]?.provider ?? 'gemini',
        }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`);
      setTestNote(body.ok ? 'Test batch answered — watch the nodes.' : 'Test reached the API but no line came back.');
    } catch (error) {
      setTestNote(error instanceof Error ? error.message : 'Test failed');
    } finally {
      setTestBusy(false);
    }
  };

  const now = Date.now();
  const coolingCount = nodes.filter((node) => (cooldowns[node.id] ?? 0) > now).length;
  const failed = nodes.some((node) => node.state === 'failed' && (cooldowns[node.id] ?? 0) <= now);
  const badge = failed ? 'Attention' : coolingCount ? `${coolingCount} cooling` : nodes.length ? 'All ready' : 'No nodes';
  const badgeTone = failed ? 'text-rose-300 border-rose-500/40 bg-rose-500/10'
    : coolingCount ? 'text-amber-300 border-amber-500/40 bg-amber-500/10'
      : 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10';

  return (
    <div className="space-y-4">
      <style>{`
        @keyframes api-fly { 0% { transform: translate(0,0) scale(.4); opacity: 0 } 18% { opacity: 1 } 100% { transform: translate(var(--dx), var(--dy)) scale(1); opacity: 0 } }
        @keyframes api-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,.5) } 50% { box-shadow: 0 0 0 9px rgba(56,189,248,0) } }
        @keyframes api-shake { 0%,100% { transform: translateX(0) } 25% { transform: translateX(-4px) } 50% { transform: translateX(4px) } 75% { transform: translateX(-2px) } }
        @keyframes api-ripple { 0% { transform: translate(-50%,-50%) scale(.7); opacity: .5 } 100% { transform: translate(-50%,-50%) scale(1.8); opacity: 0 } }
        @keyframes api-float { 0% { transform: translate(-50%,0); opacity: 1 } 100% { transform: translate(-50%,-26px); opacity: 0 } }
        .api-fly { animation: api-fly 560ms ease-out forwards }
        .api-ripple { animation: api-ripple 700ms ease-out forwards }
        .api-float { animation: api-float 900ms ease-out forwards }
        .api-sending { animation: api-pulse 1.2s ease-in-out infinite }
        .api-shake { animation: api-shake 380ms ease-in-out }
        @media (prefers-reduced-motion: reduce) {
          .api-fly, .api-ripple, .api-float, .api-sending, .api-shake { animation: none !important }
        }
      `}</style>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Radio size={14} className={live ? 'text-emerald-400' : 'text-slate-600'} />
          {live ? 'Live over WebSocket' : 'Disconnected — retrying'}
          <span className="text-slate-600">·</span>
          <span>{Math.round(snapshot?.req_per_min ?? 0)} req/min</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void test()}
            disabled={testBusy || !nodes.length}
            className="flex items-center gap-1.5 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 transition hover:bg-sky-500/20 disabled:opacity-40"
          >
            {testBusy ? <Loader2 size={13} className="animate-spin" /> : <Zap size={13} />}
            {testBusy ? 'Sending…' : 'Test 3 lines'}
          </button>
          <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold ${badgeTone}`}>{badge}</span>
        </div>
      </div>

      {testNote && (
        <p className="rounded-lg border border-white/10 bg-slate-900/70 px-3 py-2 text-[11px] text-slate-300">{testNote}</p>
      )}

      <div
        ref={boxRef}
        className="relative h-[360px] overflow-hidden rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_50%_45%,rgba(14,165,233,0.10),rgba(2,6,23,0.6))]"
      >
        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden>
          {geometry.points.map((point, index) => (
            <line
              key={nodes[index]?.id ?? index}
              x1={geometry.cx} y1={geometry.cy} x2={point.x} y2={point.y}
              stroke="rgba(148,163,184,0.28)" strokeWidth={1}
              strokeDasharray="4 6"
            />
          ))}
        </svg>

        <div
          className="absolute z-10 flex h-28 w-28 flex-col items-center justify-center rounded-full border border-sky-400/40 bg-slate-950/90 text-center shadow-[0_0_40px_rgba(14,165,233,0.25)]"
          style={{ left: geometry.cx, top: geometry.cy, transform: 'translate(-50%,-50%)' }}
        >
          <Activity size={18} className="text-sky-300" />
          <p className="mt-1 text-[11px] font-semibold text-slate-100">Translate Router</p>
          <p className="text-[10px] text-slate-400">{totals.requests} req · {formatTokens(totals.tokens_total)} tok</p>
          {ripples.map((at) => (
            <span
              key={at}
              className="api-ripple pointer-events-none absolute left-1/2 top-1/2 h-28 w-28 rounded-full border border-sky-300/60"
              onAnimationEnd={() => setRipples((value) => value.filter((item) => item !== at))}
            />
          ))}
        </div>

        {flights.map((flight) => (
          <span
            key={flight.key}
            className="api-fly pointer-events-none absolute h-2 w-2 rounded-full"
            style={{
              left: flight.x, top: flight.y, background: flight.color,
              boxShadow: `0 0 10px ${flight.color}`,
              ['--dx' as string]: `${flight.dx}px`,
              ['--dy' as string]: `${flight.dy}px`,
            } as React.CSSProperties}
            onAnimationEnd={() => setFlights((value) => value.filter((item) => item.key !== flight.key))}
          />
        ))}

        {floats.map((float) => (
          <span
            key={float.key}
            className="api-float pointer-events-none absolute z-20 whitespace-nowrap text-[10px] font-semibold text-emerald-300"
            style={{ left: float.x, top: float.y }}
            onAnimationEnd={() => setFloats((value) => value.filter((item) => item.key !== float.key))}
          >
            {float.text}
          </span>
        ))}

        {nodes.map((node, index) => {
          const point = geometry.points[index];
          if (!point) return null;
          const flash = flashes[node.id];
          const cooling = (cooldowns[node.id] ?? 0) > now;
          const state = cooling ? 'cooling' : !node.enabled ? 'disabled' : flash?.state ?? node.state;
          const tone = state === 'success' ? 'border-emerald-400/60 bg-emerald-500/10'
            : state === 'fail' || state === 'failed' ? 'border-rose-400/60 bg-rose-500/10'
              : state === 'retry' || state === 'cooling' || state === 'cooldown' ? 'border-amber-400/60 bg-amber-500/10'
                : state === 'disabled' ? 'border-slate-700 bg-slate-900/60'
                  : state === 'sending' ? 'border-sky-400/70 bg-sky-500/10' : 'border-cyan-500/30 bg-slate-900/80';
          const dot = state === 'success' ? 'bg-emerald-400'
            : state === 'fail' || state === 'failed' ? 'bg-rose-400'
              : cooling ? 'bg-amber-400' : !node.enabled ? 'bg-slate-600' : 'bg-cyan-400';
          return (
            <div
              key={node.id}
              className={`absolute w-[152px] rounded-xl border px-2.5 py-2 backdrop-blur transition-colors ${tone} ${state === 'sending' ? 'api-sending' : ''} ${state === 'fail' ? 'api-shake' : ''}`}
              style={{ left: point.x, top: point.y, transform: 'translate(-50%,-50%)' }}
              title={node.last_error ?? undefined}
            >
              <div className="flex items-center gap-1.5">
                <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                <p className="truncate text-[11px] font-medium text-slate-100">{node.label}</p>
                {flash?.state === 'retry' && (
                  <span className="ml-auto rounded bg-amber-500/20 px-1 text-[9px] text-amber-200">retry</span>
                )}
              </div>
              <p className="mt-0.5 truncate font-mono text-[10px] text-slate-400">{node.model || '—'}</p>
              <p className="truncate font-mono text-[10px] text-slate-500">{node.key_hint || 'saved key'}</p>
              <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                <span>⇄ {node.requests}</span>
                <span>{node.last_latency_ms ? `${node.last_latency_ms} ms` : '—'}</span>
                <span>{formatTokens(node.tokens_total)}</span>
              </div>
              {cooling && (
                <p className="mt-0.5 text-[9px] text-amber-300">
                  cooling {ago(Math.round(((cooldowns[node.id] ?? now) - now) / 1000))}
                </p>
              )}
            </div>
          );
        })}

        {!nodes.length && (
          <p className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
            No credentials yet — add a key or a pool entry.
          </p>
        )}
      </div>

      <div className="rounded-xl border border-white/10 bg-slate-950/60">
        <div className="flex items-center justify-between border-b border-white/5 px-3 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Event log</p>
          <p className="text-[10px] text-slate-600">{log.length} / 50</p>
        </div>
        <div className="max-h-52 overflow-y-auto px-3 py-1 font-mono text-[10px] leading-5">
          {!log.length && <p className="py-2 text-slate-600">Waiting for API activity…</p>}
          {log.map((row, index) => {
            const node = nodes.find((item) => item.id === row.node_id);
            const label = node?.label ?? row.node_id;
            const tone = row.status === 'success' ? 'text-emerald-300'
              : row.status === 'retry' ? 'text-amber-300' : 'text-rose-300';
            return (
              <div key={`${row.at}-${index}`} className="flex gap-2 whitespace-nowrap">
                <span className="text-slate-600">{new Date(row.at).toLocaleTimeString()}</span>
                <span className="w-24 truncate text-slate-400">{label}</span>
                <span className="w-36 truncate text-slate-500">{row.model || '—'}</span>
                <span className="w-8 text-slate-500">{row.lines || ''}</span>
                <span className="w-14 text-slate-500">{row.latency_ms} ms</span>
                <span className="w-16 text-slate-500">{formatTokens(row.tokens)} tok</span>
                <span className={tone}>
                  {row.status}{row.http_status ? ` ${row.http_status}` : ''}
                  {row.error_short ? ` · ${row.error_short}` : ''}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <p className="flex items-center gap-1.5 text-[10px] text-slate-600">
        {failed ? <AlertTriangle size={11} className="text-rose-400" />
          : coolingCount ? <CircleSlash size={11} className="text-amber-400" />
            : <CheckCircle2 size={11} className="text-emerald-400" />}
        Keys are never shown — only the last four characters. Counters reset when the service restarts.
        {' '}Providers: {Object.values(PROVIDER_LABELS).join(' · ')}
      </p>
    </div>
  );
}
