import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Check, Database, Pencil, Plus, RefreshCw, RotateCcw, Trash2, Zap } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import type { ApiPoolEntry, ApiPoolProviderId, ApiPoolStrategy } from '../types';

/**
 * API Pool Manager — the panel for the backend's multi-provider pool.
 *
 * It is a view over `/api/api-pool/*` only: no credential is ever rendered (the
 * backend sends `has_key` and the last four characters), and every number shown
 * is either measured by the backend or a limit the user typed in here. A failed
 * configuration is explained ("cooling down 42s", "daily limit reached 20/20")
 * instead of quietly doing nothing.
 */

const PROVIDER_ORDER: ApiPoolProviderId[] = ['gemini', 'openai', 'codecraft', 'custom'];

const PROVIDER_META: Record<ApiPoolProviderId, {
  label: string;
  defaultModel: string;
  modelPlaceholder: string;
  baseUrlHint: string;
}> = {
  gemini: {
    label: 'Google Gemini',
    defaultModel: 'gemini-flash-latest',
    modelPlaceholder: 'gemini-flash-latest',
    baseUrlHint: 'Optional — an OpenAI-style gateway in front of Gemini',
  },
  openai: {
    label: 'OpenAI',
    defaultModel: 'gpt-4o-mini',
    modelPlaceholder: 'gpt-4o-mini',
    baseUrlHint: 'Optional — a compatible proxy (https://…/v1)',
  },
  codecraft: {
    label: 'CodeCraft',
    defaultModel: 'deepseek-v4-flash-0731',
    modelPlaceholder: 'deepseek-v4-flash-0731',
    baseUrlHint: 'Optional — defaults to codecraftapi.com/v1',
  },
  custom: {
    label: 'Custom API',
    defaultModel: '',
    modelPlaceholder: 'mistralai/mistral-large-2512',
    baseUrlHint: 'Required — https://…/v1 or http://localhost:11434/v1',
  },
};

const STRATEGY: Record<ApiPoolStrategy, { label: string; hint: string }> = {
  round_robin: { label: 'Round robin', hint: 'One request per configuration, in order.' },
  least_used: { label: 'Least used', hint: 'The configuration with the fewest requests today goes first.' },
  hybrid: { label: 'Smart hybrid', hint: 'Prefers a healthy configuration with allowance left and little recent load.' },
};

type Tone = 'emerald' | 'amber' | 'rose' | 'slate' | 'sky';

const TONE_CLASS: Record<Tone, string> = {
  emerald: 'border-emerald-500/25 bg-emerald-500/15 text-emerald-200',
  amber: 'border-amber-500/25 bg-amber-500/15 text-amber-200',
  rose: 'border-rose-500/25 bg-rose-500/15 text-rose-200',
  sky: 'border-sky-500/25 bg-sky-500/15 text-sky-200',
  slate: 'border-white/10 bg-slate-700/40 text-slate-300',
};

function Chip({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${TONE_CLASS[tone]}`}>{children}</span>;
}

function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

interface Draft {
  id: string;
  provider: ApiPoolProviderId;
  label: string;
  model: string;
  base_url: string;
  key: string;
  daily_limit: string;
  monthly_limit: string;
  weight: string;
  enabled: boolean;
}

const emptyDraft = (provider: ApiPoolProviderId = 'openai'): Draft => ({
  id: '',
  provider,
  label: '',
  model: PROVIDER_META[provider].defaultModel,
  base_url: '',
  key: '',
  daily_limit: '',
  monthly_limit: '',
  weight: '1',
  enabled: true,
});

const draftFromEntry = (entry: ApiPoolEntry): Draft => ({
  id: entry.id,
  provider: entry.provider,
  label: entry.label,
  model: entry.model,
  base_url: entry.base_url,
  key: '',
  daily_limit: entry.daily_limit === null ? '' : String(entry.daily_limit),
  monthly_limit: entry.monthly_limit === null ? '' : String(entry.monthly_limit),
  weight: String(entry.weight),
  enabled: entry.enabled,
});

export default function ApiPoolManager() {
  const {
    apiPool,
    refreshApiPool,
    savePoolEntry,
    deletePoolEntry,
    testPoolEntry,
    saveApiPoolSettings,
    retryPoolQueue,
    clearPoolCache,
    importLegacyPool,
    resetPoolHealth,
  } = useStudioStore();

  const [draft, setDraft] = useState<Draft | null>(null);
  const [draftError, setDraftError] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [messageTone, setMessageTone] = useState<Tone>('slate');
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, string>>({});
  const [settingsDraft, setSettingsDraft] = useState({
    strategy: 'hybrid' as ApiPoolStrategy,
    cache_enabled: true,
    cache_ttl_days: '7',
    max_attempts: '3',
    max_parallel: '4',
  });

  useEffect(() => {
    void refreshApiPool();
  }, [refreshApiPool]);

  useEffect(() => {
    if (!apiPool) return;
    setSettingsDraft({
      strategy: apiPool.settings.strategy,
      cache_enabled: apiPool.settings.cache_enabled,
      cache_ttl_days: String(apiPool.settings.cache_ttl_days),
      max_attempts: String(apiPool.settings.max_attempts),
      max_parallel: String(apiPool.settings.max_parallel),
    });
  }, [apiPool?.settings.strategy, apiPool?.settings.cache_enabled, apiPool?.settings.cache_ttl_days, apiPool?.settings.max_attempts, apiPool?.settings.max_parallel]);

  const groups = useMemo(
    () => PROVIDER_ORDER.map((provider) => ({
      provider,
      info: apiPool?.providers?.[provider],
      entries: (apiPool?.entries ?? []).filter((entry) => entry.provider === provider),
    })),
    [apiPool],
  );

  const report = (text: string, tone: Tone = 'slate') => {
    setMessage(text);
    setMessageTone(tone);
  };

  const run = async (key: string, action: () => Promise<void>) => {
    setBusy(key);
    try {
      await action();
    } catch (error) {
      report(error instanceof Error ? error.message : 'Something went wrong.', 'rose');
    } finally {
      setBusy(null);
    }
  };

  const handleSave = async () => {
    if (!draft) return;
    setDraftError('');
    if (!draft.model.trim()) {
      setDraftError('A model name is required.');
      return;
    }
    if (draft.provider === 'custom' && !draft.base_url.trim() && !(apiPool?.providers?.custom?.has_legacy_credential)) {
      setDraftError('A Custom API configuration needs a base URL.');
      return;
    }
    await run('save-entry', async () => {
      await savePoolEntry({
        id: draft.id || undefined,
        provider: draft.provider,
        label: draft.label.trim(),
        model: draft.model.trim(),
        base_url: draft.base_url.trim(),
        key: draft.key.trim() || undefined,
        daily_limit: draft.daily_limit.trim() === '' ? null : Number(draft.daily_limit),
        monthly_limit: draft.monthly_limit.trim() === '' ? null : Number(draft.monthly_limit),
        weight: Number(draft.weight) || 1,
        enabled: draft.enabled,
      });
      setDraft(null);
      report(draft.id ? 'Configuration updated.' : 'Configuration added.', 'emerald');
    });
  };

  const handleTest = async (entry: ApiPoolEntry) => {
    await run(`test-${entry.id}`, async () => {
      try {
        const result = await testPoolEntry(entry.id);
        setTestResults((current) => ({ ...current, [entry.id]: result }));
        report(result, 'emerald');
      } catch (error) {
        setTestResults((current) => ({ ...current, [entry.id]: '' }));
        throw error;
      }
    });
  };

  const handleDelete = async (entry: ApiPoolEntry) => {
    if (confirmDelete !== entry.id) {
      setConfirmDelete(entry.id);
      return;
    }
    setConfirmDelete(null);
    await run(`delete-${entry.id}`, async () => {
      await deletePoolEntry(entry.id);
      report('Configuration deleted.', 'emerald');
    });
  };

  const handleApplySettings = async () => {
    await run('save-settings', async () => {
      await saveApiPoolSettings({
        strategy: settingsDraft.strategy,
        cache_enabled: settingsDraft.cache_enabled,
        cache_ttl_days: Number(settingsDraft.cache_ttl_days) || 7,
        max_attempts: Number(settingsDraft.max_attempts) || 3,
        max_parallel: Number(settingsDraft.max_parallel) || 4,
      });
      report('Pool settings applied.', 'emerald');
    });
  };

  const handleRetryQueue = async () => {
    await run('retry-queue', async () => {
      const result = await retryPoolQueue();
      report(
        result.ok + result.failed === 0
          ? `Nothing to resend (${result.requeued} queued request(s) could not be restored).`
          : `Resent ${result.ok + result.failed} request(s): ${result.ok} answered, ${result.failed} still failing.`,
        result.failed ? 'amber' : 'emerald',
      );
    });
  };

  const handleClearCache = async () => {
    await run('clear-cache', async () => {
      const removed = await clearPoolCache();
      report(`Cache cleared (${removed} line(s)).`, 'emerald');
    });
  };

  const handleImport = async () => {
    await run('import', async () => {
      const created = await importLegacyPool();
      report(created ? `${created} configuration(s) imported from API Settings.` : 'Nothing to import — no saved key was found.', created ? 'emerald' : 'slate');
    });
  };

  const handleResetHealth = async (entryId?: string) => {
    await run(entryId ? `reset-${entryId}` : 'reset-all', async () => {
      await resetPoolHealth(entryId);
      report(entryId ? 'Cooldown cleared for this configuration.' : 'All cooldowns cleared.', 'emerald');
    });
  };

  const settings = apiPool?.settings;
  const totals = apiPool?.usage.totals.today;
  const daily = apiPool?.usage.daily ?? [];
  const maxDaily = Math.max(1, ...daily.map((row) => row.requests));
  const entries = apiPool?.entries ?? [];
  const cooling = entries.filter((entry) => entry.health.cooldown_remaining > 0).length;

  return (
    <div className="space-y-4">
      {/* Scheduling + pool settings */}
      <section className="rounded-xl border border-white/10 bg-slate-900/60 p-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-sm font-medium text-slate-200">Scheduling</p>
            <p className="mt-0.5 text-[10px] leading-4 text-slate-500">
              {STRATEGY[settingsDraft.strategy].hint} A configuration that is cooling down, over its
              limit, or failing is skipped and the next one answers.
            </p>
          </div>
          <select
            value={settingsDraft.strategy}
            onChange={(event) => setSettingsDraft((current) => ({ ...current, strategy: event.target.value as ApiPoolStrategy }))}
            className="rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
          >
            {(Object.keys(STRATEGY) as ApiPoolStrategy[]).map((value) => (
              <option key={value} value={value}>{STRATEGY[value].label}</option>
            ))}
          </select>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <label className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <span className="block text-[9px] uppercase tracking-wide text-slate-500">Cache</span>
            <span className="mt-1 flex items-center gap-1.5 text-xs text-slate-200">
              <input
                type="checkbox"
                checked={settingsDraft.cache_enabled}
                onChange={(event) => setSettingsDraft((current) => ({ ...current, cache_enabled: event.target.checked }))}
              />
              Reuse validated lines
            </span>
          </label>
          <label className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <span className="block text-[9px] uppercase tracking-wide text-slate-500">Cache TTL (days)</span>
            <input
              type="number"
              min={1}
              max={365}
              value={settingsDraft.cache_ttl_days}
              onChange={(event) => setSettingsDraft((current) => ({ ...current, cache_ttl_days: event.target.value }))}
              className="mt-1 w-full bg-transparent text-xs text-slate-200 outline-none"
            />
          </label>
          <label className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <span className="block text-[9px] uppercase tracking-wide text-slate-500">Retries per config</span>
            <input
              type="number"
              min={1}
              max={6}
              value={settingsDraft.max_attempts}
              onChange={(event) => setSettingsDraft((current) => ({ ...current, max_attempts: event.target.value }))}
              className="mt-1 w-full bg-transparent text-xs text-slate-200 outline-none"
            />
          </label>
          <label className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <span className="block text-[9px] uppercase tracking-wide text-slate-500">Parallel lines (max)</span>
            <input
              type="number"
              min={1}
              max={8}
              value={settingsDraft.max_parallel}
              onChange={(event) => setSettingsDraft((current) => ({ ...current, max_parallel: event.target.value }))}
              className="mt-1 w-full bg-transparent text-xs text-slate-200 outline-none"
            />
          </label>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button
            onClick={() => void handleApplySettings()}
            disabled={busy !== null}
            className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-40"
          >
            {busy === 'save-settings' ? 'Applying…' : 'Apply settings'}
          </button>
          <button
            onClick={() => void handleImport()}
            disabled={busy !== null}
            className="rounded-lg border border-white/10 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
          >
            {busy === 'import' ? 'Importing…' : 'Import keys from API Settings'}
          </button>
          <button
            onClick={() => void refreshApiPool()}
            disabled={busy !== null}
            className="rounded-lg border border-white/10 bg-slate-900 px-2 py-1.5 text-slate-400 transition hover:bg-slate-800 disabled:opacity-40"
            title="Refresh the pool status"
          >
            <RefreshCw size={13} />
          </button>
          {cooling > 0 && (
            <button
              onClick={() => void handleResetHealth()}
              disabled={busy !== null}
              className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-200 transition hover:bg-amber-500/20 disabled:opacity-40"
            >
              Clear {cooling} cooldown{cooling === 1 ? '' : 's'}
            </button>
          )}
        </div>
      </section>

      {/* Provider groups */}
      {groups.map(({ provider, info, entries: providerEntries }) => (
        <section key={provider} className="rounded-xl border border-white/10 bg-slate-900/40 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <p className="text-sm font-medium text-slate-200">{PROVIDER_META[provider].label} pool</p>
              <Chip tone={info && info.usable > 0 ? 'emerald' : 'slate'}>
                {info?.usable ?? 0} of {info?.entries ?? 0} ready
              </Chip>
              {info?.has_legacy_credential && <Chip tone="sky">API Settings key</Chip>}
            </div>
            <button
              onClick={() => { setDraftError(''); setDraft(emptyDraft(provider)); }}
              className="flex items-center gap-1 rounded-lg border border-white/10 bg-slate-800 px-2 py-1 text-[11px] font-medium text-slate-200 transition hover:bg-slate-700"
            >
              <Plus size={12} /> Add
            </button>
          </div>

          {providerEntries.length === 0 && (
            <p className="mt-2 text-[10px] leading-4 text-slate-500">
              No configuration yet.{' '}
              {provider === 'custom'
                ? 'Add an OpenAI-compatible endpoint to rotate across several servers.'
                : info?.has_legacy_credential
                  ? 'Translation keeps using the single key saved in API Settings until you add one.'
                  : 'Translation for this provider stays unavailable until a key or configuration exists.'}
            </p>
          )}

          <div className="mt-2 space-y-2">
            {providerEntries.map((entry) => {
              const cooldown = Math.round(entry.health.cooldown_remaining);
              const badgeTone: Tone = !entry.enabled
                ? 'slate'
                : entry.usable
                  ? 'emerald'
                  : entry.health.state === 'cooling'
                    ? 'amber'
                    : 'rose';
              const badgeText = !entry.enabled
                ? 'disabled'
                : entry.usable
                  ? 'ready'
                  : entry.unavailable_reason || 'unavailable';
              return (
                <div key={entry.id} className="rounded-xl border border-white/10 bg-slate-950/60 p-2.5">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <p className="truncate text-xs font-medium text-slate-100">{entry.display_name}</p>
                        <Chip tone={badgeTone}>{badgeText}</Chip>
                        {entry.key_source === 'legacy' && <Chip tone="sky">API Settings key</Chip>}
                        {entry.health.state === 'degraded' && <Chip tone="amber">degraded</Chip>}
                      </div>
                      <p className="mt-1 font-mono text-[9px] leading-4 text-slate-500">
                        {entry.base_url || 'provider default endpoint'} · {entry.has_key ? `key ${entry.key_hint}` : 'no key'}
                        {' · '}weight {entry.weight}
                      </p>
                      <p className="mt-1 text-[10px] leading-4 text-slate-400">
                        {formatCount(entry.usage.requests_today)} request(s) today
                        {entry.usage.daily_limit === null ? ' · no daily limit set' : ` / ${formatCount(entry.usage.daily_limit)}`}
                        {' · '}{formatCount(entry.usage.requests_month)} this month
                        {entry.usage.monthly_limit === null ? '' : ` / ${formatCount(entry.usage.monthly_limit)}`}
                        {entry.health.last_latency_ms > 0 ? ` · last answer ${entry.health.last_latency_ms} ms` : ''}
                      </p>
                      {cooldown > 0 && (
                        <p className="mt-1 text-[10px] text-amber-300">
                          Cooling down {cooldown}s{entry.health.cooldown_reason ? ` — ${entry.health.cooldown_reason}` : ''}
                        </p>
                      )}
                      {entry.health.last_error && entry.health.state !== 'healthy' && cooldown === 0 && (
                        <p className="mt-1 break-words rounded border border-rose-500/20 bg-rose-500/5 px-2 py-1 text-[10px] leading-4 text-rose-300">
                          {entry.health.last_error}
                        </p>
                      )}
                      {testResults[entry.id] && (
                        <p className="mt-1 break-words rounded border border-emerald-500/20 bg-emerald-500/5 px-2 py-1 text-[10px] leading-4 text-emerald-200">
                          {testResults[entry.id]}
                        </p>
                      )}
                    </div>
                    <div className="flex shrink-0 flex-wrap items-center gap-1">
                      <button
                        onClick={() => void handleTest(entry)}
                        disabled={busy !== null}
                        className="rounded-md border border-sky-500/30 bg-sky-500/10 px-2 py-1 text-[10px] font-medium text-sky-200 transition hover:bg-sky-500/20 disabled:opacity-40"
                        title="Send one tiny real request"
                      >
                        {busy === `test-${entry.id}` ? 'Testing…' : 'Test'}
                      </button>
                      <button
                        onClick={() => { setDraftError(''); setDraft(draftFromEntry(entry)); }}
                        className="rounded-md border border-white/10 bg-slate-800 px-2 py-1 text-[10px] font-medium text-slate-200 transition hover:bg-slate-700"
                        title="Edit this configuration"
                      >
                        <Pencil size={11} />
                      </button>
                      {cooldown > 0 && (
                        <button
                          onClick={() => void handleResetHealth(entry.id)}
                          disabled={busy !== null}
                          className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-200 transition hover:bg-amber-500/20 disabled:opacity-40"
                          title="Clear this cooldown"
                        >
                          <RotateCcw size={11} />
                        </button>
                      )}
                      <button
                        onClick={() => void handleDelete(entry)}
                        disabled={busy !== null}
                        className={`rounded-md border px-2 py-1 text-[10px] font-medium transition disabled:opacity-40 ${
                          confirmDelete === entry.id
                            ? 'border-rose-400 bg-rose-500 text-white'
                            : 'border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20'
                        }`}
                        title={confirmDelete === entry.id ? 'Click again to delete' : 'Delete this configuration'}
                      >
                        {confirmDelete === entry.id ? 'Confirm' : <Trash2 size={11} />}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {draft?.provider === provider && (
            <div className="mt-2 rounded-xl border border-sky-500/30 bg-sky-500/5 p-3">
              <p className="text-xs font-semibold text-sky-100">
                {draft.id ? 'Edit configuration' : `New ${PROVIDER_META[provider].label} configuration`}
              </p>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <label className="block">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Label</span>
                  <input
                    value={draft.label}
                    onChange={(event) => setDraft({ ...draft, label: event.target.value })}
                    placeholder="e.g. personal key"
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Model</span>
                  <input
                    value={draft.model}
                    onChange={(event) => setDraft({ ...draft, model: event.target.value })}
                    placeholder={PROVIDER_META[provider].modelPlaceholder}
                    spellCheck={false}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
                  />
                </label>
                <label className="block sm:col-span-2">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Base URL</span>
                  <input
                    value={draft.base_url}
                    onChange={(event) => setDraft({ ...draft, base_url: event.target.value })}
                    placeholder={PROVIDER_META[provider].baseUrlHint}
                    spellCheck={false}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
                  />
                </label>
                <label className="block sm:col-span-2">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">
                    API key {draft.id ? '(leave empty to keep the stored key)' : ''}
                  </span>
                  <input
                    type="password"
                    value={draft.key}
                    onChange={(event) => setDraft({ ...draft, key: event.target.value })}
                    placeholder={draft.id ? '•••• (unchanged)' : provider === 'custom' ? 'Optional for a local server' : 'Paste the key for this configuration'}
                    spellCheck={false}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 font-mono text-xs text-slate-200"
                  />
                  <span className="mt-1 block text-[9px] leading-3 text-slate-500">
                    Stored in Windows Credential Manager — never returned to the browser.
                  </span>
                </label>
                <label className="block">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Daily limit (requests)</span>
                  <input
                    type="number"
                    min={0}
                    value={draft.daily_limit}
                    onChange={(event) => setDraft({ ...draft, daily_limit: event.target.value })}
                    placeholder="No limit"
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Monthly limit (requests)</span>
                  <input
                    type="number"
                    min={0}
                    value={draft.monthly_limit}
                    onChange={(event) => setDraft({ ...draft, monthly_limit: event.target.value })}
                    placeholder="No limit"
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                  />
                </label>
                <label className="block">
                  <span className="block text-[9px] uppercase tracking-wide text-slate-500">Weight (1–10)</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={draft.weight}
                    onChange={(event) => setDraft({ ...draft, weight: event.target.value })}
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                  />
                </label>
                <label className="flex items-end gap-2 pb-1">
                  <input
                    type="checkbox"
                    checked={draft.enabled}
                    onChange={(event) => setDraft({ ...draft, enabled: event.target.checked })}
                  />
                  <span className="text-xs text-slate-300">Enabled in the pool</span>
                </label>
              </div>
              <p className="mt-2 text-[9px] leading-4 text-slate-500">
                Limits are yours to set: the pool only records what the provider reports, so leaving
                them empty means “no limit known”, not “unlimited by the provider”.
              </p>
              {draftError && (
                <p className="mt-1 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{draftError}</p>
              )}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => void handleSave()}
                  disabled={busy !== null}
                  className="flex items-center gap-1 rounded-lg bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-sky-400 disabled:opacity-40"
                >
                  <Check size={12} /> {busy === 'save-entry' ? 'Saving…' : 'Save configuration'}
                </button>
                <button
                  onClick={() => { setDraft(null); setDraftError(''); }}
                  className="rounded-lg border border-white/10 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 transition hover:bg-slate-800"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </section>
      ))}

      {/* Measured usage */}
      <section className="rounded-xl border border-white/10 bg-slate-900/40 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              <Database size={13} /> Measured usage
            </p>
            <p className="mt-0.5 text-[10px] leading-4 text-slate-500">
              Counted by this app. Token numbers are what the provider reported; when it reports
              none, only the characters sent are known.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-400">
            <span>{formatCount(totals?.requests ?? 0)} today</span>
            <span>·</span>
            <span>{formatCount(totals?.failures ?? 0)} failed</span>
            <span>·</span>
            <span>{formatCount((totals?.tokens_in ?? 0) + (totals?.tokens_out ?? 0))} tokens</span>
          </div>
        </div>
        <div className="mt-3 flex items-end gap-1">
          {daily.map((row) => (
            <div key={row.day} className="flex-1 text-center" title={`${row.day}: ${row.requests} request(s), ${row.failures} failed`}>
              <div
                className={`mx-auto w-full rounded-t ${row.failures > 0 ? 'bg-amber-500/60' : 'bg-sky-500/60'}`}
                style={{ height: `${Math.max(2, Math.round((row.requests / maxDaily) * 40))}px` }}
              />
              <span className="mt-1 block text-[8px] text-slate-600">{row.day.slice(5)}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Request journal + cache */}
      <section className="rounded-xl border border-white/10 bg-slate-900/40 p-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <p className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
              <Zap size={13} /> Request queue &amp; cache
            </p>
            <p className="mt-0.5 text-[10px] leading-4 text-slate-500">
              Every pooled request is journalled before it is sent, so an interrupted run leaves a
              retryable entry instead of a silently lost line.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => void handleRetryQueue()}
              disabled={busy !== null || (apiPool?.queue.failed ?? 0) === 0}
              className="rounded-lg border border-white/10 bg-slate-800 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-40"
            >
              {busy === 'retry-queue' ? 'Resending…' : `Retry failed (${apiPool?.queue.failed ?? 0})`}
            </button>
            <button
              onClick={() => void handleClearCache()}
              disabled={busy !== null || (apiPool?.cache.entries ?? 0) === 0}
              className="rounded-lg border border-white/10 bg-slate-800 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-40"
            >
              {busy === 'clear-cache' ? 'Clearing…' : 'Clear cache'}
            </button>
          </div>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <div className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <p className="text-[9px] uppercase tracking-wide text-slate-500">Queued</p>
            <p className="text-sm text-slate-200">{apiPool?.queue.pending ?? 0}</p>
          </div>
          <div className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <p className="text-[9px] uppercase tracking-wide text-slate-500">In flight</p>
            <p className="text-sm text-slate-200">{apiPool?.queue.in_flight ?? 0}</p>
          </div>
          <div className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <p className="text-[9px] uppercase tracking-wide text-slate-500">Answered / 24 h</p>
            <p className="text-sm text-slate-200">
              {apiPool?.queue.last_24h.ok ?? 0}
              <span className="text-[10px] text-slate-500"> / {apiPool?.queue.last_24h.total ?? 0}</span>
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
            <p className="text-[9px] uppercase tracking-wide text-slate-500">Cache lines / hits</p>
            <p className="text-sm text-slate-200">
              {apiPool?.cache.entries ?? 0}
              <span className="text-[10px] text-slate-500"> / {apiPool?.cache.hits ?? 0}</span>
            </p>
          </div>
        </div>
        {(apiPool?.queue.failed_items ?? []).slice(0, 3).map((item) => (
          <p key={item.id} className="mt-1.5 break-words rounded border border-rose-500/20 bg-rose-500/5 px-2 py-1 text-[10px] leading-4 text-rose-300">
            #{item.id} {item.provider}{item.model ? ` · ${item.model}` : ''} — {item.error ?? 'failed'}
          </p>
        ))}
        {settings && (
          <p className="mt-2 text-[9px] leading-4 text-slate-600">
            Strategy {STRATEGY[settings.strategy].label.toLowerCase()} · cache {settings.cache_enabled ? `on, ${settings.cache_ttl_days}-day TTL` : 'off'} ·
            {' '}{settings.max_attempts} attempt(s) per configuration · {settings.request_timeout}s request timeout.
          </p>
        )}
      </section>

      {message && (
        <p className={`rounded-lg border px-3 py-2 text-xs ${TONE_CLASS[messageTone]}`}>{message}</p>
      )}
    </div>
  );
}
