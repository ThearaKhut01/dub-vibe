import { useEffect, useState } from 'react';
import { Check, Chrome, History, LogIn, RefreshCw, UserRoundPlus } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import type { ChromeProfile, ChromeProfileValidation } from '../types';

/**
 * Chrome Profiles — which Chrome profile this tool may drive for Gemini web.
 *
 * The list is metadata only: Chrome's `Local State` plus each profile's
 * `Preferences`. No cookie, password, token or `Login Data` file is read, and
 * nothing leaves this machine. Validating opens the profile on
 * gemini.google.com and reports what it finds; when Gemini wants a sign-in the
 * panel says so instead of trying to log in.
 */

type Tone = 'emerald' | 'amber' | 'rose' | 'slate' | 'sky';

const TONE_CLASS: Record<Tone, string> = {
  emerald: 'border-emerald-500/25 bg-emerald-500/15 text-emerald-200',
  amber: 'border-amber-500/25 bg-amber-500/15 text-amber-200',
  rose: 'border-rose-500/25 bg-rose-500/15 text-rose-200',
  sky: 'border-sky-500/25 bg-sky-500/15 text-sky-200',
  slate: 'border-white/10 bg-slate-700/40 text-slate-300',
};

type StatusKey = 'ready' | 'login_required' | 'locked' | 'unknown';

const STATUS_META: Record<StatusKey, { label: string; tone: Tone }> = {
  ready: { label: 'Ready', tone: 'emerald' },
  login_required: { label: 'Login required', tone: 'amber' },
  locked: { label: 'Window open', tone: 'rose' },
  unknown: { label: 'Unknown', tone: 'slate' },
};

function lockedText(folder: string): string {
  return `The tool's own Chrome window for ${folder} is open. Close that window, then try again.`;
}

/** `example@gmail.com` -> `ex****@gmail.com`: enough to recognise, not to leak. */
export function maskEmail(email: string): string {
  const at = email.indexOf('@');
  if (at < 1) return email;
  return `${email.slice(0, Math.min(2, at))}****${email.slice(at)}`;
}

function profileLabel(profile: ChromeProfile | null): string {
  if (!profile) return '';
  const email = profile.emails[0];
  return `${profile.profile_folder}${email ? ` / ${maskEmail(email)}` : ''}`;
}

export default function ChromeProfilesPanel() {
  const {
    chromeProfiles,
    refreshChromeProfiles,
    refreshVoiceConfig,
    selectChromeProfile,
    validateChromeProfile,
    openChromeProfileManually,
    chooseChromeAccount,
    setChromeSharedWindow,
    loadChromeChats,
  } = useStudioStore();

  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [messageTone, setMessageTone] = useState<Tone>('slate');
  const [results, setResults] = useState<Record<string, ChromeProfileValidation>>({});
  const [chats, setChats] = useState<string[] | null>(null);

  useEffect(() => {
    void refreshChromeProfiles();
    // Opening this tab is also the moment the engine picker may become usable.
    void refreshVoiceConfig();
  }, [refreshChromeProfiles, refreshVoiceConfig]);

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

  const handleSelect = (profile: ChromeProfile) => run(`select-${profile.id}`, async () => {
    await selectChromeProfile(profile.id);
    report(`Selected ${profileLabel(profile)}.`, 'emerald');
  });

  const handleValidate = (profile: ChromeProfile) => run(`validate-${profile.id}`, async () => {
    const result = await validateChromeProfile(profile.id);
    setResults((current) => ({ ...current, [profile.id]: result }));
    report(result.message, result.status === 'ready' ? 'emerald' : result.status === 'locked' ? 'rose' : 'amber');
  });

  const handleSharedWindow = (enabled: boolean) => run('shared-window', async () => {
    await setChromeSharedWindow(enabled);
    report(
      enabled
        ? 'Shared window on — Validate and Translate keep one Chrome open and work in their own tab.'
        : 'Shared window off — the tool opens and closes its own window again.',
      enabled ? 'sky' : 'slate',
    );
  });

  const handleChats = () => run('chats', async () => {
    const titles = await loadChromeChats(15);
    setChats(titles);
    report(
      titles.length
        ? `${titles.length} chat(s) in this profile — newest first. Gemini titles them from the prompt.`
        : 'This profile has no chats yet.',
      titles.length ? 'slate' : 'amber',
    );
  });

  const handleChooseAccount = (profile: ChromeProfile) => run(`choose-${profile.id}`, async () => {
    const result = await chooseChromeAccount(profile.id);
    report(result.message, result.success ? 'sky' : 'rose');
  });

  const handleManual = (profile: ChromeProfile) => run(`manual-${profile.id}`, async () => {
    const result = await openChromeProfileManually(profile.id);
    report(result.message, result.success ? 'sky' : 'rose');
  });

  const profiles = chromeProfiles?.profiles ?? [];
  const selected = chromeProfiles?.selected ?? null;
  const withGoogle = profiles.filter((profile) => profile.has_google_account);

  return (
    <section className="rounded-xl border border-white/10 bg-slate-900/40 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-sm font-medium text-slate-200">
            <Chrome size={13} /> Chrome profiles (Gemini web)
          </p>
          <p className="mt-0.5 text-[10px] leading-4 text-slate-500">
            Reuses a profile you are already signed in to. Only Chrome&apos;s own metadata is read —
            no cookie, password or token file — and nothing is uploaded.
          </p>
        </div>
        <button
          onClick={() => void handleChats()}
          disabled={busy !== null || !selected}
          className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-900 px-2.5 py-1.5 text-[11px] font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
          title="Read this profile's Gemini chat list"
        >
          <History size={12} className={busy === 'chats' ? 'animate-spin' : ''} />
          {busy === 'chats' ? 'Reading…' : 'Recent chats'}
        </button>
        <button
          onClick={() => void run('refresh', refreshChromeProfiles)}
          disabled={busy !== null}
          className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-800 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-40"
          title="Rescan %LOCALAPPDATA%\\Google\\Chrome\\User Data"
        >
          <RefreshCw size={12} className={busy === 'refresh' ? 'animate-spin' : ''} />
          {busy === 'refresh' ? 'Scanning…' : 'Refresh Profiles'}
        </button>
      </div>

      <label className="mt-2 flex items-start gap-2 rounded-lg border border-white/10 bg-slate-950/60 px-2 py-1.5">
        <input
          type="checkbox"
          className="mt-0.5"
          checked={Boolean(chromeProfiles?.shared_window)}
          disabled={busy !== null}
          onChange={(event) => void handleSharedWindow(event.target.checked)}
        />
        <span className="min-w-0">
          <span className="text-[11px] text-slate-200">Shared window</span>
          <span className="mt-0.5 block text-[9px] leading-4 text-slate-500">
            Keep one Chrome window open: you browse in your tabs while the tool chats in its own tab. Closing the
            tool's tab mid-run stops that run; turning this off closes nothing.
          </span>
        </span>
      </label>

      {chats && (
        <div className="mt-2 rounded-lg border border-white/10 bg-slate-950/60 p-2">
          <p className="text-[9px] uppercase tracking-wide text-slate-500">Gemini chats in this profile</p>
          {chats.length === 0 ? (
            <p className="mt-1 text-[10px] text-slate-500">None yet — translate something first.</p>
          ) : (
            <ol className="mt-1 space-y-0.5">
              {chats.map((title, index) => (
                <li key={`${index}-${title}`} className="truncate text-[10px] text-slate-300">
                  {index + 1}. {title}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {selected && (
        <p className="mt-2 rounded-lg border border-sky-500/25 bg-sky-500/10 px-2 py-1.5 text-[11px] text-sky-100">
          Selected profile: <span className="font-medium">{profileLabel(selected)}</span>
          {selected.display_name && selected.display_name !== selected.profile_folder && (
            <span className="text-sky-300/80"> · {selected.display_name}</span>
          )}
          {chromeProfiles?.session_account && (
            <span className="block text-[9px] text-sky-300/70">
              Gemini account in the tool's browser: {maskEmail(chromeProfiles.session_account)} — chats belong to
              this Google account, not to your other Chrome window.
            </span>
          )}
        </p>
      )}
      {!selected && chromeProfiles?.selected_missing && (
        <p className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-200">
          The saved profile ({chromeProfiles.last_selected_profile_folder || 'unknown'}) is no longer on this PC.
          Pick a profile again.
        </p>
      )}

      {chromeProfiles && !chromeProfiles.chrome_path && (
        <p className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-200">
          Google Chrome was not found. Install Chrome, or set <span className="font-mono">DUBVIBE_CHROME_PATH</span> to
          chrome.exe.
        </p>
      )}
      {chromeProfiles && !chromeProfiles.playwright_ready && (
        <p className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-200">
          Playwright is not installed in the backend&apos;s Python environment, so Validate and Gemini web cannot
          open a browser. Run <span className="font-mono">pip install playwright</span> (the app drives your
          installed Chrome, so no browser download is needed).
        </p>
      )}

      {!chromeProfiles && (
        <p className="mt-2 text-[10px] text-slate-500">Reading Chrome&apos;s profile list…</p>
      )}

      {chromeProfiles && profiles.length === 0 && (
        <p className="mt-2 text-[10px] leading-4 text-slate-500">
          No Chrome profiles with Google account metadata found. Open Chrome and sign in manually first.
        </p>
      )}

      {chromeProfiles && profiles.length > 0 && withGoogle.length === 0 && (
        <p className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1.5 text-[11px] leading-4 text-amber-200">
          Chrome profiles were found, but none carries a Google account yet. Sign in inside Chrome (Open manual
          login), then refresh.
        </p>
      )}

      <div className="mt-2 space-y-2">
        {profiles.map((profile) => {
          const validation = results[profile.id];
          const statusKey: StatusKey = profile.tool_window_open
            ? 'locked'
            : validation?.status === 'ready'
              ? 'ready'
              : validation?.status === 'login_required'
                ? 'login_required'
                : 'unknown';
          const status = STATUS_META[statusKey];
          const isSelected = selected?.id === profile.id;
          const email = profile.emails[0];
          return (
            <div
              key={profile.id}
              className={`rounded-xl border p-2.5 ${isSelected ? 'border-sky-500/50 bg-sky-500/5 ring-1 ring-sky-500/30' : 'border-white/10 bg-slate-950/60'}`}
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <p className="truncate text-xs font-medium text-slate-100">{profile.display_name}</p>
                    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${TONE_CLASS[status.tone]}`}>
                      {status.label}
                    </span>
                    {isSelected && (
                      <span className="rounded-full border border-sky-500/25 bg-sky-500/15 px-2 py-0.5 text-[10px] font-semibold text-sky-200">
                        selected
                      </span>
                    )}
                    {profile.is_last_used && <span className="text-[9px] text-slate-500">last used</span>}
                    {!profile.has_google_account && (
                      <span className="rounded-full border border-white/10 bg-slate-700/40 px-2 py-0.5 text-[10px] text-slate-400">
                        no Google account
                      </span>
                    )}
                  </div>
                  <p className="mt-1 font-mono text-[9px] leading-4 text-slate-500">
                    {profile.profile_folder} · {profile.root_channel}
                    {' · '}
                    {email ? maskEmail(email) : 'no account e-mail in Chrome metadata'}
                  </p>
                  {selected?.id === profile.id && profile.session_ready === false && (
                    <p className="mt-1 text-[10px] leading-4 text-amber-200">
                      First run for this profile: press <span className="font-medium">Open manual login</span>, sign in
                      to Gemini in the Chrome window that opens, then Validate.
                    </p>
                  )}
                  {profile.is_locked && (
                    <p className="mt-1 text-[10px] leading-4 text-slate-500">
                      Chrome has this profile open. That is fine — the tool works on its own copy of it.
                    </p>
                  )}
                  {profile.tool_window_open && (
                    <p className="mt-1 text-[10px] leading-4 text-rose-300">
                      {lockedText(profile.profile_folder)}
                    </p>
                  )}
                  {validation && (
                    <p
                      className={`mt-1 break-words rounded border px-2 py-1 text-[10px] leading-4 ${
                        validation.status === 'ready'
                          ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-200'
                          : validation.status === 'locked'
                            ? 'border-rose-500/20 bg-rose-500/5 text-rose-300'
                            : 'border-amber-500/20 bg-amber-500/5 text-amber-200'
                      }`}
                    >
                      {validation.message}
                      {validation.via === 'session_profile' ? ' (checked in the tool\'s own Chrome profile)' : ''}
                    </p>
                  )}
                  {profile.warnings.length > 0 && (
                    <p className="mt-1 text-[9px] leading-4 text-slate-600">{profile.warnings.join(' · ')}</p>
                  )}
                </div>
                <div className="flex shrink-0 flex-wrap items-center gap-1">
                  <button
                    onClick={() => void handleSelect(profile)}
                    disabled={busy !== null || isSelected}
                    className="flex items-center gap-1 rounded-md border border-white/10 bg-slate-800 px-2 py-1 text-[10px] font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-40"
                    title="Use this profile for Gemini web automation"
                  >
                    <Check size={11} /> {busy === `select-${profile.id}` ? 'Saving…' : isSelected ? 'Selected' : 'Select'}
                  </button>
                  <button
                    onClick={() => void handleValidate(profile)}
                    disabled={busy !== null}
                    className="rounded-md border border-sky-500/30 bg-sky-500/10 px-2 py-1 text-[10px] font-medium text-sky-200 transition hover:bg-sky-500/20 disabled:opacity-40"
                    title="Open this profile on gemini.google.com and check whether it is signed in"
                  >
                    {busy === `validate-${profile.id}` ? 'Checking…' : 'Validate Gemini'}
                  </button>
                  <button
                    onClick={() => void handleChooseAccount(profile)}
                    disabled={busy !== null}
                    className="flex items-center gap-1 rounded-md border border-violet-500/30 bg-violet-500/10 px-2 py-1 text-[10px] font-medium text-violet-200 transition hover:bg-violet-500/20 disabled:opacity-40"
                    title="Open Google's account picker in the tool's own Chrome window"
                  >
                    <UserRoundPlus size={11} /> {busy === `choose-${profile.id}` ? 'Opening…' : 'Choose account'}
                  </button>
                  <button
                    onClick={() => void handleManual(profile)}
                    disabled={busy !== null}
                    className="flex items-center gap-1 rounded-md border border-white/10 bg-slate-900 px-2 py-1 text-[10px] font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
                    title="Open this profile in normal Chrome so you can sign in yourself"
                  >
                    <LogIn size={11} /> {busy === `manual-${profile.id}` ? 'Opening…' : 'Open manual login'}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {(chromeProfiles?.errors.length ?? 0) > 0 && (
        <div className="mt-2 space-y-1">
          {chromeProfiles?.errors.map((error) => (
            <p key={error} className="break-words rounded border border-amber-500/20 bg-amber-500/5 px-2 py-1 text-[10px] leading-4 text-amber-200">
              {error}
            </p>
          ))}
        </div>
      )}

      {message && (
        <p className={`mt-2 rounded-lg border px-3 py-2 text-[11px] ${TONE_CLASS[messageTone]}`}>{message}</p>
      )}

      <p className="mt-2 text-[9px] leading-4 text-slate-600">
        The app keeps its own Chrome profile per detection (<span className="font-mono">data\chrome-profile-sessions</span>):
        sign in once in that window and the session is reused for every check and prompt. It is a separate profile
        from your normal Chrome, so your browsing session is never touched — and this app never decrypts a cookie.
      </p>
    </section>
  );
}
