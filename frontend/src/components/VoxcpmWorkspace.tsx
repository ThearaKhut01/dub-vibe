import { useEffect, useRef, useState } from 'react';
import { Cpu, Download, Mic, RefreshCw, Trash2, Upload, Wand2, Play, Square, Send } from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';

const apiBaseUrl = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';

type VoxMode = 'tts' | 'design' | 'clone' | 'ultimate';

interface VoxJob {
  id: string;
  mode: VoxMode;
  status: 'running' | 'done' | 'cancelled' | 'error';
  progress: number;
  message: string;
  output: string | null;
  sample_rate: number | null;
  duration: number | null;
  error_code: string | null;
}

interface VoxHardware {
  device: 'cpu' | 'cuda';
  gpu: string | null;
  vram_gb: number | null;
  ram_total_gb: number;
  ram_available_gb: number;
  fits: boolean;
  reason: string | null;
  loaded: boolean;
}

const MODE_LABELS: Record<VoxMode, string> = {
  tts: 'Text to Speech',
  design: 'Voice Design',
  clone: 'Voice Cloning',
  ultimate: 'Ultimate Cloning',
};

const MODE_HELP: Record<VoxMode, string> = {
  tts: 'Speak the text in the default VoxCPM2 voice.',
  design: 'Describe the voice in plain English. The model creates a new voice from the description.',
  clone: 'Speak the text in the voice of one reference recording. Use only a voice you are authorized to use.',
  ultimate: 'Speak the text continuing from a reference recording. Requires the exact transcript of that recording.',
};

async function readError(response: Response, fallback: string): Promise<string> {
  const problem = await response.json().catch(() => ({}));
  return typeof problem.detail === 'string' ? problem.detail : `${fallback} (HTTP ${response.status})`;
}

export function VoxcpmWorkspace() {
  const voxcpmModel = useStudioStore((state) => state.voxcpmModel);
  const downloadVoxcpm = useStudioStore((state) => state.downloadVoxcpm);
  const refreshModels = useStudioStore((state) => state.refreshModels);
  const setMode = useStudioStore((state) => state.setMode);
  const projectList = useStudioStore((state) => state.projectList);

  const [mode, setVoxMode] = useState<VoxMode>('tts');
  const [targetProject, setTargetProject] = useState('');
  const [sending, setSending] = useState(false);
  const [sendNotice, setSendNotice] = useState<string | null>(null);
  const [text, setText] = useState('Hello, this is a VoxCPM voice generation test.');
  const [description, setDescription] = useState('a warm, calm female narrator');
  const [promptText, setPromptText] = useState('');
  const [reference, setReference] = useState<{ id: string; name: string } | null>(null);
  const [cfgValue, setCfgValue] = useState(2.0);
  const [timesteps, setTimesteps] = useState(10);
  const [speed, setSpeed] = useState(1.0);
  const [job, setJob] = useState<VoxJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hardware, setHardware] = useState<VoxHardware | null>(null);
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void refreshModels();
    fetch(`${apiBaseUrl}/api/voxcpm/hardware`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => setHardware(data))
      .catch(() => setHardware(null));
  }, [refreshModels]);

  // Poll the running job until it finishes.
  useEffect(() => {
    if (!job || job.status !== 'running') return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/voxcpm/jobs/${job.id}`);
        if (!response.ok) throw new Error(await readError(response, 'Unable to read job status'));
        const data = await response.json();
        setJob(data.job as VoxJob);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Lost contact with the job.');
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [job]);

  // Load the finished WAV for in-app preview. The previous blob is revoked only after
  // the new one is ready and has replaced it, so the <audio> element never reads a
  // revoked URL (that caused ERR_ABORTED on every new generation).
  const finishedJobId = job?.status === 'done' ? job.id : null;
  // The blob currently shown by the player, and the one it replaced. A replaced blob is
  // revoked only once the player has finished loading the new source (onLoadedData).
  const shownUrlRef = useRef<string | null>(null);
  const retiredUrlRef = useRef<string | null>(null);
  useEffect(() => {
    if (!finishedJobId) return;
    let cancelled = false;
    fetch(`${apiBaseUrl}/api/voxcpm/jobs/${finishedJobId}/audio`)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load the generated audio.');
        return response.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const next = URL.createObjectURL(blob);
        if (retiredUrlRef.current) URL.revokeObjectURL(retiredUrlRef.current);
        retiredUrlRef.current = shownUrlRef.current;
        shownUrlRef.current = next;
        setPreviewUrl(next);
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'Could not load audio.');
      });
    return () => { cancelled = true; };
  }, [finishedJobId]);

  const handleAudioLoaded = () => {
    // The player now reads the new clip, so the replaced one can be released safely.
    if (retiredUrlRef.current) {
      URL.revokeObjectURL(retiredUrlRef.current);
      retiredUrlRef.current = null;
    }
  };

  // Release the preview blobs when the workspace unmounts (mode switch away).
  useEffect(() => () => {
    if (shownUrlRef.current) URL.revokeObjectURL(shownUrlRef.current);
    if (retiredUrlRef.current) URL.revokeObjectURL(retiredUrlRef.current);
  }, []);

  const installed = Boolean(voxcpmModel?.installed);
  const running = job?.status === 'running';

  const handleUpload = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const response = await fetch(`${apiBaseUrl}/api/voxcpm/references`, { method: 'POST', body: form });
      if (!response.ok) throw new Error(await readError(response, 'Upload failed'));
      const data = await response.json();
      setReference({ id: data.reference_id, name: file.name });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Upload failed.');
    }
  };

  const handleGenerate = async () => {
    setError(null);
    setBusy(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/voxcpm/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          text,
          description,
          reference_id: reference?.id ?? null,
          prompt_text: promptText,
          cfg_value: cfgValue,
          inference_timesteps: timesteps,
          speed,
        }),
      });
      if (!response.ok) throw new Error(await readError(response, 'Unable to start generation'));
      const data = await response.json();
      setJob(data.job as VoxJob);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to start generation.');
    } finally {
      setBusy(false);
    }
  };

  const handleCancel = async () => {
    if (!job) return;
    try {
      const response = await fetch(`${apiBaseUrl}/api/voxcpm/jobs/${job.id}/cancel`, { method: 'POST' });
      if (!response.ok) throw new Error(await readError(response, 'Unable to cancel'));
      const data = await response.json();
      setJob(data.job as VoxJob);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to cancel.');
    }
  };

  const handleSendToProject = async () => {
    if (!job || !targetProject) return;
    setSending(true);
    setSendNotice(null);
    try {
      const response = await fetch(`${apiBaseUrl}/api/voxcpm/jobs/${job.id}/send-to-project/${targetProject}`, { method: 'POST' });
      if (!response.ok) throw new Error(await readError(response, 'Could not send the clip'));
      const data = await response.json();
      const name = projectList.find((item) => item.id === targetProject)?.name ?? 'the project';
      setSendNotice(`Saved as ${data.file} in ${name}.`);
    } catch (caught) {
      setSendNotice(caught instanceof Error ? caught.message : 'Could not send the clip.');
    } finally {
      setSending(false);
    }
  };

  const handleUnload = async () => {
    try {
      await fetch(`${apiBaseUrl}/api/voxcpm/unload`, { method: 'POST' });
      setHardware((current) => (current ? { ...current, loaded: false } : current));
    } catch {
      setError('Unable to release the model.');
    }
  };

  const canGenerate = installed
    && !running
    && !busy
    && text.trim().length > 0
    && (mode !== 'design' || description.trim().length > 0)
    && ((mode !== 'clone' && mode !== 'ultimate') || reference !== null)
    && (mode !== 'ultimate' || promptText.trim().length > 0)
    && hardware?.fits !== false;

  return (
    <div className="min-h-[calc(100vh-64px)] bg-[#050b14] px-6 py-6 text-slate-100">
      <div className="mx-auto grid max-w-5xl gap-6">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-violet-300">VoxCPM Mode</p>
          <h2 className="text-xl font-semibold">Standalone AI voice generation</h2>
          <p className="mt-1 text-sm text-slate-400">No video needed. Generated audio can be previewed and exported as WAV.</p>
        </div>

        {!installed && (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
            <p className="text-sm font-medium text-amber-100">VoxCPM2 is not installed</p>
            <p className="mt-1 text-xs text-amber-200/80">
              {voxcpmModel?.verify_error ?? 'Download the model to generate speech.'}
            </p>
            <button
              onClick={() => void downloadVoxcpm()}
              disabled={voxcpmModel?.status === 'downloading'}
              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/20 px-3 py-1.5 text-xs font-medium text-amber-100 hover:bg-amber-500/30 disabled:opacity-40"
            >
              <Download size={14} />
              {voxcpmModel?.status === 'downloading' ? `Downloading ${voxcpmModel.progress}%` : 'Download VoxCPM2 (~4.7 GB)'}
            </button>
          </div>
        )}

        {hardware && (
          <div className={`rounded-xl border p-3 text-xs ${hardware.fits ? 'border-white/10 bg-slate-900/70 text-slate-300' : 'border-rose-500/30 bg-rose-500/10 text-rose-200'}`}>
            <span className="inline-flex items-center gap-2">
              <Cpu size={13} />
              {hardware.device === 'cuda' ? `GPU: ${hardware.gpu} (${hardware.vram_gb} GB VRAM)` : 'Running on CPU (slower; an NVIDIA GPU with CUDA torch is much faster)'}
              {' · '}RAM {hardware.ram_available_gb} GB free of {hardware.ram_total_gb} GB
              {hardware.loaded ? ' · model loaded' : ''}
            </span>
            {hardware.reason && <p className="mt-1">{hardware.reason}</p>}
          </div>
        )}

        <div className="grid gap-6 md:grid-cols-[1fr_300px]">
          <section className="grid gap-4 rounded-2xl border border-white/10 bg-slate-900/70 p-5">
            <div className="flex flex-wrap gap-2">
              {(Object.keys(MODE_LABELS) as VoxMode[]).map((key) => (
                <button
                  key={key}
                  onClick={() => setVoxMode(key)}
                  aria-pressed={mode === key}
                  className={`rounded-lg px-3 py-2 text-sm font-medium transition ${mode === key ? 'bg-violet-500 text-slate-950' : 'bg-slate-950 text-slate-400 hover:text-slate-200'}`}
                >
                  {MODE_LABELS[key]}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-400">{MODE_HELP[mode]}</p>

            {mode === 'design' && (
              <label className="grid gap-1 text-sm">
                <span className="text-slate-300">Voice description</span>
                <input
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="for example: a warm, calm female narrator, slightly low pitch"
                  className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                />
              </label>
            )}

            {(mode === 'clone' || mode === 'ultimate') && (
              <div className="grid gap-2 rounded-xl border border-white/10 bg-slate-950/60 p-3">
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={() => fileRef.current?.click()}
                    className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm text-violet-200 hover:bg-violet-500/20"
                  >
                    <Upload size={14} /> {reference ? 'Replace reference' : 'Upload reference recording'}
                  </button>
                  <span className="text-xs text-slate-400">{reference ? reference.name : 'WAV, MP3, FLAC, OGG, or M4A, up to 25 MB'}</span>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
                    className="hidden"
                    onChange={(event) => { void handleUpload(event.target.files?.[0]); event.currentTarget.value = ''; }}
                  />
                </div>
                <p className="text-[11px] text-amber-200/80">Only upload a voice you have permission to clone.</p>
                {mode === 'ultimate' && (
                  <label className="grid gap-1 text-sm">
                    <span className="text-slate-300">Exact transcript of the reference recording</span>
                    <textarea
                      value={promptText}
                      onChange={(event) => setPromptText(event.target.value)}
                      rows={2}
                      className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    />
                  </label>
                )}
              </div>
            )}

            <label className="grid gap-1 text-sm">
              <span className="text-slate-300">Text to speak</span>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={5}
                maxLength={1000}
                className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
              />
              <span className="text-right text-[11px] text-slate-500">{text.length} / 1000</span>
            </label>

            <div className="grid gap-3 rounded-xl border border-white/10 bg-slate-950/60 p-3 sm:grid-cols-3">
              <label className="grid gap-1 text-xs text-slate-300">
                Guidance (CFG): {cfgValue.toFixed(1)}
                <input type="range" min={1} max={3} step={0.1} value={cfgValue} onChange={(event) => setCfgValue(Number(event.target.value))} />
              </label>
              <label className="grid gap-1 text-xs text-slate-300">
                Quality steps: {timesteps}
                <input type="range" min={4} max={30} step={1} value={timesteps} onChange={(event) => setTimesteps(Number(event.target.value))} />
              </label>
              <label className="grid gap-1 text-xs text-slate-300">
                Pace: {speed.toFixed(2)}x
                <input type="range" min={0.5} max={1.5} step={0.05} value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
              </label>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => void handleGenerate()}
                disabled={!canGenerate}
                className="inline-flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Wand2 size={15} /> {busy ? 'Starting...' : 'Generate speech'}
              </button>
              {running && (
                <button
                  onClick={() => void handleCancel()}
                  className="inline-flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200 hover:bg-rose-500/20"
                >
                  <Square size={13} /> Cancel
                </button>
              )}
              {!installed && <span className="text-xs text-slate-500">Download VoxCPM2 to enable generation.</span>}
            </div>

            {job && (
              <div className="grid gap-2 rounded-xl border border-white/10 bg-slate-950/60 p-3">
                <div className="flex items-center justify-between text-xs text-slate-300">
                  <span>{job.status === 'running' ? 'Generating' : job.status === 'done' ? 'Finished' : job.status === 'cancelled' ? 'Cancelled' : 'Failed'}</span>
                  <span>{job.progress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div className={`h-full rounded-full ${job.status === 'error' ? 'bg-rose-500' : 'bg-gradient-to-r from-violet-500 to-sky-500'}`} style={{ width: `${job.progress}%` }} />
                </div>
                <p className="text-xs text-slate-400">{job.message}</p>
              </div>
            )}

            {error && (
              <p className="rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{error}</p>
            )}
            {job?.status === 'error' && job.error_code === 'insufficient_memory' && (
              <p className="text-xs text-amber-200">Free up memory or close other applications, then try again.</p>
            )}
          </section>

          <aside className="grid content-start gap-4 rounded-2xl border border-white/10 bg-slate-900/70 p-5">
            <div>
              <p className="text-sm font-semibold text-slate-200">Preview and export</p>
              <p className="mt-1 text-xs text-slate-500">Finished audio is a 16-bit WAV at the model&apos;s native sample rate.</p>
            </div>
            {previewUrl && job?.status === 'done' ? (
              <>
                <audio controls src={previewUrl} onLoadedData={handleAudioLoaded} className="w-full" />
                <p className="text-xs text-slate-400">
                  {job.duration} s · {job.sample_rate} Hz · mono 16-bit WAV
                </p>
                <a
                  href={`${apiBaseUrl}/api/voxcpm/jobs/${job.id}/audio`}
                  download={`voxcpm-${job.id}.wav`}
                  className="inline-flex items-center justify-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-200 hover:bg-emerald-500/20"
                >
                  <Download size={14} /> Export WAV
                </a>
                <div className="grid gap-2 border-t border-white/10 pt-3">
                  <p className="text-[11px] text-slate-500">Send to a Dubbing project as a standalone clip (does not change its dub timeline).</p>
                  <select
                    value={targetProject}
                    onChange={(event) => setTargetProject(event.target.value)}
                    className="rounded-lg border border-white/10 bg-slate-950 px-2 py-2 text-xs text-slate-200"
                  >
                    <option value="">Choose a project...</option>
                    {projectList.map((item) => (
                      <option key={item.id} value={item.id}>{item.name}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => void handleSendToProject()}
                    disabled={!targetProject || sending}
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-200 hover:bg-sky-500/20 disabled:opacity-40"
                  >
                    <Send size={13} /> {sending ? 'Sending...' : 'Send to Dubbing project'}
                  </button>
                  {sendNotice && <p className="text-[11px] text-emerald-200">{sendNotice}</p>}
                </div>
              </>
            ) : (
              <p className="flex items-center gap-2 text-xs text-slate-500"><Play size={12} /> Generate speech to preview it here.</p>
            )}
            <div className="grid gap-2 border-t border-white/10 pt-4">
              <button
                onClick={() => void handleUnload()}
                disabled={!hardware?.loaded}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-40"
              >
                <Trash2 size={13} /> Release model from memory
              </button>
              <button
                onClick={() => void refreshModels()}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800"
              >
                <RefreshCw size={13} /> Refresh status
              </button>
              <p className="flex items-center gap-2 text-[11px] text-slate-500"><Mic size={11} /> Model stays loaded only while in use and unloads after 10 minutes idle.</p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
