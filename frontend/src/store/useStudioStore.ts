import { create } from 'zustand';
import type {
  CudaPackInfo,
  HardwareInfo,
  MediaAsset,
  ModelStorageInfo,
  ProjectSummary,
  ProviderStatus,
  TimelineSegment,
  WhisperModelInfo,
  WorkflowStep,
} from '../types';

interface LogEntry {
  id: number;
  time: string;
  message: string;
  level: 'info' | 'success' | 'error';
}

interface StudioState {
  project: ProjectSummary;
  projectList: ProjectSummary[];
  providers: ProviderStatus[];
  transcript: TimelineSegment[];
  dubAudioSegments: TimelineSegment[];
  videoChunks: Array<{ index: number; start: number; duration: number }>;
  chunkSeconds: number | null;
  dubbedVideoReady: boolean;
  mediaVersion: number;
  exportedParts: Array<{ index: number; name: string }>;
  defaultExportFolder: string | null;
  exportFolder: string | null;
  separated: boolean;
  processingLog: LogEntry[];
  models: WhisperModelInfo[];
  hardware: HardwareInfo | null;
  modelStorage: ModelStorageInfo | null;
  qwenModel: WhisperModelInfo | null;
  nllbModel: WhisperModelInfo | null;
  activeTranslateEngine: string | null;
  demucsModel: WhisperModelInfo | null;
  mmsKhmModels: WhisperModelInfo[];
  cudaPack: CudaPackInfo | null;
  providerKeys: { gemini: { has_key: boolean }; openai: { has_key: boolean } };
  voiceCapabilities: Record<string, { voices: Array<{ id: string; style: string }>; emotions: string[]; style_control: boolean; khmer_supported: boolean; languages: string[]; default_model: string; label: string; installed?: boolean }> & { emotions?: string[] };
  characters: { speakers: string[]; characters: Record<string, Record<string, unknown>> };
  workflowSteps: WorkflowStep[];
  mediaLibrary: MediaAsset[];
  activeTool: string;
  mode: 'EDIT' | 'DUB';
  setMode: (mode: 'EDIT' | 'DUB') => void;
  setActiveTool: (tool: string) => void;
  setProjectName: (name: string) => void;
  appendLog: (message: string, level?: LogEntry['level']) => void;
  clearLog: () => void;
  runSeparate: () => Promise<{ message?: string }>;
  refreshModels: () => Promise<void>;
  downloadModel: (modelId: string) => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;
  loadModel: (modelId: string) => Promise<void>;
  unloadModel: (modelId: string) => Promise<void>;
  selectModel: (modelId: string) => Promise<void>;
  setDevice: (device: 'auto' | 'cuda' | 'cpu') => Promise<void>;
  cancelTranscription: () => Promise<void>;
  downloadQwen: () => Promise<void>;
  deleteQwen: () => Promise<void>;
  downloadNllb: () => Promise<void>;
  deleteNllb: () => Promise<void>;
  downloadDemucs: () => Promise<void>;
  deleteDemucs: () => Promise<void>;
  downloadMmsKhm: (engineId: string) => Promise<void>;
  deleteMmsKhm: (engineId: string) => Promise<void>;
  downloadCudaPack: () => Promise<void>;
  deleteCudaPack: () => Promise<void>;
  detectSpeakers: () => Promise<{ speakers: string[]; message: string }>;
  cutRange: (start: number, end: number, mode: 'remove' | 'keep') => Promise<{ message: string }>;
  patchSegmentLocal: (segmentId: string, patch: { text?: string; translation?: string; start?: number; end?: number }) => void;
  updateSegment: (projectId: string, segmentId: string, patch: { text?: string; translation?: string; start?: number; end?: number }) => Promise<void>;
  splitSegment: (projectId: string, segmentId: string, at: number) => Promise<void>;
  mergeSegmentWithNext: (projectId: string, segmentId: string) => Promise<void>;
  deleteSegment: (projectId: string, segmentId: string) => Promise<void>;
  refreshVoiceConfig: () => Promise<void>;
  setProviderKey: (provider: string, key: string) => Promise<void>;
  removeProviderKey: (provider: string) => Promise<void>;
  testProvider: (provider: string) => Promise<{ models: string[] }>;
  previewLine: (payload: { text: string; provider: string; voice?: string; model?: string; emotion: string; intensity: number; speed?: number; language: string }) => Promise<string>;
  refreshCharacters: () => Promise<void>;
  saveCharacters: (characters: Record<string, Record<string, unknown>>) => Promise<void>;
  hydrateStudio: () => Promise<void>;
  refreshProjects: () => Promise<void>;
  switchProject: (projectId: string) => Promise<void>;
  deleteProject: (projectId: string) => Promise<void>;
  clearAllProjects: () => Promise<void>;
  createProject: (name: string, mode: 'EDIT' | 'DUB', targetLanguage?: string) => Promise<ProjectSummary>;
  uploadMedia: (projectId: string, file: File) => Promise<void>;
  runChunk: (chunkSeconds: number | null) => Promise<{ message?: string } | undefined>;
  runWorkflowStep: (workflow: 'transcribe' | 'translate' | 'tts', projectId?: string, options?: { targetLanguage?: string; voice?: string; transcribeLanguage?: string; beamSize?: number; ttsProvider?: string; ttsModel?: string; emotion?: string; intensity?: number; speed?: number; acting?: boolean; openaiActing?: boolean; openaiReview?: boolean; translationProvider?: string }) => Promise<void>;
  runExport: (options: {
    includeSubtitles: boolean;
    burnSubtitles: boolean;
    preserveOriginalAudio: boolean;
    originalAudioVolume: number;
    exportMode: 'single' | 'parts' | 'both';
    exportFolder: string | null;
    deliverToFolder: boolean;
    autoDuck: boolean;
    keepMusic: boolean;
    keepSfx: boolean;
    platformStyle: 'original' | 'vertical' | 'standard' | 'square';
    platformFit: 'blur' | 'crop';
  }) => Promise<{ vtt: string | null; videoUrl: string | null; parts: Array<{ index: number; name: string }>; folder: string | null; files: string[] }>;
  clearProject: () => void;
  updateDubAudioSegments: (segments: TimelineSegment[]) => void;
}

interface BackendProject {
  id: string;
  name: string;
  mode: string;
  status: string;
  progress: number;
  duration: string;
  has_media: boolean;
  media_name?: string | null;
  media_resolution?: string | null;
  media_size?: number | null;
}

interface WorkspacePayload {
  project: BackendProject;
  segments: Array<Record<string, any>>;
  audio_segments: Array<Record<string, any>>;
  chunks?: Array<Record<string, any>>;
  chunk_seconds?: number | null;
  parts_count?: number;
  export_folder?: string | null;
  separated?: boolean;
  workflow: { imported: boolean; chunked?: boolean; transcribed: boolean; translated: boolean; dubbed: boolean; subtitles_exported: boolean; video_ready?: boolean };
}

const initialProject: ProjectSummary = {
  id: '',
  name: 'Untitled project',
  mode: 'DUB',
  duration: '00:00:00',
  status: 'Ready',
  progress: 0,
  hasMedia: false,
  mediaName: null,
  mediaResolution: null,
  mediaSize: null,
};

const initialProviders: ProviderStatus[] = [
  { id: 'whisper', name: 'Faster-Whisper', mode: 'LOCAL AI', connected: false, primary: true, fallback: false },
  { id: 'google_translate', name: 'Google Translate', mode: 'FREE ONLINE', connected: false, primary: false, fallback: true },
  { id: 'edge_tts', name: 'Edge TTS', mode: 'FREE ONLINE', connected: false, primary: false, fallback: true },
];

const initialSegments: TimelineSegment[] = [];

const initialWorkflowSteps: WorkflowStep[] = [
  { id: 'import', label: 'Import video', detail: 'Choose a video to begin', status: 'Queued', progress: 0 },
  { id: 'chunk', label: 'Chunk video', detail: 'Optional: split long videos into parts', status: 'Queued', progress: 0 },
  { id: 'transcribe', label: 'Transcribe', detail: 'Waiting for a video', status: 'Queued', progress: 0 },
  { id: 'translate', label: 'Translate', detail: 'Waiting for transcript', status: 'Queued', progress: 0 },
  { id: 'voice', label: 'Generate dubbed voice', detail: 'Waiting for translation', status: 'Queued', progress: 0 },
  { id: 'subtitle', label: 'Subtitle sync', detail: 'Waiting for transcript', status: 'Queued', progress: 0 },
];

const initialMedia: MediaAsset[] = [];
const initialProjectList: ProjectSummary[] = [];
const apiBase = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';

const readJson = async <T>(input: RequestInfo | URL): Promise<T> => {
  const url = typeof input === 'string' ? `${apiBase}${input}` : input;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
};

const toProjectSummary = (raw: BackendProject): ProjectSummary => ({
  id: raw.id,
  name: raw.name,
  mode: raw.mode as ProjectSummary['mode'],
  status: raw.status as ProjectSummary['status'],
  progress: raw.progress,
  duration: raw.duration,
  hasMedia: raw.has_media,
  mediaName: raw.media_name ?? null,
  mediaResolution: raw.media_resolution ?? null,
  mediaSize: raw.media_size ?? null,
});

const workspaceToPatch = (workspace: WorkspacePayload, state: StudioState) => ({
  project: toProjectSummary(workspace.project),
  transcript: workspace.segments.map((segment, index) => ({
    id: String(segment.id ?? `segment-${index + 1}`),
    speaker: String(segment.speaker ?? `Speaker ${index + 1}`),
    start: Number(segment.start ?? 0),
    end: Number(segment.end ?? 0),
    text: String(segment.text ?? ''),
    translation: String(segment.translation ?? ''),
    voice: String(segment.voice ?? 'Male Voice A'),
    color: index % 2 === 0 ? '#38bdf8' : '#a78bfa',
  })),
  dubAudioSegments: workspace.audio_segments.map((segment, index) => ({
    id: String(segment.id ?? `dub-${index + 1}`),
    speaker: String(segment.voice ?? 'Dubbed Voice'),
    start: Number(segment.start ?? 0),
    end: Number(segment.end ?? 0),
    text: String(segment.text ?? ''),
    translation: String(segment.text ?? ''),
    voice: String(segment.voice ?? 'Dubbed Voice'),
    color: '#34d399',
    path: String(segment.path ?? ''),
  })),
  workflowSteps: state.workflowSteps.map((step) => {
    const completed = step.id === 'import' ? workspace.workflow.imported
      : step.id === 'chunk' ? Boolean(workspace.workflow.chunked)
        : step.id === 'transcribe' ? workspace.workflow.transcribed
          : step.id === 'translate' ? workspace.workflow.translated
            : step.id === 'voice' ? workspace.workflow.dubbed
              : workspace.workflow.subtitles_exported;
    return completed
      ? { ...step, status: 'Completed' as const, detail: 'Restored from saved project', progress: 100 }
      : step;
  }),
  mediaLibrary: workspace.project.has_media ? [{
    id: workspace.project.id,
    name: workspace.project.media_name ?? 'Imported video',
    type: 'Video' as const,
    duration: workspace.project.duration,
    size: `${((workspace.project.media_size ?? 0) / (1024 * 1024)).toFixed(1)} MB`,
    resolution: workspace.project.media_resolution ?? 'Unknown',
    badge: workspace.project.media_resolution ?? 'Video',
  }] : [],
  videoChunks: (workspace.chunks ?? []).map((chunk: Record<string, any>, index: number) => ({
    index: Number(chunk.index ?? index + 1),
    start: Number(chunk.start ?? 0),
    duration: Number(chunk.duration ?? 0),
  })),
  chunkSeconds: workspace.chunk_seconds ?? null,
  dubbedVideoReady: Boolean(workspace.workflow.video_ready),
  exportedParts: Array.from({ length: Number(workspace.parts_count ?? 0) }, (_, i) => ({
    index: i + 1,
    name: `part_${String(i + 1).padStart(3, '0')}.mp4`,
  })),
  defaultExportFolder: null,
  exportFolder: workspace.export_folder ?? null,
  separated: Boolean(workspace.separated),
});

// --- Live workflow progress over WebSocket ---------------------------------

let logSeq = 0;

const STAGE_TO_STEP: Record<string, string> = {
  'Chunking video': 'chunk',
  Transcribing: 'transcribe',
  Translating: 'translate',
  'Generating Voice': 'voice',
  Exporting: 'subtitle',
};

let progressSocket: WebSocket | null = null;
let progressRetryTimer: number | undefined;

const scheduleProgressReconnect = () => {
  if (progressRetryTimer !== undefined) return;
  progressRetryTimer = window.setTimeout(() => {
    progressRetryTimer = undefined;
    connectProgressSocket();
  }, 5000);
};

const connectProgressSocket = () => {
  if (progressSocket && progressSocket.readyState <= WebSocket.OPEN) return;
  const wsBase = import.meta.env.DEV
    ? `${window.location.protocol === 'https:' ? 'wss://' : 'ws://'}${window.location.host}`
    : 'ws://127.0.0.1:8001';
  let socket: WebSocket;
  try {
    socket = new WebSocket(`${wsBase}/ws`);
  } catch {
    scheduleProgressReconnect();
    return;
  }
  progressSocket = socket;
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(String(event.data)) as { project_id: string; stage: string; progress: number; message: string; engine?: string | null };
      useStudioStore.getState().appendLog(`[${message.stage}] ${message.message}`);
      const stepId = STAGE_TO_STEP[message.stage];
      useStudioStore.setState((state) => {
        if (!state.project.id || message.project_id !== state.project.id) return state;
        return {
          project: { ...state.project, progress: message.progress },
          ...(message.stage === 'Translating' && message.engine ? { activeTranslateEngine: message.engine } : {}),
          workflowSteps: stepId
            ? state.workflowSteps.map((step) => step.id === stepId
              ? { ...step, status: 'Active', detail: message.message, progress: message.progress }
              : step)
            : state.workflowSteps,
        };
      });
    } catch {
      // Ignore malformed progress frames.
    }
  };
  socket.onclose = () => {
    progressSocket = null;
    scheduleProgressReconnect();
  };
  socket.onerror = () => {
    socket.close();
  };
};

export const useStudioStore = create<StudioState>((set, get) => ({
  project: initialProject,
  projectList: initialProjectList,
  providers: initialProviders,
  transcript: initialSegments,
  dubAudioSegments: [],
  videoChunks: [],
  chunkSeconds: null,
  dubbedVideoReady: false,
  mediaVersion: 0,
  exportedParts: [],
  defaultExportFolder: null,
  exportFolder: null,
  separated: false,
  processingLog: [],
  models: [],
  hardware: null,
  modelStorage: null,
  qwenModel: null,
  nllbModel: null,
  activeTranslateEngine: null,
  demucsModel: null,
  mmsKhmModels: [],
  cudaPack: null,
  providerKeys: { gemini: { has_key: false }, openai: { has_key: false } },
  voiceCapabilities: {},
  characters: { speakers: [], characters: {} },
  workflowSteps: initialWorkflowSteps,
  mediaLibrary: initialMedia,
  activeTool: 'Auto Dubbing',
  mode: 'DUB',
  setMode: (mode) => set((state) => ({ ...state, mode, project: { ...state.project, mode } })),
  setActiveTool: (tool) => set({ activeTool: tool }),
  setProjectName: (name) => set((state) => ({ project: { ...state.project, name } })),
  appendLog: (message, level = 'info') =>
    set((state) => ({
      processingLog: [
        ...state.processingLog.slice(-299),
        { id: ++logSeq, time: new Date().toLocaleTimeString('en-GB', { hour12: false }), message, level },
      ],
    })),
  clearLog: () => set({ processingLog: [] }),
  runSeparate: async () => {
    const projectId = get().project.id;
    get().appendLog('Separating original audio into voice and sound effects (this can take a while on CPU)...');
    set((state) => ({ project: { ...state.project, status: 'Processing' } }));
    try {
      const response = await fetch(`${apiBase}/api/process/separate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId, workflow: 'separate' }),
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.detail ?? `Unable to separate the audio (HTTP ${response.status})`);
      }
      const payload = await response.json();
      get().appendLog(payload.message ?? 'Audio separated.', 'success');
      set((state) => ({ project: { ...state.project, status: 'Ready' }, separated: true }));
      return payload as { message?: string };
    } catch (error) {
      get().appendLog(error instanceof Error ? error.message : 'Audio separation failed', 'error');
      set((state) => ({ project: { ...state.project, status: 'Ready' } }));
      throw error;
    }
  },
  refreshModels: async () => {
    try {
      const [models, hardware, storage, qwen, demucs, mmsKhm, nllb, cudaPack] = await Promise.all([
        readJson<WhisperModelInfo[]>('/api/models/faster-whisper'),
        readJson<HardwareInfo>('/api/system/hardware'),
        readJson<ModelStorageInfo>('/api/models/storage').catch(() => null),
        readJson<WhisperModelInfo>('/api/models/qwen').catch(() => null),
        readJson<WhisperModelInfo>('/api/models/demucs').catch(() => null),
        readJson<{ models: WhisperModelInfo[] }>('/api/models/mms-khm').catch(() => null),
        readJson<WhisperModelInfo>('/api/models/nllb').catch(() => null),
        readJson<CudaPackInfo>('/api/models/cuda-pack').catch(() => null),
      ]);
      set({ models, hardware, modelStorage: storage, qwenModel: qwen, demucsModel: demucs, mmsKhmModels: mmsKhm?.models ?? [], nllbModel: nllb, cudaPack });
    } catch {
      // Backend offline; keep the last known state.
    }
  },
  downloadModel: async (modelId) => {
    const model = get().models.find((item) => item.id === modelId);
    get().appendLog(`Downloading ${model?.name ?? modelId}...`);
    const response = await fetch(`${apiBase}/api/models/faster-whisper/${modelId}/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      get().appendLog(problem.detail ?? `Unable to start the download (HTTP ${response.status})`, 'error');
      throw new Error(problem.detail ?? `Unable to start the download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteModel: async (modelId) => {
    const response = await fetch(`${apiBase}/api/models/faster-whisper/${modelId}`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete the model (HTTP ${response.status})`);
    }
    get().appendLog(`Model deleted: ${modelId}`, 'success');
    await get().refreshModels();
  },
  loadModel: async (modelId) => {
    get().appendLog(`Loading ${modelId} into memory...`);
    const response = await fetch(`${apiBase}/api/models/faster-whisper/${modelId}/load`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      get().appendLog(problem.detail ?? `Unable to load the model (HTTP ${response.status})`, 'error');
      throw new Error(problem.detail ?? `Unable to load the model (HTTP ${response.status})`);
    }
    get().appendLog(`Model loaded: ${modelId}`, 'success');
    await get().refreshModels();
  },
  unloadModel: async (modelId) => {
    await fetch(`${apiBase}/api/models/faster-whisper/${modelId}/unload`, { method: 'POST' });
    get().appendLog(`Model unloaded from memory: ${modelId}`);
    await get().refreshModels();
  },
  selectModel: async (modelId) => {
    const response = await fetch(`${apiBase}/api/models/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected_whisper_model: modelId }),
    });
    if (!response.ok) {
      throw new Error(`Unable to select the model (HTTP ${response.status})`);
    }
    get().appendLog(`Transcription model selected: ${modelId}`, 'success');
    await get().refreshModels();
  },
  setDevice: async (device) => {
    const response = await fetch(`${apiBase}/api/models/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device }),
    });
    if (!response.ok) {
      throw new Error(`Unable to set the device (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  cancelTranscription: async () => {
    const projectId = get().project.id;
    await fetch(`${apiBase}/api/process/transcribe/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId }),
    }).catch(() => undefined);
    get().appendLog('Cancelling transcription...');
  },
  downloadQwen: async () => {
    const response = await fetch(`${apiBase}/api/models/qwen/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to start the Qwen download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteQwen: async () => {
    const response = await fetch(`${apiBase}/api/models/qwen`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete Qwen (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  downloadNllb: async () => {
    const response = await fetch(`${apiBase}/api/models/nllb/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to start the NLLB download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteNllb: async () => {
    const response = await fetch(`${apiBase}/api/models/nllb`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete NLLB (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  downloadDemucs: async () => {
    const response = await fetch(`${apiBase}/api/models/demucs/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to start the Demucs download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteDemucs: async () => {
    const response = await fetch(`${apiBase}/api/models/demucs`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete Demucs (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  downloadMmsKhm: async (engineId) => {
    const response = await fetch(`${apiBase}/api/models/mms/${engineId}/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to start the Khmer model download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteMmsKhm: async (engineId) => {
    const response = await fetch(`${apiBase}/api/models/mms/${engineId}`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete the Khmer model (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  downloadCudaPack: async () => {
    const response = await fetch(`${apiBase}/api/models/cuda-pack/download`, { method: 'POST' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to start the GPU pack download (HTTP ${response.status})`);
    }
    void get().refreshModels();
  },
  deleteCudaPack: async () => {
    const response = await fetch(`${apiBase}/api/models/cuda-pack`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete the GPU pack (HTTP ${response.status})`);
    }
    await get().refreshModels();
  },
  detectSpeakers: async () => {
    const projectId = get().project.id;
    const response = await fetch(`${apiBase}/api/process/speaker`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Speaker detection failed (HTTP ${response.status})`);
    }
    const payload = await response.json() as { speakers: string[]; message: string };
    get().appendLog(payload.message ?? `Detected ${payload.speakers.length} speakers.`, 'success');
    // Reload the workspace so transcript speakers + character voices refresh.
    await get().hydrateStudio();
    await get().refreshCharacters();
    return payload;
  },
  cutRange: async (start, end, mode) => {
    const projectId = get().project.id;
    const response = await fetch(`${apiBase}/api/projects/${projectId}/cut`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId, start, end, cut_mode: mode }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Cutting failed (HTTP ${response.status})`);
    }
    const payload = await response.json() as { message: string; duration: string };
    get().appendLog(payload.message ?? 'Cut applied to the timeline.', 'success');
    // The backend re-writes the media file in place: refresh everything and
    // bump mediaVersion so the <video> element reloads the new file.
    await get().hydrateStudio();
    set((state) => ({ mediaVersion: state.mediaVersion + 1 }));
    return payload;
  },
  patchSegmentLocal: (segmentId, patch) =>
    set((state) => ({
      transcript: state.transcript.map((segment) => (segment.id === segmentId ? { ...segment, ...patch } : segment)),
      // Dub audio follows timing edits; text edits make it stale (backend drops it too).
      dubAudioSegments: state.dubAudioSegments.map((audio) => {
        if (audio.id !== segmentId) return audio;
        const next = { ...audio };
        if (patch.start !== undefined) next.start = patch.start;
        if (patch.end !== undefined) next.end = patch.end;
        return next;
      }),
    })),
  updateSegment: async (projectId, segmentId, patch) => {
    get().patchSegmentLocal(segmentId, patch);
    const response = await fetch(`${apiBase}/api/projects/${projectId}/segments/${segmentId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to save the segment (HTTP ${response.status})`);
    }
    // Text edits invalidate that line's dub audio (the backend drops it too).
    if (patch.text !== undefined || patch.translation !== undefined) {
      set((state) => ({ dubAudioSegments: state.dubAudioSegments.filter((audio) => audio.id !== segmentId) }));
    }
  },
  splitSegment: async (projectId, segmentId, at) => {
    const segment = get().transcript.find((item) => item.id === segmentId);
    if (!segment) throw new Error('Segment not found');
    const fraction = Math.max(0, Math.min(1, (at - segment.start) / Math.max(0.1, segment.end - segment.start)));
    const splitText = (text: string) => {
      const words = text.split(' ').filter(Boolean);
      const cut = Math.max(1, Math.round(words.length * fraction));
      return [words.slice(0, cut).join(' '), words.slice(cut).join(' ')] as const;
    };
    const [text1, text2] = splitText(segment.text);
    const [translation1, translation2] = splitText(segment.translation || '');
    const response = await fetch(`${apiBase}/api/projects/${projectId}/segments/${segmentId}/split`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ at, text1, text2, translation1, translation2 }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to split the segment (HTTP ${response.status})`);
    }
    const pieces = (await response.json()).segments as Array<{ id: string; start: number; end: number; text: string; translation: string }>;
    set((state) => ({
      transcript: state.transcript.flatMap((segment) => {
        if (segment.id !== segmentId) return [segment];
        return pieces.map((piece, index) => ({
          ...segment,
          id: piece.id,
          start: Number(piece.start),
          end: Number(piece.end),
          text: piece.text,
          translation: piece.translation,
          color: index === 0 ? segment.color : '#a78bfa',
        }));
      }),
      // Structural edit: the dub mix must be rebuilt, so drop stale lines.
      dubAudioSegments: [],
    }));
    get().appendLog('Segment split in two. Re-dub to regenerate the audio.', 'success');
  },
  mergeSegmentWithNext: async (projectId, segmentId) => {
    const response = await fetch(`${apiBase}/api/projects/${projectId}/segments/${segmentId}/merge-next`, {
      method: 'POST',
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to merge (HTTP ${response.status})`);
    }
    const merged = (await response.json()).segment as { id: string; end: number; text: string; translation: string };
    set((state) => {
      const index = state.transcript.findIndex((segment) => segment.id === segmentId);
      if (index < 0 || index + 1 >= state.transcript.length) return { transcript: state.transcript };
      const first = state.transcript[index];
      const second = state.transcript[index + 1];
      const mergedSegment = {
        ...first,
        end: merged.end,
        text: merged.text,
        translation: merged.translation || first.translation,
      };
      const transcript = [...state.transcript];
      transcript.splice(index, 2, mergedSegment);
      return { transcript };
    });
    get().appendLog('Segments merged. Re-dub to regenerate the audio.', 'success');
  },
  deleteSegment: async (projectId, segmentId) => {
    const response = await fetch(`${apiBase}/api/projects/${projectId}/segments/${segmentId}`, { method: 'DELETE' });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to delete the segment (HTTP ${response.status})`);
    }
    set((state) => ({
      transcript: state.transcript.filter((segment) => segment.id !== segmentId),
      dubAudioSegments: [],
    }));
    get().appendLog('Segment deleted.', 'success');
  },
  refreshVoiceConfig: async () => {
    try {
      const [keys, capabilities] = await Promise.all([
        readJson<{ gemini: { has_key: boolean }; openai: { has_key: boolean } }>('/api/providers/keys').catch(() => null),
        readJson<Record<string, { voices: Array<{ id: string; style: string }>; emotions: string[]; style_control: boolean; khmer_supported: boolean; languages: string[]; default_model: string; label: string; installed?: boolean }> & { emotions?: string[] }>('/api/tts/capabilities').catch(() => null),
      ]);
      set({
        providerKeys: keys ?? get().providerKeys,
        voiceCapabilities: capabilities ?? get().voiceCapabilities,
      });
    } catch {
      // Backend offline; keep last known state.
    }
  },
  setProviderKey: async (provider, key) => {
    const response = await fetch(`${apiBase}/api/providers/${provider}/key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Unable to store the key (HTTP ${response.status})`);
    }
    get().appendLog(`${provider} API key stored in Windows Credential Manager.`, 'success');
    await get().refreshVoiceConfig();
  },
  removeProviderKey: async (provider) => {
    await fetch(`${apiBase}/api/providers/${provider}/key`, { method: 'DELETE' });
    get().appendLog(`${provider} API key removed.`);
    await get().refreshVoiceConfig();
  },
  testProvider: async (provider) => {
    const response = await fetch(`${apiBase}/api/providers/${provider}/test`, { method: 'POST' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail ?? `Connection test failed (HTTP ${response.status})`);
    }
    await get().refreshVoiceConfig();
    return { models: payload.tts_models ?? [] };
  },
  previewLine: async (payload) => {
    const response = await fetch(`${apiBase}/api/tts/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail ?? `Preview failed (HTTP ${response.status})`);
    }
    const data = await response.json();
    return data.audio as string;
  },
  refreshCharacters: async () => {
    const projectId = get().project.id;
    if (!projectId) return;
    try {
      const characters = await readJson<{ speakers: string[]; characters: Record<string, Record<string, unknown>> }>(`/api/projects/${projectId}/characters`);
      set({ characters });
    } catch {
      // Keep last known characters.
    }
  },
  saveCharacters: async (characters) => {
    const projectId = get().project.id;
    const response = await fetch(`${apiBase}/api/projects/${projectId}/characters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ characters }),
    });
    if (!response.ok) {
      throw new Error(`Unable to save character voices (HTTP ${response.status})`);
    }
    set((state) => ({ characters: { ...state.characters, characters } }));
    get().appendLog('Character voices saved.', 'success');
  },
  hydrateStudio: async () => {
    connectProgressSocket();
    let savedProjectId = localStorage.getItem('dubvibe-project-id');
    try {
      const saved = localStorage.getItem('dubvibe-project');
      if (saved) {
        const parsed = JSON.parse(saved);
        savedProjectId = savedProjectId ?? parsed.project?.id ?? null;
        // Merge saved steps with the current step list so newly added steps
        // (e.g. "Chunk video") also appear for restored sessions.
        const savedSteps: WorkflowStep[] | undefined = Array.isArray(parsed.workflowSteps) ? parsed.workflowSteps : undefined;
        set({
          project: parsed.project ?? get().project,
          transcript: parsed.transcript ?? get().transcript,
          workflowSteps: savedSteps?.length
            ? initialWorkflowSteps.map((step) => savedSteps.find((savedStep) => savedStep.id === step.id) ?? step)
            : get().workflowSteps,
        });
      }
    } catch (error) {
      console.warn('Saved project data could not be restored.', error);
    }
    try {
      const [providers, projectList, workspace] = await Promise.all([
        readJson<Array<{ id: string; name: string; mode: string; connected: boolean; primary: boolean; fallback: boolean }>>('/api/providers'),
        readJson<BackendProject[]>('/api/projects').catch(() => [] as BackendProject[]),
        savedProjectId
          ? readJson<WorkspacePayload>(`/api/projects/${savedProjectId}/workspace`)
          : Promise.resolve(null),
      ]);
      set({
        providers: providers.length ? providers.map((provider) => ({
          id: provider.id,
          name: provider.name,
          mode: provider.mode as 'LOCAL AI' | 'FREE ONLINE' | 'PAID API',
          connected: provider.connected,
          primary: provider.primary,
          fallback: provider.fallback,
        })) : get().providers,
        projectList: projectList.map(toProjectSummary),
        ...(workspace ? workspaceToPatch(workspace, get()) : {}),
      });
      if (workspace) localStorage.setItem('dubvibe-project-id', workspace.project.id);
      if (get().project.id) {
        try {
          const destination = await readJson<{ folder: string }>(`/api/export/destination?project_id=${get().project.id}`);
          set({ defaultExportFolder: destination.folder });
        } catch {
          // The default destination is a convenience; ignore failures.
        }
      }
      get().appendLog('Connected to the local processing service.', 'success');
      void get().refreshCharacters();
    } catch (error) {
      get().appendLog('Local processing service is offline.', 'error');
      console.warn('Backend is unavailable; using local studio seed data.', error);
    }
  },
  refreshProjects: async () => {
    try {
      const list = await readJson<BackendProject[]>('/api/projects');
      set({ projectList: list.map(toProjectSummary) });
    } catch {
      // Backend offline; keep the current list.
    }
  },
  switchProject: async (projectId) => {
    const workspace = await readJson<WorkspacePayload>(`/api/projects/${projectId}/workspace`);
    localStorage.setItem('dubvibe-project-id', projectId);
    localStorage.removeItem('dubvibe-project');
    set(workspaceToPatch(workspace, get()));
    try {
      const destination = await readJson<{ folder: string }>(`/api/export/destination?project_id=${projectId}`);
      set({ defaultExportFolder: destination.folder });
    } catch {
      // The default destination is a convenience; ignore failures.
    }
  },
  deleteProject: async (projectId) => {
    const response = await fetch(`${apiBase}/api/projects/${projectId}`, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(`Unable to delete the project (HTTP ${response.status})`);
    }
    if (get().project.id === projectId) {
      get().clearProject();
    }
    await get().refreshProjects();
  },
  clearAllProjects: async () => {
    const response = await fetch(`${apiBase}/api/projects`, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(`Unable to clear the project history (HTTP ${response.status})`);
    }
    // Every project is gone, so the workspace always resets to empty.
    get().clearProject();
    await get().refreshProjects();
  },
  createProject: async (name, mode, targetLanguage = 'Khmer') => {
    const response = await fetch(`${apiBase}/api/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, mode, target_language: targetLanguage }),
    });

    if (!response.ok) {
      throw new Error('Unable to create project');
    }

    const project = (await response.json()) as ProjectSummary;
    localStorage.setItem('dubvibe-project-id', project.id);
    set({
      project: {
        ...get().project,
        ...project,
        hasMedia: false,
        mediaName: null,
        mediaResolution: null,
        mediaSize: null,
      },
      mode,
      transcript: [],
      dubAudioSegments: [],
      mediaLibrary: [],
      workflowSteps: initialWorkflowSteps.map((step) => ({ ...step })),
    });
    void get().refreshProjects();
    try {
      const destination = await readJson<{ folder: string }>(`/api/export/destination?project_id=${project.id}`);
      set({ defaultExportFolder: destination.folder });
    } catch {
      // Optional convenience.
    }
    return project;
  },
  uploadMedia: async (projectId, file) => {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch(`${apiBase}/api/projects/${projectId}/media`, {
      method: 'POST',
      body: form,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail ?? `Unable to import video (HTTP ${response.status})`);
    }
    const metadata = await response.json();
    get().appendLog(`${metadata.filename} imported · ${metadata.duration} · ${metadata.resolution}`, 'success');
    const media: MediaAsset = {
      id: projectId,
      name: metadata.filename,
      type: 'Video',
      duration: metadata.duration,
      size: `${(metadata.size / (1024 * 1024)).toFixed(1)} MB`,
      resolution: metadata.resolution,
      badge: metadata.resolution === 'Unknown' ? 'Video' : metadata.resolution,
    };
    set((state) => ({
      project: {
        ...state.project,
        duration: metadata.duration,
        hasMedia: true,
        mediaName: metadata.filename,
        mediaResolution: metadata.resolution,
        mediaSize: metadata.size,
      },
      mediaLibrary: [media, ...state.mediaLibrary.filter((item) => item.type !== 'Video')],
      videoChunks: [],
      chunkSeconds: null,
      workflowSteps: state.workflowSteps.map((step) => step.id === 'import'
        ? { ...step, status: 'Completed', detail: `${metadata.filename} imported`, progress: 100 }
        : step),
    }));
  },
  runChunk: async (chunkSeconds) => {
    const projectId = get().project.id;
    get().appendLog(chunkSeconds ? `Splitting video into ${Math.round(chunkSeconds / 60)}-minute parts...` : 'Removing video split...');
    set((state) => ({
      project: { ...state.project, status: 'Processing' },
      workflowSteps: state.workflowSteps.map((step) => step.id === 'chunk'
        ? { ...step, status: 'Active', detail: chunkSeconds ? 'Splitting video into parts...' : 'Removing split...', progress: 20 }
        : step),
    }));
    try {
      const response = await fetch(`${apiBase}/api/process/chunk`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId, workflow: 'chunk', chunk_seconds: chunkSeconds }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail ?? `Unable to split the video (HTTP ${response.status})`);
      }
      const payload = await response.json();
      const chunks: Array<Record<string, any>> = Array.isArray(payload.chunks) ? payload.chunks : [];
      get().appendLog(payload.message ?? (chunks.length ? `Video split into ${chunks.length} parts.` : 'Video split removed.'), 'success');
      set((state) => ({
        project: { ...state.project, status: 'Ready' },
        videoChunks: chunks.map((chunk, index) => ({
          index: Number(chunk.index ?? index + 1),
          start: Number(chunk.start ?? 0),
          duration: Number(chunk.duration ?? 0),
        })),
        chunkSeconds: payload.chunk_seconds ?? null,
        workflowSteps: state.workflowSteps.map((step) => {
          if (step.id !== 'chunk') return step;
          return chunks.length
            ? { ...step, status: 'Completed' as const, detail: `${chunks.length} parts × ${Math.round((payload.chunk_seconds ?? 0) / 60)} min`, progress: 100 }
            : { ...step, status: 'Queued' as const, detail: 'Optional: split long videos into parts', progress: 0 };
        }),
      }));
      return payload as { message?: string };
    } catch (error) {
      get().appendLog(error instanceof Error ? error.message : 'Video splitting failed', 'error');
      set((state) => ({
        project: { ...state.project, status: 'Ready' },
        workflowSteps: state.workflowSteps.map((step) => step.id === 'chunk'
          ? { ...step, status: 'Failed', detail: error instanceof Error ? error.message : 'Splitting failed' }
          : step),
      }));
      throw error;
    }
  },
  runWorkflowStep: async (workflow, projectId, options) => {
    const currentProjectId = projectId ?? get().project.id;
    const stepId = workflow === 'tts' ? 'voice' : workflow;
    const label = workflow === 'tts' ? 'Generating dubbed voice' : workflow === 'transcribe' ? 'Transcribing' : 'Translating';
    set((state) => ({
      project: { ...state.project, status: 'Processing' },
      activeTranslateEngine: null,
      workflowSteps: state.workflowSteps.map((step) => step.id === stepId
        ? { ...step, status: 'Active', detail: `${label}...`, progress: 20 }
        : step),
    }));

    try {
      const response = await fetch(`${apiBase}/api/process/${workflow}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: currentProjectId,
          workflow,
          target_language: options?.targetLanguage,
          voice: options?.voice,
          language: options?.transcribeLanguage || null,
          beam_size: options?.beamSize ?? 5,
          tts_provider: options?.ttsProvider ?? 'auto',
          tts_model: options?.ttsModel || null,
          emotion: options?.emotion ?? 'Auto',
          intensity: options?.intensity ?? 70,
          speed: options?.speed ?? 1,
          acting: options?.acting ?? true,
          openai_acting: options?.openaiActing ?? false,
          openai_review: options?.openaiReview ?? false,
          translation_provider: options?.translationProvider ?? 'auto',
        }),
      });

      if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.detail ?? `Unable to run ${workflow} (HTTP ${response.status})`);
      }

      const payload = await response.json();
      const computedProgress = workflow === 'transcribe' ? 58 : workflow === 'translate' ? 82 : 96;
      if (workflow === 'transcribe' && payload.result?.cancelled) {
        get().appendLog(`Transcription cancelled — ${payload.result.segments?.length ?? 0} segments kept.`, 'info');
      } else {
        get().appendLog(`${label} complete.`, 'success');
      }
      set((state) => ({
        project: {
          ...state.project,
          id: currentProjectId,
          progress: computedProgress,
          status: 'Ready',
        },
        workflowSteps: state.workflowSteps.map((step) => step.id === stepId
          ? { ...step, status: 'Completed', detail: `${label} complete`, progress: 100 }
          : step),
        transcript: workflow === 'transcribe' && Array.isArray(payload.result?.segments)
          ? payload.result.segments.map((segment: any, index: number) => ({
            id: segment.id ?? `segment-${index + 1}`,
            speaker: segment.speaker ?? `Speaker ${index + 1}`,
            start: Number(segment.start ?? 0),
            end: Number(segment.end ?? 0),
            text: segment.text ?? '',
            translation: segment.translation ?? '',
            voice: options?.voice ?? (index % 2 === 0 ? 'Male Voice A' : 'Female Voice B'),
            color: index % 2 === 0 ? '#38bdf8' : '#a78bfa',
          }))
          : workflow === 'translate' && Array.isArray(payload.translation)
            ? state.transcript.map((segment, index) => ({
              ...segment,
              translation: payload.translation[index]?.translation ?? segment.translation,
            }))
            : state.transcript,
        dubAudioSegments: workflow === 'tts' && Array.isArray(payload.result?.segments)
          ? payload.result.segments.map((segment: any, index: number) => ({
            id: segment.id ?? `dub-${index + 1}`,
            speaker: segment.voice ?? options?.voice ?? 'Dubbed Voice',
            start: Number(segment.start ?? 0),
            end: Number(segment.end ?? 0),
            text: segment.text ?? '',
            translation: segment.text ?? '',
            voice: segment.voice ?? options?.voice ?? 'Dubbed Voice',
            color: '#34d399',
            path: segment.path ?? '',
          }))
          : state.dubAudioSegments,
      }));
    } catch (error) {
      set((state) => ({
        project: { ...state.project, status: 'Ready' },
        workflowSteps: state.workflowSteps.map((step) => step.id === stepId
          ? { ...step, status: 'Failed', detail: error instanceof Error ? error.message : 'Workflow failed' }
          : step),
      }));
      throw error;
    }
  },
  runExport: async (options) => {
    const response = await fetch(`${apiBase}/api/process/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: get().project.id,
        workflow: 'export',
        include_subtitles: options.includeSubtitles,
        burn_subtitles: options.burnSubtitles,
        preserve_original_audio: options.preserveOriginalAudio,
        original_audio_volume: options.originalAudioVolume,
        export_mode: options.exportMode,
        export_folder: options.exportFolder,
        deliver_to_folder: options.deliverToFolder,
        auto_duck: options.autoDuck,
        keep_music: options.keepMusic,
        keep_sfx: options.keepSfx,
        platform_style: options.platformStyle,
        platform_fit: options.platformFit,
      }),
    });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      const message = problem.detail ?? `Unable to export subtitles (HTTP ${response.status})`;
      get().appendLog(message, 'error');
      throw new Error(message);
    }
    const payload = await response.json();
    const parts: Array<{ index: number; name: string }> = Array.isArray(payload.export?.parts) ? payload.export.parts : [];
    get().appendLog(payload.export?.message ?? 'Export complete.', 'success');
    const fitted = Array.isArray(payload.audio_segments) ? payload.audio_segments : null;
    set((state) => ({
      project: { ...state.project, progress: 100, status: 'Ready' },
      dubbedVideoReady: Boolean(payload.export?.video_url),
      dubAudioSegments: fitted
        ? fitted.map((segment: Record<string, unknown>, index: number) => ({
          id: String(segment.id ?? `dub-${index + 1}`),
          speaker: String(segment.voice ?? 'Dubbed Voice'),
          start: Number(segment.start ?? 0),
          end: Number(segment.end ?? 0),
          text: String(segment.text ?? ''),
          translation: String(segment.text ?? ''),
          voice: String(segment.voice ?? 'Dubbed Voice'),
          color: '#34d399',
          path: String(segment.path ?? ''),
        }))
        : state.dubAudioSegments,
      exportedParts: parts,
      exportFolder: payload.export?.folder ?? state.exportFolder,
      workflowSteps: state.workflowSteps.map((step) => step.id === 'subtitle'
        ? options.includeSubtitles
          ? { ...step, status: 'Completed', detail: 'Subtitle export ready', progress: 100 }
          : { ...step, status: 'Skipped', detail: 'Subtitle sync disabled by user', progress: 0 }
        : step),
    }));
    return {
      vtt: payload.export?.payload ?? null,
      videoUrl: payload.export?.video_url ? `${apiBase}${payload.export.video_url}` : null,
      parts,
      folder: payload.export?.folder ?? null,
      files: Array.isArray(payload.export?.files) ? payload.export.files : [],
    };
  },
  clearProject: () => {
    localStorage.removeItem('dubvibe-project');
    localStorage.removeItem('dubvibe-project-id');
    set({
      project: initialProject,
      transcript: initialSegments,
      dubAudioSegments: [],
      videoChunks: [],
      chunkSeconds: null,
      dubbedVideoReady: false,
      exportedParts: [],
      defaultExportFolder: null,
      exportFolder: null,
      separated: false,
      mediaLibrary: initialMedia,
      models: [],
      hardware: null,
      modelStorage: null,
      qwenModel: null,
      nllbModel: null,
      activeTranslateEngine: null,
      demucsModel: null,
      mmsKhmModels: [],
      cudaPack: null,
      providerKeys: { gemini: { has_key: false }, openai: { has_key: false } },
      voiceCapabilities: {},
      characters: { speakers: [], characters: {} },
      workflowSteps: initialWorkflowSteps.map((step) => ({ ...step })),
    });
  },
  updateDubAudioSegments: (segments) => set({ dubAudioSegments: segments }),
}));
