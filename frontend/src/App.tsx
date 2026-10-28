import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AudioLines,
  Clock3,
  Cog,
  Cpu,
  Download,
  Languages,
  Mic,
  Play,
  Pause,
  Scissors,
  Search,
  Sparkles,
  SplitSquareVertical,
  Video,
  FileVideo,
  CheckCircle2,
  BadgeCheck,
  FolderOpen,
  Save,
  RefreshCw,
  SkipBack,
  SkipForward,
  FilePlus,
  Trash2,
  Wand2,
} from 'lucide-react';
import { useStudioStore } from './store/useStudioStore';
import { DubbingTrackEditor } from './components/DubbingTrackEditor';
import { TimelineEditor } from './components/TimelineEditor';
import type { ProjectSummary, OverlapProposal, FixOverlapsResponse } from './types';

const toolActions = [
  'Auto Dubbing',
  'Translate',
  'Auto Subtitle',
];

const topActions = [
  { label: 'New Project', icon: FilePlus },
  { label: 'Import Video', icon: FileVideo },
  { label: 'Refresh', icon: RefreshCw },
  { label: 'Save', icon: Save },
  { label: 'Preview', icon: Play },
  { label: 'Export', icon: Download },
  { label: 'Clear', icon: Trash2 },
];

const actionButtons = [
  { label: 'Transcribe', icon: Mic },
  { label: 'Translate', icon: Languages },
  { label: 'Dub', icon: AudioLines },
  { label: 'Subtitle', icon: SplitSquareVertical },
];

const trackLabels = ['Video', 'Original Speech', 'Translation', 'Dubbed Voice', 'Subtitle'];

const apiBaseUrl = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';

/** Target languages with their ISO code (the code the capability tables use). */
const TARGET_LANGUAGES: Array<{ name: string; code: string }> = [
  { name: 'Khmer', code: 'km' },
  { name: 'English', code: 'en' },
  { name: 'French', code: 'fr' },
  { name: 'Spanish', code: 'es' },
  { name: 'Chinese', code: 'zh' },
  { name: 'Japanese', code: 'ja' },
  { name: 'Korean', code: 'ko' },
  { name: 'Thai', code: 'th' },
  { name: 'Vietnamese', code: 'vi' },
  { name: 'German', code: 'de' },
  { name: 'Portuguese', code: 'pt' },
  { name: 'Russian', code: 'ru' },
  { name: 'Arabic', code: 'ar' },
  { name: 'Hindi', code: 'hi' },
  { name: 'Indonesian', code: 'id' },
];

/** Engine order used when the engine is "Auto" (mirrors tts_service.QUALITY_ORDER). */
const VOICE_ENGINE_ORDER = ['gemini', 'openai', 'edge', 'voxcpm2', 'sapi5'] as const;

const VOICE_ENGINE_LABELS: Record<string, string> = {
  gemini: 'Google Gemini TTS',
  openai: 'OpenAI TTS',
  edge: 'Edge TTS',
  voxcpm2: 'VoxCPM2',
  voxcpm_clone: 'VoxCPM2 Voice Clone',
  sapi5: 'System TTS',
};

/** Map the backend's audio_segments rows to the timeline's dub segments. */
const toDubSegments = (rows: Array<Record<string, unknown>>): import('./types').TimelineSegment[] =>
  rows.map((row, index) => ({
    id: String(row.id ?? `dub-${index + 1}`),
    speaker: String(row.voice ?? 'Dubbed Voice'),
    start: Number(row.start ?? 0),
    end: Number(row.end ?? 0),
    text: String(row.text ?? ''),
    translation: String(row.text ?? ''),
    voice: String(row.voice ?? 'Dubbed Voice'),
    color: '#34d399',
    path: row.path ? String(row.path) : '',
    ...(row.gender === 'Male' || row.gender === 'Female' ? { gender: row.gender } : {}),
  }));

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const dubbedVideoRef = useRef<HTMLVideoElement>(null);
  const {
    project,
    projectList,
    providers,
    transcript,
    dubAudioSegments,
    videoChunks,
    chunkSeconds,
    dubbedVideoReady,
    mediaVersion,
    dubAudioVersion,
    exportedParts,
    defaultExportFolder,
    exportFolder,
    separated,
    processingLog,
    workflowSteps,
    mediaLibrary,
    activeTool,
    mode,
    setMode,
    setActiveTool,
    setProjectName,
    appendLog,
    clearLog,
    hydrateStudio,
    refreshProjects,
    switchProject,
    deleteProject,
    clearAllProjects,
    createProject,
    uploadMedia,
    runChunk,
    runSeparate,
    runWorkflowStep,
    runExport,
    cutRange,
    clearProject,
    models,
    hardware,
    refreshModels,
    downloadModel,
    deleteModel,
    loadModel,
    unloadModel,
    selectModel,
    setDevice,
    cancelTranscription,
    qwenModel,
    downloadQwen,
    deleteQwen,
    demucsModel,
    downloadDemucs,
    deleteDemucs,
    voxcpmModel,
    downloadVoxcpm,
    deleteVoxcpm,
    nllbModel,
    downloadNllb,
    deleteNllb,
    genderModel,
    downloadGender,
    deleteGender,
    activeTranslateEngine,
    activeSeparateEngine,
    cudaPack,
    downloadCudaPack,
    deleteCudaPack,
    patchSegmentLocal,
    updateSegment,
    splitSegment,
    mergeSegmentWithNext,
    deleteSegment,
    providerKeys,
    voiceCapabilities,
    refreshVoiceConfig,
    setProviderKey,
    removeProviderKey,
    testProvider,
    previewLine,
    characters,
    refreshCharacters,
    saveCharacters,
    updateDubAudioSegments,
    canUndoOverlaps,
    setCanUndoOverlaps,
  } = useStudioStore();
  const [importedFile, setImportedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState(() => localStorage.getItem('dubvibe-target-language') ?? 'Khmer');
  const [selectedVoice, setSelectedVoice] = useState(() => localStorage.getItem('dubvibe-voice') ?? 'Male Voice A');
  const [chunkSelection, setChunkSelection] = useState('2');
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null);
  const [subtitleSync, setSubtitleSync] = useState(() => localStorage.getItem('dubvibe-subtitle-sync') !== 'false');
  const [burnSubtitles, setBurnSubtitles] = useState(() => localStorage.getItem('dubvibe-burn-subtitles') === 'true');
  const [preserveOriginalAudio, setPreserveOriginalAudio] = useState(() => localStorage.getItem('dubvibe-preserve-audio') !== 'false');
  const [originalAudioVolume, setOriginalAudioVolume] = useState(() => Number(localStorage.getItem('dubvibe-original-volume') ?? '0.25'));
  const [exportFiles, setExportFiles] = useState<{ subtitlesUrl: string | null; videoUrl: string | null } | null>(null);
  const [exportMode, setExportMode] = useState(() => localStorage.getItem('dubvibe-export-mode') ?? 'both');
  const [exportFolderInput, setExportFolderInput] = useState(() => localStorage.getItem('dubvibe-export-folder') ?? '');
  const [pickedFolder, setPickedFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [autoDuck, setAutoDuck] = useState(() => localStorage.getItem('dubvibe-auto-duck') !== 'false');
  const [keepMusic, setKeepMusic] = useState(() => localStorage.getItem('dubvibe-keep-music') !== 'false');
  const [keepSfx, setKeepSfx] = useState(() => localStorage.getItem('dubvibe-keep-sfx') !== 'false');
  const [platformStyle, setPlatformStyle] = useState(() => localStorage.getItem('dubvibe-platform-style') ?? 'original');
  const [platformFit, setPlatformFit] = useState(() => localStorage.getItem('dubvibe-platform-fit') ?? 'blur');
  const [showModels, setShowModels] = useState(false);
  const [modelBusy, setModelBusy] = useState<string | null>(null);
  const [nllbConfirm, setNllbConfirm] = useState(false);
  const [voxcpmConfirm, setVoxcpmConfirm] = useState(false);
  const [genderConfirm, setGenderConfirm] = useState(false);
  const [transcribeLanguage, setTranscribeLanguage] = useState(() => localStorage.getItem('dubvibe-transcribe-language') ?? '');
  const [beamSize, setBeamSize] = useState(() => Number(localStorage.getItem('dubvibe-beam-size') ?? 5));
  const [deleteTarget, setDeleteTarget] = useState<ProjectSummary | null>(null);
  const [showClearAllConfirm, setShowClearAllConfirm] = useState(false);
  const [showApiKeys, setShowApiKeys] = useState(false);
  const [geminiKeyInput, setGeminiKeyInput] = useState('');
  const [openaiKeyInput, setOpenaiKeyInput] = useState('');
  const [codecraftKeyInput, setCodecraftKeyInput] = useState('');
  const [keyTestMessage, setKeyTestMessage] = useState('');
  const [keyBusy, setKeyBusy] = useState<string | null>(null);
  const [ttsProvider, setTtsProvider] = useState(() => localStorage.getItem('dubvibe-tts-provider') ?? 'auto');
  const [translationProvider, setTranslationProvider] = useState(() => localStorage.getItem('dubvibe-translation-provider') ?? 'auto');
  const [ttsModel, setTtsModel] = useState(() => localStorage.getItem('dubvibe-tts-model') ?? '');
  const [geminiModels, setGeminiModels] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('dubvibe-gemini-models') ?? '[]'); } catch { return []; }
  });
  const [voiceEmotion, setVoiceEmotion] = useState(() => localStorage.getItem('dubvibe-emotion') ?? 'Auto');
  const [voiceIntensity, setVoiceIntensity] = useState(() => Number(localStorage.getItem('dubvibe-voice-intensity') ?? 70));
  const [voiceSpeed, setVoiceSpeed] = useState(() => {
    const saved = Number(localStorage.getItem('dubvibe-voice-speed') ?? 1);
    return Number.isFinite(saved) && saved >= 0.5 && saved <= 1 ? saved : 1;
  });
  const [actingEnabled, setActingEnabled] = useState(() => localStorage.getItem('dubvibe-acting') !== 'false');
  const [openaiActing, setOpenaiActing] = useState(() => localStorage.getItem('dubvibe-openai-acting') === 'true');
  const [openaiReview, setOpenaiReview] = useState(() => localStorage.getItem('dubvibe-openai-review') === 'true');
  const [previewText, setPreviewText] = useState('');
  const [previewAudio, setPreviewAudio] = useState('');
  const [previewBusy, setPreviewBusy] = useState(false);
  const [charEdits, setCharEdits] = useState<Record<string, { provider?: string; voice?: string }>>({});
  const transcribing = workflowSteps.some((step) => step.id === 'transcribe' && step.status === 'Active');

  // EDIT MODE: timeline segment editing
  const [editingSegmentId, setEditingSegmentId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<{ text: string; translation: string; start: number; end: number } | null>(null);
  const [cutBusy, setCutBusy] = useState(false);

  // Auto-Fix Speech Overlaps state
  const [fixOverlapsState, setFixOverlapsState] = useState<'idle' | 'analyzing' | 'preview' | 'applying'>('idle');
  const [fixProposals, setFixProposals] = useState<OverlapProposal[]>([]);
  const [fixError, setFixError] = useState<string | null>(null);
    const [fixOverlapsFound, setFixOverlapsFound] = useState(0);

  const openSegmentEditor = (segmentId: string) => {
    const fresh = useStudioStore.getState().transcript.find((item) => item.id === segmentId);
    setEditingSegmentId(segmentId);
    if (fresh) setEditDraft({ text: fresh.text, translation: fresh.translation, start: fresh.start, end: fresh.end });
  };

  const handleSaveSegmentEdit = async () => {
    if (!editingSegmentId || !editDraft) return;
    try {
      await updateSegment(project.id, editingSegmentId, { ...editDraft });
      setNotice('Segment saved.');
      setEditingSegmentId(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not save the segment.');
    }
  };

  const handleSplitSegmentEdit = async () => {
    if (!editingSegmentId || !editDraft) return;
    try {
      await splitSegment(project.id, editingSegmentId, (editDraft.start + editDraft.end) / 2);
      setNotice('Segment split. Re-dub to generate audio for both halves.');
      setEditingSegmentId(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not split the segment.');
    }
  };

  const handleMergeSegmentEdit = async () => {
    if (!editingSegmentId) return;
    try {
      await mergeSegmentWithNext(project.id, editingSegmentId);
      setNotice('Segments merged. Re-dub to regenerate the audio.');
      setEditingSegmentId(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not merge the segments.');
    }
  };

  const handleDeleteSegmentEdit = async () => {
    if (!editingSegmentId) return;
    try {
      await deleteSegment(project.id, editingSegmentId);
      setNotice('Segment deleted. Re-dub any affected lines.');
      setEditingSegmentId(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not delete the segment.');
    }
  };

  const editingSegment = editingSegmentId ? transcript.find((segment) => segment.id === editingSegmentId) : null;
  const canMergeEditing = editingSegment
    ? transcript.findIndex((segment) => segment.id === editingSegment.id) + 1 < transcript.length
    : false;

  // CapCut Timeline States
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dubbedTime, setDubbedTime] = useState(0);
  const [dubbedDuration, setDubbedDuration] = useState(0);
  const [dubbedPlaying, setDubbedPlaying] = useState(false);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const dubAudioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());
  const logScrollRef = useRef<HTMLDivElement>(null);
  const [trackStates, setTrackStates] = useState<Record<string, { muted: boolean; hidden: boolean; locked: boolean }>>({
    'Video': { muted: false, hidden: false, locked: false },
    'Original Speech': { muted: false, hidden: false, locked: false },
    'Translation': { muted: false, hidden: false, locked: false },
    'Dubbed Voice': { muted: false, hidden: false, locked: false },
    'Subtitle': { muted: false, hidden: false, locked: false },
  });

  const projectDurationSeconds = useMemo(() => {
    if (videoDuration && videoDuration > 0) return videoDuration;
    const [hours, minutes, seconds] = project.duration.split(':').map(Number);
    const parsed = (hours || 0) * 3600 + (minutes || 0) * 60 + (seconds || 0);
    const maxEnd = Math.max(0, ...transcript.map((s) => s.end), ...dubAudioSegments.map((s) => s.end));
    return Math.max(1, parsed, maxEnd);
  }, [videoDuration, project.duration, transcript, dubAudioSegments]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    setNotice('Refreshing studio state, tools, and timeline...');
    try {
      const healthRes = await fetch(`${apiBaseUrl}/health`).catch(() => null);
      setApiAvailable(healthRes?.ok ?? false);
      await hydrateStudio();
      await refreshVoiceConfig();
      setNotice('Studio, AI engines, and timeline refreshed successfully.');
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Refresh failed.');
    } finally {
      setTimeout(() => setIsRefreshing(false), 600);
    }
  };

  const seekTo = (time: number) => {
    const clamped = Math.max(0, Math.min(time, projectDurationSeconds));
    setCurrentTime(clamped);
    if (videoRef.current) {
      videoRef.current.currentTime = clamped;
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) {
      setNotice('Import a video first to play.');
      return;
    }
    if (videoRef.current.paused) {
      videoRef.current.play().catch(console.error);
    } else {
      videoRef.current.pause();
    }
  };

  const toggleDubbedPlay = () => {
    const video = dubbedVideoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play().catch(() => { });
    } else {
      video.pause();
    }
  };

  const skipVideo = (ref: React.RefObject<HTMLVideoElement | null>, delta: number) => {
    const video = ref.current;
    if (!video) return;
    const duration = video.duration || 0;
    video.currentTime = Math.max(0, Math.min(duration, video.currentTime + delta));
  };

  const formatTimecode = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
  };

  const toggleTrackMute = (trackName: string) => {
    setTrackStates((prev) => ({
      ...prev,
      [trackName]: { ...prev[trackName], muted: !prev[trackName]?.muted },
    }));
  };

  const toggleTrackHide = (trackName: string) => {
    setTrackStates((prev) => ({
      ...prev,
      [trackName]: { ...prev[trackName], hidden: !prev[trackName]?.hidden },
    }));
  };

  const toggleTrackLock = (trackName: string) => {
    setTrackStates((prev) => ({
      ...prev,
      [trackName]: { ...prev[trackName], locked: !prev[trackName]?.locked },
    }));
  };

  const folderPickerSupported = typeof (window as { showDirectoryPicker?: unknown }).showDirectoryPicker === 'function';

  const handlePickFolder = async () => {
    try {
      const picker = (window as unknown as {
        showDirectoryPicker: (options?: { mode?: 'read' | 'readwrite' }) => Promise<FileSystemDirectoryHandle>;
      }).showDirectoryPicker;
      const handle = await picker.call(window, { mode: 'readwrite' });
      setPickedFolder(handle);
      setNotice(`Export folder selected: "${handle.name}". Press Export to write the files there.`);
    } catch {
      // User cancelled the dialog.
    }
  };

  const writeFilesToFolder = async (handle: FileSystemDirectoryHandle, files: Array<{ url: string; name: string }>) => {
    for (const file of files) {
      const response = await fetch(file.url);
      if (!response.ok) throw new Error(`Could not read ${file.name} from the local service.`);
      const writable = await (await handle.getFileHandle(file.name, { create: true })).createWritable();
      await writable.write(await response.blob());
      await writable.close();
    }
  };

  const stopDubAudio = () => {
    dubAudioRefs.current.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });
  };

  const clearDubAudio = () => {
    stopDubAudio();
    dubAudioRefs.current.clear();
  };

  const syncDubAudio = () => {
    const video = videoRef.current;
    if (!video || video.paused || trackStates['Dubbed Voice']?.muted) {
      stopDubAudio();
      return;
    }
    const time = video.currentTime;
    const activeIndex = dubAudioSegments.findIndex((segment, index) => {
      const clip = dubAudioRefs.current.get(`${project.id}:${index}`);
      const spoken = (clip && Number.isFinite(clip.duration) && clip.duration > 0)
        ? clip.duration
        : Math.max(segment.end - segment.start, 0.25);
      return time >= segment.start && time < segment.start + spoken;
    });
    const activeKey = activeIndex === -1 ? '' : `${project.id}:${activeIndex}`;
    dubAudioRefs.current.forEach((audio, key) => {
      if (key !== activeKey) {
        audio.pause();
      }
    });
    if (activeIndex === -1 || !dubAudioSegments[activeIndex].path) {
      return;
    }
    const key = activeKey;
    let audio = dubAudioRefs.current.get(key);
    if (!audio) {
      // The version query busts the cached clip after a line is regenerated,
      // an overlap fix is applied/undone, or an export re-fits the audio.
      audio = new Audio(`${apiBaseUrl}/api/projects/${project.id}/dub-audio/${activeIndex + 1}?v=${dubAudioVersion}`);
      audio.preload = 'auto';
      dubAudioRefs.current.set(key, audio);
    }
    const offset = Math.max(0, time - dubAudioSegments[activeIndex].start);
    if (audio.paused) {
      audio.currentTime = offset;
      void audio.play().catch(() => { });
    } else if (Math.abs(audio.currentTime - offset) > 0.25) {
      // Seeking inside the same line leaves the clip running from its old position.
      audio.currentTime = offset;
    }
  };

  // Drop cached dub audio elements when the project or its dub track changes.
  useEffect(() => {
    clearDubAudio();
  }, [project.id, dubAudioSegments.length, dubAudioVersion]);

  const handleCutRange = async (range: { start: number; end: number }, cutMode: 'remove' | 'keep'): Promise<boolean> => {
    if (!range || !project.id) return false;
    setCutBusy(true);
    setNotice(cutMode === 'remove' ? 'Cutting the selected range from the video…' : 'Trimming the video to the selected range…');
    try {
      const result = await cutRange(range.start, range.end, cutMode);
      setImportedFile(null); // preview falls back to the re-written project media
      seekTo(0);
      setNotice(result.message ?? 'Cut applied.');
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Cutting failed.');
      return false;
    } finally {
      setCutBusy(false);
    }
  };

  const handleFixOverlaps = async () => {
    if (!project.id) return;
    setFixOverlapsState('analyzing');
    setFixError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/api/projects/${project.id}/fix-overlaps`, {
        method: 'POST',
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(err.detail ?? `Fix-overlaps failed (HTTP ${response.status})`);
      }
      const data = await response.json() as FixOverlapsResponse;
      setFixProposals(data.proposals);
      setFixOverlapsState('preview');
    } catch (e) {
      setFixError(e instanceof Error ? e.message : 'Unknown error');
      setFixOverlapsState('idle');
    }
  };

  const handleApplyFix = async () => {
    if (!project.id) return;
    setFixOverlapsState('applying');
    try {
      const response = await fetch(`${apiBaseUrl}/api/projects/${project.id}/fix-overlaps/apply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposals: fixProposals }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(err.detail ?? `Apply fix failed (HTTP ${response.status})`);
      }
      const data = await response.json() as {
        applied?: number;
        can_undo?: boolean;
        audio_segments: Array<Record<string, unknown>>;
      };
      updateDubAudioSegments(toDubSegments(data.audio_segments));
      setFixProposals([]);
      // Only the proposals that were actually applied count as fixed.
      setFixOverlapsFound(Number(data.applied ?? 0));
      setCanUndoOverlaps(Boolean(data.can_undo));
      setFixOverlapsState('idle');
    } catch (e) {
      setFixError(e instanceof Error ? e.message : 'Unknown error');
      setFixOverlapsState('idle');
    }
  };

  const handleUndoFix = async () => {
    if (!project.id) return;
    try {
      // The backend kept a copy of the clips it overwrote; restoring them is the
      // only way back, because applying a fix replaces the audio in place.
      const response = await fetch(`${apiBaseUrl}/api/projects/${project.id}/fix-overlaps/undo`, {
        method: 'POST',
      });
      const data = await response.json().catch(() => ({})) as {
        detail?: string;
        audio_segments?: Array<Record<string, unknown>>;
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `Undo failed (HTTP ${response.status})`);
      }
      updateDubAudioSegments(toDubSegments(data.audio_segments ?? []));
      setCanUndoOverlaps(false);
      setFixOverlapsFound(0);
      setNotice('Overlap fixes undone.');
    } catch (e) {
      setFixError(e instanceof Error ? e.message : 'Unknown error');
    }
  };

  useEffect(() => {
    if (!isPlaying) return;
    // Keep the displayed timecode responsive without re-rendering the studio
    // at 60 fps: the timeline paints its own playhead from the video element.
    const interval = window.setInterval(() => {
      const video = videoRef.current;
      if (video) setCurrentTime(video.currentTime);
    }, 120);
    return () => window.clearInterval(interval);
  }, [isPlaying]);

  useEffect(() => {
    if (apiAvailable) void hydrateStudio();
  }, [apiAvailable, hydrateStudio]);

  useEffect(() => {
    let mounted = true;
    const checkApi = () => {
      // Relative in dev (goes through the Vite proxy, so any port works);
      // absolute sidecar address in the packaged app.
      void fetch(`${apiBaseUrl}/health`)
        .then((response) => { if (mounted) setApiAvailable(response.ok); })
        .catch(() => { if (mounted) setApiAvailable(false); });
    };
    checkApi();
    const interval = window.setInterval(checkApi, 5000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!importedFile) {
      setVideoUrl('');
      return;
    }
    const url = URL.createObjectURL(importedFile);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [importedFile]);

  useEffect(() => () => {
    if (exportFiles?.subtitlesUrl?.startsWith('blob:')) URL.revokeObjectURL(exportFiles.subtitlesUrl);
  }, [exportFiles]);

  const hasMedia = Boolean(importedFile || project.hasMedia);
  // mediaVersion busts the browser cache after a cut rewrites the media file.
  const previewUrl = videoUrl || (project.hasMedia && project.id
    ? `${apiBaseUrl}/api/projects/${project.id}/media?v=${mediaVersion}`
    : '');

  const dubbedPreviewUrl = exportFiles?.videoUrl ?? (dubbedVideoReady && project.id
    ? `${apiBaseUrl}/api/projects/${project.id}/export/video`
    : '');

  // Keep the processing log scrolled to the newest entry.
  useEffect(() => {
    const element = logScrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [processingLog.length]);

  const handleClearProject = () => {
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl);
    }
    if (exportFiles?.subtitlesUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(exportFiles.subtitlesUrl);
    }
    setImportedFile(null);
    setVideoUrl('');
    setExportFiles(null);
    setCurrentTime(0);
    setVideoDuration(0);
    setIsPlaying(false);
    setSelectedSegmentId(null);
    clearDubAudio();
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    clearProject();
    setNotice('Project reset. Ready to import a new video.');
  };

  const handleImport = async (file?: File) => {
    if (!file) return;
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl);
    }
    if (exportFiles?.subtitlesUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(exportFiles.subtitlesUrl);
    }
    setCurrentTime(0);
    setVideoDuration(0);
    setIsPlaying(false);
    setSelectedSegmentId(null);
    setExportFiles(null);
    clearDubAudio();
    setImportedFile(file);
    setProjectName(file.name.replace(/\.[^.]+$/, ''));
    setNotice(`Uploading ${file.name} to the local processing service...`);
    try {
      // Video import only exists in EDIT and DUB.
      const projectMode = mode === 'EDIT' ? 'EDIT' : 'DUB';
      const created = await createProject(file.name.replace(/\.[^.]+$/, ''), projectMode, targetLanguage);
      await uploadMedia(created.id, file);
      setNotice(`${file.name} imported. Ready to transcribe.`);
    } catch (error) {
      console.warn('Project API unavailable', error);
      setNotice(error instanceof Error ? error.message : 'Could not import the video into the local project.');
    }
  };

  const handleExport = async () => {
    if (!hasMedia) {
      setNotice('Import a video before exporting.');
      fileInputRef.current?.click();
      return;
    }
    setBusy(true);
    try {
      const exported = await runExport({
        includeSubtitles: subtitleSync,
        burnSubtitles: burnSubtitles && subtitleSync,
        preserveOriginalAudio,
        originalAudioVolume,
        exportMode: (exportMode as 'single' | 'parts' | 'both'),
        exportFolder: pickedFolder ? null : (exportFolderInput.trim() || defaultExportFolder || null),
        deliverToFolder: !pickedFolder,
        autoDuck,
        keepMusic,
        keepSfx,
        platformStyle: (platformStyle as 'original' | 'vertical' | 'standard' | 'square'),
        platformFit: (platformFit as 'blur' | 'crop'),
      });
      setExportFiles({
        subtitlesUrl: exported.vtt
          ? URL.createObjectURL(new Blob([exported.vtt], { type: 'text/vtt' }))
          : null,
        videoUrl: exported.videoUrl,
      });
      const produced: string[] = [];
      if (exported.videoUrl) produced.push('the combined dubbed MP4');
      if (exported.parts.length) produced.push(`${exported.parts.length} part files`);
      if (exported.vtt) produced.push('the subtitle file');
      if (pickedFolder) {
        // Deliver the generated files straight into the folder the user picked.
        const files: Array<{ url: string; name: string }> = [];
        if (exported.videoUrl) files.push({ url: exported.videoUrl, name: 'dubbed-video.mp4' });
        if (exported.vtt) files.push({ url: `${apiBaseUrl}/api/projects/${project.id}/export/subtitle`, name: 'dubbed-subtitles.vtt' });
        for (const part of exported.parts) {
          files.push({ url: `${apiBaseUrl}/api/projects/${project.id}/export/parts/${part.index}`, name: part.name });
        }
        try {
          await writeFilesToFolder(pickedFolder, files);
          setNotice(`Export complete — ${files.length} file(s) saved to the selected folder "${pickedFolder.name}".`);
        } catch (writeError) {
          setNotice(writeError instanceof Error
            ? `${writeError.message} The files are still available in the export list above.`
            : 'Could not write the files to the selected folder.');
        }
      } else if (exported.folder) {
        setNotice(`Export complete: ${produced.length ? produced.join(', ') : 'no files'} saved to ${exported.folder}`);
      } else {
        setNotice(`Export complete from ${importedFile?.name ?? 'the imported video'}: ${produced.length ? produced.join(', ') : 'nothing — run Dub first'}.`);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Subtitle export failed.');
    } finally {
      setBusy(false);
    }
  };

  const handleWorkflow = async (action: string) => {
    if (!hasMedia) {
      setNotice('Import a video before running this step.');
      return;
    }
    if (action === 'Subtitle') {
      await handleExport();
      return;
    }
    const workflow = action === 'Transcribe' ? 'transcribe' : action === 'Translate' ? 'translate' : 'tts';
    setBusy(true);
    try {
      await runWorkflowStep(workflow, undefined, {
        targetLanguage,
        voice: selectedVoice,
        transcribeLanguage: transcribeLanguage || undefined,
        beamSize,
        translationProvider,
        ttsProvider,
        ttsModel: ttsModel || undefined,
        emotion: voiceEmotion,
        intensity: voiceIntensity,
        acting: actingEnabled,
        openaiActing,
        openaiReview,
      });
      setNotice(`${action} complete.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `${action} failed.`);
    } finally {
      setBusy(false);
    }
  };

  const handleChunk = async () => {
    const seconds = chunkSelection ? Number(chunkSelection) * 60 : null;
    if (!seconds && videoChunks.length === 0) {
      setNotice('Pick a part length (for example 2 minutes) before splitting.');
      return;
    }
    setBusy(true);
    try {
      const payload = await runChunk(seconds);
      setNotice(payload?.message ?? (seconds ? 'Video split into parts.' : 'Video split removed.'));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not split the video.');
    } finally {
      setBusy(false);
    }
  };

  const handleSeparate = async () => {
    setBusy(true);
    try {
      const payload = await runSeparate();
      setNotice(payload?.message ?? 'Audio separated: original voice removed from the export bed.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not separate the audio.');
    } finally {
      setBusy(false);
    }
  };

  const runModelAction = async (key: string, action: () => Promise<void>) => {
    setModelBusy(key);
    try {
      await action();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Model action failed.');
    } finally {
      setModelBusy(null);
    }
  };

  // Poll model status so download/load progress shows up everywhere — the
  // manager and Provider health. Faster while the manager is open.
  useEffect(() => {
    void refreshModels();
    const interval = window.setInterval(() => void refreshModels(), showModels ? 1500 : 3000);
    return () => window.clearInterval(interval);
  }, [showModels]);

  // Provider health live state: model downloads in progress + the pipeline
  // stage currently running, mapped to the provider rows that do the work.
  const activeStep = workflowSteps.find((step) => step.status === 'Active');
  const workingProviderIds = (() => {
    // Audio separation is not a workflow step; the backend announces its engine directly.
    if (activeSeparateEngine === 'demucs') return new Set(['demucs']);
    if (!activeStep) return new Set<string>();
    if (activeStep.id === 'transcribe') return new Set(['whisper']);
    if (activeStep.id === 'translate') {
      // The running engine is announced by the backend at the start of the run.
      // Use it when present; otherwise show the engine the user chose.
      const engineRows: Record<string, string> = {
        nllb: 'nllb',
        qwen: 'qwen',
        argos: 'argos_translate',
        google: 'google_translate',
        gemini: 'gemini',
        openai: 'openai',
        codecraft: 'codecraft',
      };
      if (activeTranslateEngine && engineRows[activeTranslateEngine]) {
        return new Set([engineRows[activeTranslateEngine]]);
      }
      if (translationProvider in engineRows) return new Set([engineRows[translationProvider]]);
      return new Set<string>();
    }
    if (activeStep.id === 'voice') {
      if (ttsProvider === 'edge') return new Set(['edge_tts']);
      if (ttsProvider === 'voxcpm2') return new Set(['voxcpm2']);
      if (ttsProvider === 'sapi5') return new Set(['pyttsx3']);
      if (ttsProvider === 'gemini' || ttsProvider === 'openai') return new Set<string>();
      return new Set(['edge_tts', 'pyttsx3']);
    }
    return new Set<string>();
  })();
  const downloadingModel = (providerId: string) => {
    if (providerId === 'whisper') {
      return models.find((model) => model.status === 'downloading' || model.status === 'validating') ?? null;
    }
    if (providerId === 'qwen') {
      return qwenModel && (qwenModel.status === 'downloading' || qwenModel.status === 'validating') ? qwenModel : null;
    }
    if (providerId === 'nllb') {
      return nllbModel && (nllbModel.status === 'downloading' || nllbModel.status === 'validating') ? nllbModel : null;
    }
    if (providerId === 'demucs') {
      return demucsModel && (demucsModel.status === 'downloading' || demucsModel.status === 'validating') ? demucsModel : null;
    }
    return null;
  };

  // Per-chunk progress for the Video parts card above the timeline: how many
  // of each chunk's speech segments have dubbed audio already. During the
  // dubbing (voice) step the parts light up one by one, estimated from the
  // live step progress until real per-line counts arrive.
  const partStats = useMemo(() => {
    const stats = videoChunks.map((chunk) => {
      const start = chunk.start;
      const end = chunk.start + chunk.duration;
      const inRange = (segment: { start: number; end: number }) =>
        segment.start < end && (segment.end || segment.start) > start;
      const speech = transcript.filter(inRange).length;
      const dubbed = dubAudioSegments.filter(inRange).length;
      return { index: chunk.index, start, end, speech, dubbed, cumStart: 0, cumEnd: 0 };
    });
    let cumulative = 0;
    for (const stat of stats) {
      stat.cumStart = cumulative;
      cumulative += stat.speech;
      stat.cumEnd = cumulative;
    }
    return { stats, totalSpeech: cumulative };
  }, [videoChunks, transcript, dubAudioSegments]);

  const activeVoiceProgress = activeStep?.id === 'voice' ? activeStep.progress : null;
  const dubEstimateLines = activeVoiceProgress !== null && partStats.totalSpeech > 0
    ? Math.min(partStats.totalSpeech, Math.floor((activeVoiceProgress / 100) * partStats.totalSpeech))
    : null;

  const partState = (stat: { speech: number; dubbed: number; cumStart: number; cumEnd: number }): 'complete' | 'processing' | 'pending' | 'no-speech' => {
    if (stat.speech === 0) return 'no-speech';
    if (stat.dubbed >= stat.speech) return 'complete';
    if (dubEstimateLines !== null) {
      if (stat.cumEnd <= dubEstimateLines) return 'complete';
      if (stat.cumStart < dubEstimateLines) return 'processing';
      return 'pending';
    }
    return stat.dubbed > 0 ? 'processing' : 'pending';
  };

  const partProgress = (stat: { speech: number; dubbed: number; cumStart: number; cumEnd: number }) => {
    const state = partState(stat);
    if (state === 'complete') return 100;
    if (state === 'no-speech' || state === 'pending') return 0;
    if (dubEstimateLines !== null && stat.speech > 0) {
      return Math.min(99, Math.max(5, Math.round(((dubEstimateLines - stat.cumStart) / stat.speech) * 100)));
    }
    return stat.speech > 0 ? Math.round((stat.dubbed / stat.speech) * 100) : 0;
  };

  const completedParts = partStats.stats.filter((stat) => partState(stat) === 'complete').length;

  // Voice provider config (keys status + capability tables). The desktop app
  // starts the Python sidecar in parallel with the UI, so a request issued on
  // mount can land before the API is up. Retry when the health check confirms
  // it is, and keep retrying while the capability table is still empty —
  // otherwise the Voice list stays at "Default voice" for every target
  // language (no Piseth/Sreymom for Khmer).
  useEffect(() => {
    if (apiAvailable !== true) return;
    void refreshVoiceConfig();
  }, [apiAvailable, refreshVoiceConfig]);

  const capabilitiesLoaded = Object.keys(voiceCapabilities).length > 0;
  useEffect(() => {
    if (apiAvailable !== true || capabilitiesLoaded) return;
    // Keep retrying until the table arrives: a single retry can land while the
    // sidecar is still warming up and then never run again.
    const timer = window.setInterval(() => void refreshVoiceConfig(), 5000);
    return () => window.clearInterval(timer);
  }, [apiAvailable, capabilitiesLoaded, refreshVoiceConfig]);
  // Character list follows the loaded project.
  useEffect(() => { void refreshCharacters(); }, [project.id, refreshCharacters]);

  const handleTestKey = async (provider: string) => {
    setKeyBusy(provider);
    setKeyTestMessage('');
    try {
      const result = await testProvider(provider);
      const voiceCount = result.models.length;
      const translationCount = result.translationModels.length;
      setKeyTestMessage(
        `${provider}: connected — ${voiceCount} TTS model${voiceCount === 1 ? '' : 's'} and `
        + `${translationCount} translation model${translationCount === 1 ? '' : 's'} available`,
      );
      if (provider === 'gemini') {
        setGeminiModels(result.models);
        localStorage.setItem('dubvibe-gemini-models', JSON.stringify(result.models));
      }
    } catch (error) {
      setKeyTestMessage(`${provider}: ${error instanceof Error ? error.message : 'connection failed'}`);
    } finally {
      setKeyBusy(null);
    }
  };

  const handleSaveKey = async (provider: string, key: string) => {
    if (!key.trim()) { setKeyTestMessage(`${provider}: enter an API key first`); return; }
    setKeyBusy(provider);
    setKeyTestMessage('');
    try {
      await setProviderKey(provider, key.trim());
      await handleTestKey(provider);
    } catch (error) {
      setKeyTestMessage(`${provider}: ${error instanceof Error ? error.message : 'could not store key'}`);
    } finally {
      setKeyBusy(null);
    }
  };

  const handlePreviewLine = async () => {
    const fallbackLine = transcript.find((segment) => segment.translation)?.translation ?? transcript[0]?.text ?? '';
    const text = previewText.trim() || fallbackLine;
    if (!text) { setNotice('Enter a line to preview (or transcribe a video first).'); return; }
    setPreviewBusy(true);
    try {
      const audio = await previewLine({
        text,
        // Preview the engine that will actually run, with the voice picked for it.
        provider: effectiveVoiceEngine,
        voice: selectedVoice === 'mixed' || selectedVoice === 'auto' ? undefined : selectedVoice,
        model: ttsModel || undefined,
        emotion: voiceEmotion,
        intensity: voiceIntensity,
        speed: voiceSpeed,
        language: targetLanguage,
      });
      setPreviewAudio(audio);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Preview failed.');
    } finally {
      setPreviewBusy(false);
    }
  };

  const languageCodeOf = (language: string): string =>
    TARGET_LANGUAGES.find((entry) => entry.name === language)?.code ?? language.toLowerCase().slice(0, 2);

  const engineHasCredentials = (engine: string): boolean => {
    if (engine === 'gemini') return providerKeys.gemini.has_key;
    if (engine === 'openai') return providerKeys.openai.has_key;
    return true;
  };

  const engineSupportsLanguage = (engine: string, language: string): boolean => {
    const caps = voiceCapabilities[engine];
    if (!caps) return engine === 'edge' || engine === 'sapi5';
    return caps.languages.includes(languageCodeOf(language));
  };

  // The engine that will actually synthesize this project: the explicit pick, or
  // the first usable engine in quality order (keys + model + language support).
  const effectiveVoiceEngine = useMemo(() => {
    if (ttsProvider !== 'auto') return ttsProvider;
    const usable = VOICE_ENGINE_ORDER.find((engine) => engineHasCredentials(engine)
      && engineSupportsLanguage(engine, targetLanguage)
      && !(engine === 'voxcpm2' && voiceCapabilities.voxcpm2?.installed === false));
    return usable ?? 'edge';
  }, [ttsProvider, targetLanguage, providerKeys, voiceCapabilities]);

  /** Voices the engine really has for this language — never a hardcoded list. */
  const voiceOptionsFor = (engine: string, language: string): Array<{ id: string; label: string }> => {
    const resolved = engine === 'auto' ? effectiveVoiceEngine : engine;
    if (resolved === 'gemini' || resolved === 'openai') {
      return (voiceCapabilities[resolved]?.voices ?? []).map((voice) => ({ id: voice.id, label: `${voice.id} (${voice.style})` }));
    }
    if (resolved === 'edge') {
      const byLanguage = voiceCapabilities.edge?.voices_by_language ?? {};
      const voices = byLanguage[language] ?? [];
      if (voices.length > 0) return voices.map((voice) => ({ id: voice.id, label: voice.label }));
      return [{ id: 'auto', label: 'Default voice for this language' }];
    }
    if (resolved === 'sapi5') return [{ id: 'auto', label: 'System voice (offline)' }];
    return [{ id: 'auto', label: "The model's own voice (no choice)" }];
  };

  const selectableVoices = voiceOptionsFor(ttsProvider, targetLanguage);
  const canMixVoices = selectableVoices.length > 1;
  const voiceChoiceIsValid = selectedVoice === 'mixed'
    ? canMixVoices
    : selectableVoices.some((voice) => voice.id === selectedVoice);

  // Keep the saved voice valid when the engine or target language changes
  // (e.g. Edge + Khmer has no "Narrator" voice).
  useEffect(() => {
    if (Object.keys(voiceCapabilities).length === 0) return; // capabilities not loaded yet
    if (voiceChoiceIsValid) return;
    const fallback = selectableVoices[0]?.id ?? 'auto';
    setSelectedVoice(fallback);
    localStorage.setItem('dubvibe-voice', fallback);
  }, [voiceChoiceIsValid, selectableVoices, voiceCapabilities]);

  const voiceHint = useMemo(() => {
    const engine = VOICE_ENGINE_LABELS[effectiveVoiceEngine] ?? effectiveVoiceEngine;
    // Name the model that will actually run, so the voice list is explainable.
    const engineWithModel = ttsModel && (effectiveVoiceEngine === 'gemini' || effectiveVoiceEngine === 'openai')
      ? `${engine} · ${ttsModel}`
      : engine;
    if (selectableVoices.length === 0) {
      return `${engineWithModel} has no voices listed for ${targetLanguage}. Test the connection in API Settings.`;
    }
    if (selectableVoices.length === 1) {
      return `${engineWithModel} · ${targetLanguage} — this engine has no voice choice.`;
    }
    return `${selectableVoices.length} voice${selectableVoices.length === 1 ? '' : 's'} available · ${engineWithModel} · ${targetLanguage}`;
  }, [effectiveVoiceEngine, selectableVoices, targetLanguage, ttsModel]);

  const providerCaps = ttsProvider && ttsProvider !== 'auto' ? voiceCapabilities[ttsProvider] : null;
  // VoxCPM2 is verified for English only; picking it for another target would silently use Edge.
  const voxcpmUnavailable = voiceCapabilities['voxcpm2']?.installed === false || targetLanguage !== 'English';

  const handleStartDub = async () => {
    if (!hasMedia) {
      setNotice('Import a video before starting the dubbing workflow.');
      fileInputRef.current?.click();
      return;
    }
    setBusy(true);
    try {
      // Auto-cut long videos into parts before dubbing when chunking is on.
      if (chunkSelection && videoChunks.length === 0) {
        await runChunk(Number(chunkSelection) * 60);
      }
      const voiceOptions = { targetLanguage, voice: selectedVoice, transcribeLanguage: transcribeLanguage || undefined, beamSize, translationProvider, ttsProvider, ttsModel: ttsModel || undefined, emotion: voiceEmotion, intensity: voiceIntensity, speed: voiceSpeed, acting: actingEnabled, openaiActing, openaiReview };
      await runWorkflowStep('transcribe', undefined, voiceOptions);
      await runWorkflowStep('translate', undefined, voiceOptions);
      await runWorkflowStep('tts', undefined, voiceOptions);
      setNotice('Dubbing workflow complete. Export when ready.');
    } catch (error) {
      console.error('Unable to start dubbing pipeline', error);
      setNotice(error instanceof Error ? error.message : 'Unable to start dubbing pipeline.');
    } finally {
      setBusy(false);
    }
  };

  const handleToolAction = (tool: string) => {
    setActiveTool(tool);
    if (tool === 'Auto Dubbing') void handleStartDub();
    if (tool === 'Translate') void handleWorkflow('Translate');
    if (tool === 'Auto Subtitle') void handleExport();
  };

  const handleSave = () => {
    const state = useStudioStore.getState();
    localStorage.setItem('dubvibe-project', JSON.stringify({
      project: state.project,
      transcript: state.transcript,
      workflowSteps: state.workflowSteps,
    }));
    localStorage.setItem('dubvibe-project-id', state.project.id);
    setNotice('Project state saved in this browser. Imported media is not copied into the project.');
  };

  const resetWorkspaceUi = () => {
    if (videoUrl.startsWith('blob:')) URL.revokeObjectURL(videoUrl);
    if (exportFiles?.subtitlesUrl?.startsWith('blob:')) URL.revokeObjectURL(exportFiles.subtitlesUrl);
    setImportedFile(null);
    setVideoUrl('');
    setExportFiles(null);
    setCurrentTime(0);
    setVideoDuration(0);
    setIsPlaying(false);
    setSelectedSegmentId(null);
    clearDubAudio();
  };

  const handleSwitchProject = async (projectId: string) => {
    if (projectId === project.id) return;
    setBusy(true);
    try {
      resetWorkspaceUi();
      await switchProject(projectId);
      setNotice(`Loaded project "${useStudioStore.getState().project.name}".`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not load the project.');
    } finally {
      setBusy(false);
    }
  };

  const requestDeleteProject = (projectId: string) => {
    setDeleteTarget(projectList.find((item) => item.id === projectId) ?? null);
  };

  const handleDeleteProject = async () => {
    if (!deleteTarget) return;
    const { id, name } = deleteTarget;
    setDeleteTarget(null);
    setBusy(true);
    try {
      await deleteProject(id);
      setNotice(`Project "${name}" deleted.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not delete the project.');
    } finally {
      setBusy(false);
    }
  };

  const handleClearAllProjects = async () => {
    setShowClearAllConfirm(false);
    const count = projectList.length;
    setBusy(true);
    try {
      await clearAllProjects();
      setNotice(count > 0 ? `All ${count} projects deleted.` : 'Project history is already empty.');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not clear the project history.');
    } finally {
      setBusy(false);
    }
  };

  const handleTopAction = (label: string) => {
    if (label === 'New Project' || label === 'Clear') handleClearProject();
    if (label === 'Import' || label === 'Import Video') fileInputRef.current?.click();
    if (label === 'Refresh') void handleRefresh();
    if (label === 'Save') handleSave();
    if (label === 'Preview') togglePlay();
    if (label === 'Export') void handleExport();
  };

  const handleSaveCharacters = async () => {
    const merged: Record<string, Record<string, unknown>> = { ...characters.characters };
    for (const [name, edit] of Object.entries(charEdits)) {
      merged[name] = { ...(merged[name] ?? {}), ...edit };
    }
    try {
      await saveCharacters(merged);
      setCharEdits({});
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not save character voices.');
    }
  };

  const statusBadge = useMemo(() => {
    if (project.status === 'Processing') return 'bg-blue-500/20 text-blue-200 border border-blue-500/30';
    if (project.status === 'Exporting') return 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/30';
    return 'bg-slate-700/60 text-slate-200 border border-slate-600';
  }, [project.status]);

  const modeSwitcher = (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
      <div />
      <div className="flex items-center gap-2" role="tablist" aria-label="Studio mode">
        <button
          role="tab"
          aria-selected={mode === 'EDIT'}
          onClick={() => setMode('EDIT')}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${mode === 'EDIT' ? 'bg-slate-800 text-white ring-1 ring-white/20' : 'bg-slate-900 text-slate-400 hover:text-slate-200'}`}
        >
          EDIT MODE
        </button>
        <button
          role="tab"
          aria-selected={mode === 'DUB'}
          onClick={() => setMode('DUB')}
          className={`rounded-lg px-3 py-2 text-sm font-medium ${mode === 'DUB' ? 'bg-sky-500 text-slate-950' : 'bg-slate-900 text-slate-400 hover:text-slate-200'}`}
        >
          DUB MODE
        </button>
      </div>
      <div className="flex items-center justify-end gap-3">
        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900 px-3 py-1.5 text-sm text-slate-300">
          <Clock3 size={14} /> {project.duration}
        </span>
        <span className={`inline-flex items-center rounded-full px-3 py-1.5 text-xs font-semibold ${statusBadge}`}>
          {project.status}
        </span>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#050b14] text-slate-100">
      <header className="flex h-16 items-center justify-between border-b border-white/10 bg-slate-950/80 px-6 backdrop-blur-xl">
        <div className="flex items-center gap-4">
          <img src="/dubvibe-icon.png" alt="" className="h-10 w-10 rounded-xl object-cover shadow-glow" />
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-slate-400">DubVibe</p>
            <h1 className="text-lg font-semibold">AI Dubbing Studio</h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowModels(true)}
            title="AI Model Manager"
            className="flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-sm font-medium text-sky-200 transition hover:border-sky-500/60 hover:bg-sky-500/20"
          >
            <Cpu size={15} />
            AI Models
          </button>
          <button
            onClick={() => setShowApiKeys(true)}
            title="Online AI provider settings"
            className="flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm font-medium text-violet-200 transition hover:border-violet-500/60 hover:bg-violet-500/20"
          >
            API Keys
          </button>
          {topActions.map(({ label, icon: Icon }) => (
            <button key={label} onClick={() => handleTopAction(label)} disabled={busy} title={label} className="flex items-center gap-2 rounded-lg border border-white/10 bg-slate-900/80 px-3 py-2 text-sm text-slate-200 transition hover:border-sky-500/60 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40">
              <Icon size={15} />
              {label}
            </button>
          ))}
        </div>
        <input ref={fileInputRef} className="hidden" type="file" accept="video/*" onChange={(event) => { void handleImport(event.target.files?.[0]); event.currentTarget.value = ''; }} />
      </header>

      <main className="grid h-[calc(100vh-64px)] grid-cols-[220px_minmax(0,1fr)_340px]">
        <aside className="border-r border-white/10 bg-slate-950/80 p-4">
          <div className="mb-4">
            <p className="mb-2 text-[10px] uppercase tracking-[0.24em] text-slate-500">Tools</p>
            <div className="space-y-2">
              {toolActions.map((tool) => (
                <button
                  key={tool}
                  onClick={() => handleToolAction(tool)}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition ${activeTool === tool
                    ? 'border-sky-500/60 bg-sky-500/10 text-sky-100'
                    : 'border-white/5 bg-slate-900/80 text-slate-300 hover:border-slate-600 hover:text-white'
                    }`}
                >
                  <span>{tool}</span>
                  <Sparkles size={14} className="opacity-70" />
                </button>
              ))}
            </div>
            <div className="mt-4 pt-3 border-t border-white/10">
              <button
                onClick={() => void handleRefresh()}
                disabled={isRefreshing}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs font-medium text-sky-200 hover:bg-sky-500/20 hover:border-sky-500/50 transition disabled:opacity-50"
              >
                <RefreshCw size={13} className={isRefreshing ? 'animate-spin-fast' : ''} />
                <span>{isRefreshing ? 'Refreshing...' : 'Refresh Tools'}</span>
              </button>
            </div>
          </div>

          <div className="border-t border-white/10 pt-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-[10px] uppercase tracking-[0.24em] text-slate-500">Projects</p>
              <div className="flex items-center gap-1.5">
                {projectList.length > 0 && (
                  <button
                    onClick={() => setShowClearAllConfirm(true)}
                    disabled={busy}
                    title="Delete ALL projects"
                    className="text-slate-500 transition hover:text-rose-400 disabled:opacity-50"
                  >
                    <Trash2 size={11} />
                  </button>
                )}
                <button
                  onClick={() => void refreshProjects()}
                  disabled={isRefreshing}
                  title="Reload project list"
                  className="text-slate-500 hover:text-slate-300 transition disabled:opacity-50"
                >
                  <RefreshCw size={11} />
                </button>
              </div>
            </div>
            <div className="max-h-60 space-y-1 overflow-y-auto pr-0.5">
              {projectList.length === 0 && (
                <p className="text-xs text-slate-600">No projects yet</p>
              )}
              {projectList.map((item) => (
                <div
                  key={item.id}
                  className={`group flex items-center gap-1 rounded-lg border px-2 py-1.5 text-xs transition ${item.id === project.id
                    ? 'border-sky-500/40 bg-sky-500/10 text-sky-100'
                    : 'border-white/5 bg-slate-900/80 text-slate-300 hover:border-slate-600'
                    }`}
                >
                  <button
                    onClick={() => void handleSwitchProject(item.id)}
                    className="flex-1 truncate text-left"
                    title={`${item.name} · ${item.duration}`}
                  >
                    {item.name}
                  </button>
                  <button
                    onClick={() => requestDeleteProject(item.id)}
                    disabled={busy}
                    className="shrink-0 text-slate-500 opacity-0 transition hover:text-rose-400 group-hover:opacity-100 disabled:opacity-30"
                    title="Delete project"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          </div>

        </aside>

        <section className="flex min-w-0 flex-col border-r border-white/10 bg-slate-950/60">
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-5">
            {modeSwitcher}
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-900/80 p-4">
              <div className="flex items-center gap-4 min-w-0">
                <div className="rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 p-2 text-slate-950 shrink-0">
                  <Video size={20} />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Project</p>
                    {hasMedia && (
                      <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-400 border border-sky-500/20 font-medium">
                        Active Media
                      </span>
                    )}
                  </div>
                  <h2 className="text-xl font-semibold truncate max-w-[280px] lg:max-w-md">{project.name}</h2>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {hasMedia && (
                  <>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={busy}
                      title="Replace current video"
                      className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-800/80 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition disabled:opacity-40"
                    >
                      <FolderOpen size={13} />
                      <span>Change</span>
                    </button>
                    <button
                      onClick={handleClearProject}
                      disabled={busy}
                      title="Remove current video and reset project"
                      className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs font-medium text-rose-300 hover:bg-rose-500/20 hover:border-rose-500/50 transition disabled:opacity-40"
                    >
                      <Trash2 size={13} />
                      <span>Remove</span>
                    </button>
                  </>
                )}
                <button onClick={() => void handleWorkflow('Transcribe')} disabled={!hasMedia || busy} className="rounded-lg border border-white/10 bg-slate-800 px-3 py-2 text-sm text-slate-200 disabled:opacity-40">Analyze Audio</button>
                {transcribing && (
                  <button
                    onClick={() => void cancelTranscription()}
                    title="Stop the running transcription and keep the segments found so far"
                    className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-200 transition hover:bg-rose-500/20"
                  >
                    Cancel transcription
                  </button>
                )}
                <button
                  onClick={() => void handleStartDub()}
                  disabled={busy || apiAvailable === false || !hasMedia}
                  className="rounded-lg bg-gradient-to-r from-sky-500 to-violet-500 px-4 py-2 text-sm font-semibold text-white shadow-glow disabled:opacity-50"
                >
                  {busy ? 'PROCESSING...' : 'START DUBBING'}
                </button>
              </div>
            </div>

            {notice && <div role="status" className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">{notice}</div>}
            {(exportFiles || exportedParts.length > 0 || exportFolder) && <div className="flex flex-wrap items-center gap-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3">
              <span className="mr-auto text-sm text-emerald-100">Export files</span>
              {exportFiles?.subtitlesUrl && <a href={exportFiles.subtitlesUrl} download={`${project.name || 'dubvibe'}-subtitles.vtt`} className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-slate-950/70 px-3 py-2 text-sm text-white hover:border-emerald-400/60">
                <Download size={15} /> Subtitles (.vtt)
              </a>}
              {exportFiles?.videoUrl
                ? <a href={exportFiles.videoUrl} download={`${project.name || 'dubvibe'}-dubbed.mp4`} className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-slate-950/70 px-3 py-2 text-sm text-white hover:border-emerald-400/60">
                  <Video size={15} /> Combined video (.mp4)
                </a>
                : exportFiles && !exportedParts.length ? <span className="text-xs text-amber-100">Dub the translated segments to create an MP4.</span> : null}
              {exportedParts.map((part) => (
                <a
                  key={part.index}
                  href={`${apiBaseUrl}/api/projects/${project.id}/export/parts/${part.index}`}
                  download={part.name}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/15 bg-slate-950/70 px-2.5 py-2 text-xs text-white hover:border-emerald-400/60"
                >
                  <Download size={12} /> Part {part.index}
                </a>
              ))}
              {exportFolder && (
                <span className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-emerald-100" title={exportFolder}>
                  <FolderOpen size={12} className="shrink-0" /> Saved to: {exportFolder}
                </span>
              )}
            </div>}
            <div role="status" className={`rounded-lg border px-4 py-3 text-xs ${apiAvailable === false ? 'border-rose-500/30 bg-rose-500/10 text-rose-100' : 'border-amber-500/30 bg-amber-500/10 text-amber-100'}`}>
              {apiAvailable === null
                ? 'Connecting to the local processing service...'
                : apiAvailable === false
                  ? 'Local processing service is offline. Run npm run dev from the project folder. Video preview remains available.'
                  : 'Transcription runs locally via Faster-Whisper. Translation: offline (Argos) or online (Google). TTS: online (Edge) with offline fallback (SAPI5).'}
            </div>

            <div className="flex gap-3 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
              {actionButtons.map(({ label, icon: Icon }) => (
                <button key={label} onClick={() => void handleWorkflow(label)} disabled={busy || apiAvailable !== true || !hasMedia} className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-slate-200 hover:border-sky-500/50 disabled:cursor-not-allowed disabled:opacity-40">
                  <Icon size={15} /> {label}
                </button>
              ))}
            </div>

            <div className="grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
              <div className="flex min-w-0 flex-col gap-4">
                <div className="rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.12),_transparent_50%),rgba(15,23,42,0.9)] p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-slate-300">
                      <Play className="fill-current text-sky-400" size={15} />
                      Preview
                    </div>
                    <div className="flex items-center gap-2">
                      {hasMedia && (
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => fileInputRef.current?.click()}
                            title="Import a different video"
                            className="flex items-center gap-1 rounded border border-white/10 bg-slate-900/90 px-2 py-1 text-[11px] text-slate-300 hover:text-white hover:border-sky-500/40 transition"
                          >
                            <FolderOpen size={12} />
                            <span>Replace</span>
                          </button>
                          <button
                            onClick={handleClearProject}
                            title="Remove video and reset project"
                            className="flex items-center gap-1 rounded border border-rose-500/30 bg-rose-950/40 px-2 py-1 text-[11px] text-rose-300 hover:text-rose-100 hover:bg-rose-900/60 transition"
                          >
                            <Trash2 size={12} />
                            <span>Remove</span>
                          </button>
                        </div>
                      )}
                      <span className="text-xs uppercase tracking-[0.2em] text-slate-500">Source preview</span>
                    </div>
                  </div>

                  <div className="relative flex h-64 items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black">
                    {previewUrl ? (
                      <video
                        ref={videoRef}
                        src={previewUrl}
                        onTimeUpdate={(e) => { setCurrentTime(e.currentTarget.currentTime); syncDubAudio(); }}
                        onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration)}
                        onPlay={() => { setIsPlaying(true); syncDubAudio(); }}
                        onPause={() => { setIsPlaying(false); stopDubAudio(); }}
                        onSeeked={() => syncDubAudio()}
                        className="h-full w-full object-contain"
                      />
                    ) : (
                      <button onClick={() => fileInputRef.current?.click()} className="flex h-full w-full flex-col items-center justify-center gap-3 text-sm text-slate-400 hover:text-white group transition">
                        <div className="rounded-full border border-dashed border-white/20 p-4 group-hover:border-sky-500/50 group-hover:bg-sky-500/10 transition">
                          <FileVideo size={32} className="text-slate-400 group-hover:text-sky-400 transition" />
                        </div>
                        <div className="text-center">
                          <p className="font-medium text-slate-200 group-hover:text-sky-300">Click to import video</p>
                          <p className="text-xs text-slate-500 mt-0.5">MP4, MOV, MKV, WebM supported</p>
                        </div>
                      </button>
                    )}
                  </div>
                  {previewUrl && (
                    <div className="mt-3 flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900/80 px-3 py-2">
                      <button onClick={() => skipVideo(videoRef, -5)} title="Back 5 seconds" className="rounded p-1 text-slate-400 transition hover:text-white">
                        <SkipBack size={14} />
                      </button>
                      <button onClick={togglePlay} title={isPlaying ? 'Pause' : 'Play'} className="rounded-md bg-white/10 p-1.5 text-white transition hover:bg-white/20">
                        {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                      </button>
                      <button onClick={() => skipVideo(videoRef, 5)} title="Forward 5 seconds" className="rounded p-1 text-slate-400 transition hover:text-white">
                        <SkipForward size={14} />
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={videoDuration || 0}
                        step={0.1}
                        value={Math.min(currentTime, videoDuration || 0)}
                        onChange={(event) => seekTo(Number(event.target.value))}
                        title="Seek"
                        className="min-w-0 flex-1 accent-sky-500"
                      />
                      <span className="shrink-0 font-mono text-[10px] text-slate-400">
                        {formatTimecode(currentTime)} / {formatTimecode(videoDuration)}
                      </span>
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-slate-300">
                      <AudioLines size={15} className="text-emerald-400" />
                      Dubbed preview
                    </div>
                    <div className="flex items-center gap-2">
                      {dubbedPreviewUrl && (
                        <a
                          href={dubbedPreviewUrl}
                          download={`${project.name || 'dubvibe'}-dubbed.mp4`}
                          className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-800/80 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
                        >
                          <Download size={12} />
                          <span>Download</span>
                        </a>
                      )}
                      <span className="text-xs uppercase tracking-[0.2em] text-slate-500">Dubbed output</span>
                    </div>
                  </div>
                  <div className="relative flex h-64 items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-black">
                    {dubbedPreviewUrl ? (
                      <video
                        key={dubbedPreviewUrl}
                        ref={dubbedVideoRef}
                        src={dubbedPreviewUrl}
                        onTimeUpdate={(e) => setDubbedTime(e.currentTarget.currentTime)}
                        onLoadedMetadata={(e) => setDubbedDuration(e.currentTarget.duration || 0)}
                        onPlay={() => setDubbedPlaying(true)}
                        onPause={() => setDubbedPlaying(false)}
                        className="h-full w-full object-contain"
                      />
                    ) : (
                      <div className="flex flex-col items-center gap-3 text-center">
                        <div className="rounded-full border border-dashed border-white/15 p-4">
                          <AudioLines size={28} className="text-slate-600" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-400">No dubbed video yet</p>
                          <p className="mt-0.5 max-w-[280px] text-xs text-slate-600">
                            Run Dub, then Export — the dubbed MP4 plays here directly, no need to save the file first.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                  {dubbedPreviewUrl && (
                    <div className="mt-3 flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900/80 px-3 py-2">
                      <button onClick={() => skipVideo(dubbedVideoRef, -5)} title="Back 5 seconds" className="rounded p-1 text-slate-400 transition hover:text-white">
                        <SkipBack size={14} />
                      </button>
                      <button onClick={toggleDubbedPlay} title={dubbedPlaying ? 'Pause' : 'Play'} className="rounded-md bg-white/10 p-1.5 text-white transition hover:bg-white/20">
                        {dubbedPlaying ? <Pause size={14} /> : <Play size={14} />}
                      </button>
                      <button onClick={() => skipVideo(dubbedVideoRef, 5)} title="Forward 5 seconds" className="rounded p-1 text-slate-400 transition hover:text-white">
                        <SkipForward size={14} />
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={dubbedDuration || 0}
                        step={0.1}
                        value={Math.min(dubbedTime, dubbedDuration || 0)}
                        onChange={(event) => { if (dubbedVideoRef.current) dubbedVideoRef.current.currentTime = Number(event.target.value); }}
                        title="Seek"
                        className="min-w-0 flex-1 accent-emerald-500"
                      />
                      <span className="shrink-0 font-mono text-[10px] text-slate-400">
                        {formatTimecode(dubbedTime)} / {formatTimecode(dubbedDuration)}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">Workflow</p>
                  <FileVideo size={14} className="text-slate-400" />
                </div>

                <div className="space-y-3">
                  {workflowSteps.map((step) => (
                    <div key={step.id} className="rounded-xl border border-white/10 bg-slate-950/80 p-3">
                      <div className="mb-2 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {step.status === 'Completed' ? <CheckCircle2 size={14} className="text-emerald-400" /> : step.status === 'Active' ? <BadgeCheck size={14} className="text-sky-400" /> : <FolderOpen size={14} className="text-slate-400" />}
                          <span className="text-sm font-medium text-slate-200">{step.label}</span>
                        </div>
                        <span className="text-[10px] uppercase tracking-[0.14em] text-slate-400">{step.status}</span>
                      </div>
                      <div className="mb-2 h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${step.progress}%` }} />
                      </div>
                      <p className="text-xs text-slate-400">{step.id === 'subtitle' && !subtitleSync ? 'Subtitle sync disabled by user' : step.detail}</p>
                      {step.id === 'subtitle' && <label className="mt-3 flex cursor-pointer items-center gap-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          checked={subtitleSync}
                          onChange={(event) => {
                            const enabled = event.target.checked;
                            setSubtitleSync(enabled);
                            localStorage.setItem('dubvibe-subtitle-sync', String(enabled));
                            if (!enabled) {
                              setBurnSubtitles(false);
                              localStorage.setItem('dubvibe-burn-subtitles', 'false');
                            }
                            setExportFiles(null);
                          }}
                          className="h-4 w-4 accent-sky-500"
                        />
                        Include synced subtitles in export
                      </label>}
                      {step.id === 'subtitle' && <label className={`mt-2 flex items-center gap-2 text-xs ${subtitleSync ? 'cursor-pointer text-slate-300' : 'cursor-not-allowed text-slate-500'}`}>
                        <input
                          type="checkbox"
                          checked={burnSubtitles}
                          disabled={!subtitleSync}
                          onChange={(event) => {
                            setBurnSubtitles(event.target.checked);
                            localStorage.setItem('dubvibe-burn-subtitles', String(event.target.checked));
                            setExportFiles(null);
                          }}
                          className="h-4 w-4 accent-sky-500"
                        />
                        Burn captions into video
                      </label>}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Video parts — every chunk with live one-by-one dubbing progress */}
            {(videoChunks.length > 0 || (activeStep?.id === 'chunk' && activeStep.status === 'Active')) && (
              <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <p className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-slate-500">
                    <Scissors size={12} className="text-slate-500" />
                    Video parts
                  </p>
                  {videoChunks.length > 0 && (
                    <p className="text-[10px] text-slate-500">
                      <span className="font-semibold text-slate-300">{completedParts}</span> of {videoChunks.length} parts complete
                      {partStats.totalSpeech > 0 && <> · {partStats.stats.reduce((sum, s) => sum + s.dubbed, 0)}/{partStats.totalSpeech} lines</>}
                    </p>
                  )}
                </div>
                {videoChunks.length === 0 ? (
                  <div className="flex items-center gap-2 py-3 text-xs text-slate-500">
                    <RefreshCw size={12} className="animate-spin-fast text-sky-400" />
                    Splitting video into {chunkSeconds ? Math.round(chunkSeconds / 60) : 2}-minute parts…
                  </div>
                ) : (
                  <div className="grid max-h-52 gap-2 overflow-y-auto pr-1 [grid-template-columns:repeat(auto-fill,minmax(150px,1fr))]">
                    {partStats.stats.map((stat) => {
                      const state = partState(stat);
                      const progress = partProgress(stat);
                      return (
                        <button
                          key={stat.index}
                          onClick={() => seekTo(stat.start)}
                          title={`Part ${stat.index} · ${formatTimecode(stat.start)} – ${formatTimecode(stat.end)} (${formatTimecode(stat.end - stat.start)}) · ${stat.dubbed} of ${stat.speech} lines dubbed`}
                          className={`rounded-xl border p-2.5 text-left transition hover:border-sky-500/50 ${state === 'complete'
                              ? 'border-emerald-500/30 bg-emerald-500/5'
                              : state === 'processing'
                                ? 'border-amber-500/40 bg-amber-500/5'
                                : state === 'no-speech'
                                  ? 'border-white/5 bg-slate-950/60'
                                  : 'border-white/10 bg-slate-950/60'
                            }`}
                        >
                          <div className="flex items-center justify-between gap-1">
                            <span className={`text-xs font-medium ${state === 'complete' ? 'text-emerald-200' : state === 'processing' ? 'text-amber-200' : 'text-slate-300'}`}>
                              Part {stat.index}
                            </span>
                            <span className="flex items-center gap-1.5">
                              <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[9px] text-slate-400">
                                {formatTimecode(stat.end - stat.start).replace(/\.\d+$/, '')}
                              </span>
                              {state === 'processing' ? (
                                <RefreshCw size={10} className="animate-spin-fast text-amber-300" />
                              ) : (
                                <span className={`h-1.5 w-1.5 rounded-full ${state === 'complete' ? 'bg-emerald-400' : state === 'pending' ? 'bg-slate-600' : 'bg-slate-700'}`} />
                              )}
                            </span>
                          </div>
                          <p className="mt-0.5 font-mono text-[9px] text-slate-600">
                            {formatTimecode(stat.start)} – {formatTimecode(stat.end)}
                          </p>
                          {state === 'no-speech' ? (
                            <p className="mt-1.5 text-[9px] text-slate-600">No speech detected</p>
                          ) : (
                            <>
                              <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-800">
                                <div
                                  className={`h-full rounded-full ${state === 'complete' ? 'bg-emerald-400' : 'bg-amber-400'}`}
                                  style={{ width: `${progress}%` }}
                                />
                              </div>
                              <p className="mt-1 text-[9px] text-slate-500">
                                {state === 'processing' ? 'Dubbing… ' : ''}{stat.dubbed}/{stat.speech} lines
                              </p>
                            </>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {transcript.length > 0 && project.id && (
              <DubbingTrackEditor
                projectId={project.id}
                segments={transcript}
                audio={dubAudioSegments}
                targetLanguage={targetLanguage}
                provider={ttsProvider === 'auto' ? 'edge' : ttsProvider}
                currentTime={currentTime}
                videoDuration={projectDurationSeconds}
                onSeek={seekTo}
                onNotice={setNotice}
              />
            )}

            {/* Timeline editor */}
            <TimelineEditor
              projectId={project.id}
              mode={mode}
              busy={busy}
              segments={transcript}
              dubSegments={dubAudioSegments}
              dubAudioVersion={dubAudioVersion}
              hasMedia={hasMedia}
              mediaName={importedFile?.name ?? project.mediaName ?? 'Imported Video'}
              duration={projectDurationSeconds}
              currentTime={currentTime}
              isPlaying={isPlaying}
              videoRef={videoRef}
              videoChunks={videoChunks}
              chunkSeconds={chunkSeconds}
              selectedId={selectedSegmentId}
              onSelect={setSelectedSegmentId}
              onSeek={seekTo}
              onTogglePlay={togglePlay}
              onOpenEditor={openSegmentEditor}
              onNotice={setNotice}
              trackStates={trackStates}
              onToggleMute={toggleTrackMute}
              onToggleHide={toggleTrackHide}
              onToggleLock={toggleTrackLock}
              onRefresh={handleRefresh}
              isRefreshing={isRefreshing}
              onFixOverlaps={handleFixOverlaps}
              fixOverlapsState={fixOverlapsState}
              onCutRange={handleCutRange}
              cutBusy={cutBusy}
              targetLanguage={targetLanguage}
            />

            {/* Processing log */}
            <div className="shrink-0 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">Processing log</p>
                <button
                  onClick={clearLog}
                  disabled={processingLog.length === 0}
                  title="Clear log"
                  className="text-slate-500 hover:text-slate-300 transition disabled:opacity-30"
                >
                  <Trash2 size={12} />
                </button>
              </div>
              <div ref={logScrollRef} className="h-44 space-y-1 overflow-y-auto rounded-lg border border-white/5 bg-black/40 p-2 font-mono text-[10px] leading-4">
                {processingLog.length === 0 && (
                  <p className="text-slate-600">Waiting for activity. Progress from Import, Chunk, Transcribe, Translate, Dub, and Export appears here.</p>
                )}
                {processingLog.map((entry) => (
                  <div key={entry.id} className="flex gap-1.5">
                    <span className="shrink-0 text-slate-600">{entry.time}</span>
                    <span className={entry.level === 'error' ? 'text-rose-300' : entry.level === 'success' ? 'text-emerald-300' : 'text-slate-300'}>
                      {entry.message}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <aside className="bg-slate-950/70 p-4">
          <div className="mb-5 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">Inspector</p>
              <Cog size={14} className="text-slate-400" />
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Voice engine &amp; acting</label>
                <select
                  value={ttsProvider}
                  onChange={(event) => { const next = event.target.value; setTtsProvider(next); localStorage.setItem('dubvibe-tts-provider', next); if (next !== 'gemini' && next !== 'openai') { setTtsModel(''); localStorage.removeItem('dubvibe-tts-model'); } }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  <option value="auto">Auto — best available</option>
                  <option value="gemini">Google Gemini TTS (premium)</option>
                  <option value="openai">OpenAI TTS</option>
                  <option value="edge">Edge TTS (free)</option>
                  <option value="voxcpm2" disabled={voxcpmUnavailable}>
                    {voiceCapabilities['voxcpm2']?.installed === false
                      ? 'VoxCPM2 (offline, English) — download in AI Models'
                      : targetLanguage !== 'English' ? 'VoxCPM2 (offline, English) — target must be English' : 'VoxCPM2 (offline, English)'}
                  </option>
                  <option value="voxcpm_clone" disabled={voxcpmUnavailable}>
                    {voiceCapabilities['voxcpm2']?.installed === false
                      ? 'VoxCPM2 Voice Clone (English) — download in AI Models'
                      : targetLanguage !== 'English' ? 'VoxCPM2 Voice Clone (English) — target must be English' : 'VoxCPM2 Voice Clone (English, own voice)'}
                  </option>
                  <option value="sapi5">System TTS (offline)</option>
                </select>
                <p className="mt-1 text-[10px] leading-4 text-slate-500">
                  {ttsProvider === 'auto'
                    ? `Auto runs ${VOICE_ENGINE_LABELS[effectiveVoiceEngine] ?? effectiveVoiceEngine} for ${targetLanguage}.`
                    : engineSupportsLanguage(ttsProvider, targetLanguage)
                      ? `Dubs ${targetLanguage} with ${VOICE_ENGINE_LABELS[ttsProvider] ?? ttsProvider}${ttsModel ? ` · ${ttsModel}` : ''}.`
                      : `${VOICE_ENGINE_LABELS[ttsProvider] ?? ttsProvider} does not list ${targetLanguage}; the dub falls back to another engine.`}
                </p>
                {ttsProvider === 'gemini' && (
                  geminiModels.length ? (
                    <select
                      value={ttsModel}
                      onChange={(event) => { setTtsModel(event.target.value); localStorage.setItem('dubvibe-tts-model', event.target.value); }}
                      className="mt-2 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="">Default model</option>
                      {geminiModels.map((name) => <option key={name} value={name}>{name}</option>)}
                    </select>
                  ) : (
                    <p className="mt-1 text-[10px] text-amber-300">Test the Gemini connection in API Settings to list its available models.</p>
                  )
                )}
                {ttsProvider === 'openai' && (
                  <select
                    value={ttsModel}
                    onChange={(event) => { setTtsModel(event.target.value); localStorage.setItem('dubvibe-tts-model', event.target.value); }}
                    className="mt-2 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                  >
                    <option value="gpt-4o-mini-tts">gpt-4o-mini-tts (steerable)</option>
                    <option value="tts-1">tts-1</option>
                    <option value="tts-1-hd">tts-1-hd</option>
                  </select>
                )}
                {providerCaps && !providerCaps.style_control && (
                  <p className="mt-1 text-[10px] text-slate-500">Emotion control: not supported by the selected provider.</p>
                )}
                {(!ttsProvider || ttsProvider === 'auto' || (providerCaps && providerCaps.style_control)) && (
                  <>
                    <label className="mt-2 block text-[10px] uppercase tracking-[0.14em] text-slate-500">Emotion</label>
                    <select
                      value={voiceEmotion}
                      onChange={(event) => { setVoiceEmotion(event.target.value); localStorage.setItem('dubvibe-emotion', event.target.value); }}
                      className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="Auto">Auto — analyzed per line</option>
                      {(voiceCapabilities.emotions ?? []).map((emotion) => <option key={emotion} value={emotion}>{emotion}</option>)}
                    </select>
                    <label className="mt-2 block text-[10px] text-slate-500">
                      Acting intensity: {voiceIntensity}%
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={voiceIntensity}
                        onChange={(event) => { setVoiceIntensity(Number(event.target.value)); localStorage.setItem('dubvibe-voice-intensity', String(event.target.value)); }}
                        className="mt-1 block w-full accent-sky-500"
                      />
                    </label>
                  </>
                )}
                <label className="mt-2 block text-[10px] text-slate-500">
                  Speaking speed: {Math.round(voiceSpeed * 100)}%{voiceSpeed < 1 ? ' (slower)' : ' (normal)'}
                  <input
                    type="range"
                    min={0.5}
                    max={1}
                    step={0.05}
                    value={voiceSpeed}
                    onChange={(event) => { setVoiceSpeed(Number(event.target.value)); localStorage.setItem('dubvibe-voice-speed', String(event.target.value)); }}
                    title="Slows the generated voice."
                    className="mt-1 block w-full accent-sky-500"
                  />
                </label>
                <label className="mt-2 flex items-center gap-2 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={actingEnabled}
                    onChange={(event) => { setActingEnabled(event.target.checked); localStorage.setItem('dubvibe-acting', String(event.target.checked)); }}
                    className="h-4 w-4 accent-sky-500"
                  />
                  Acting analysis (context-aware delivery)
                </label>
                <label className="mt-1 flex items-center gap-2 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={openaiActing}
                    onChange={(event) => { setOpenaiActing(event.target.checked); localStorage.setItem('dubvibe-openai-acting', String(event.target.checked)); }}
                    className="h-4 w-4 accent-violet-500"
                  />
                  Use OpenAI for acting analysis (needs key)
                </label>
                <label className="mt-1 flex items-center gap-2 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={openaiReview}
                    onChange={(event) => { setOpenaiReview(event.target.checked); localStorage.setItem('dubvibe-openai-review', String(event.target.checked)); }}
                    className="h-4 w-4 accent-violet-500"
                  />
                  OpenAI translation review (costs per line)
                </label>
                <div className="mt-3 rounded-lg border border-white/10 bg-slate-950/70 p-2">
                  <p className="mb-1 text-[10px] uppercase tracking-[0.14em] text-slate-500">Line preview</p>
                  <input
                    value={previewText}
                    onChange={(event) => setPreviewText(event.target.value)}
                    placeholder={transcript.find((segment) => segment.translation)?.translation ?? 'Type a line to test the voice...'}
                    className="w-full rounded border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                  />
                  <div className="mt-1.5 flex items-center gap-2">
                    <button
                      onClick={() => void handlePreviewLine()}
                      disabled={previewBusy || busy}
                      className="rounded border border-sky-500/30 bg-sky-500/10 px-2 py-1 text-[10px] font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                    >
                      {previewBusy ? 'Generating...' : 'Generate preview'}
                    </button>
                    {previewAudio && <audio src={previewAudio} controls className="h-8 flex-1" />}
                  </div>
                </div>
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Target language</label>
                <select
                  value={targetLanguage}
                  onChange={(event) => {
                    setTargetLanguage(event.target.value);
                    localStorage.setItem('dubvibe-target-language', event.target.value);
                  }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  {TARGET_LANGUAGES.map((language) => (
                    <option key={language.code} value={language.name}>{language.name}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Voice</label>
                <select
                  value={voiceChoiceIsValid ? selectedVoice : selectableVoices[0]?.id ?? ''}
                  onChange={(event) => { setSelectedVoice(event.target.value); localStorage.setItem('dubvibe-voice', event.target.value); }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  {canMixVoices && <option value="mixed">Mixed — male &amp; female</option>}
                  {selectableVoices.map((voice) => (
                    <option key={voice.id} value={voice.id}>{voice.label}</option>
                  ))}
                </select>
                <p className="mt-1 text-[10px] leading-4 text-slate-500">{voiceHint}</p>
                {selectedVoice === 'mixed' && (
                  <p className="mt-1 text-[10px] text-slate-500">Alternates male and female voices at conversation pauses. For exact per-person voices, run Detect speakers.</p>
                )}
              </div>
              <div>
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Translation engine</label>
                <select
                  value={translationProvider}
                  onChange={(event) => { setTranslationProvider(event.target.value); localStorage.setItem('dubvibe-translation-provider', event.target.value); }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  <option value="auto">Auto — Qwen3 4B local first, Google fallback</option>
                  <option value="qwen">Qwen3 4B (local only — no online)</option>
                  <option value="nllb" disabled={!nllbModel?.installed}>
                    {nllbModel?.installed ? 'NLLB-200 (offline)' : 'NLLB-200 (offline) — download in AI Models'}
                  </option>
                  <option value="argos">Argos Translate (offline)</option>
                  <option value="google">Google Translate (online)</option>
                  <option value="gemini" disabled={!providerKeys.gemini.has_key}>
                    Google Gemini (online){providerKeys.gemini.has_key ? ' — key configured' : ' — add a key in API Settings'}
                  </option>
                  <option value="openai" disabled={!providerKeys.openai.has_key}>
                    OpenAI (online){providerKeys.openai.has_key ? ' — key configured' : ' — add a key in API Settings'}
                  </option>
                  <option value="codecraft" disabled={!providerKeys.codecraft.has_key}>
                    CodeCraft (online){providerKeys.codecraft.has_key ? ' — key configured' : ' — add a key in API Settings'}
                  </option>
                </select>
                <p className="mt-1 text-[10px] leading-4 text-slate-500">
                  {translationProvider === 'gemini' || translationProvider === 'openai'
                    ? 'Paid engine: every translated line is a billed API request. Uses the same key as the voice engine.'
                    : translationProvider === 'codecraft'
                      ? 'CodeCraft free tier: 1M tokens/month, no card required. Translation only — it has no voice engine.'
                      : translationProvider === 'qwen'
                        ? 'Local and free: Qwen3 4B only. If it cannot start, translation stops with an error — Google is never contacted.'
                        : 'Auto tries Qwen3 4B (local) first, then Argos, then Google Translate. Online API keys are only used when you pick that engine.'}
                  {' '}A line that fails moves to the next engine in this chain.
                </p>
              </div>
              <div className="border-t border-white/10 pt-3">
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Video chunking</label>
                <select value={chunkSelection} onChange={(event) => setChunkSelection(event.target.value)} className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200">
                  <option value="">Off — dub the whole video</option>
                  <option value="1">Auto-cut into 1-minute parts</option>
                  <option value="2">Auto-cut into 2-minute parts</option>
                  <option value="3">Auto-cut into 3-minute parts</option>
                  <option value="4">Auto-cut into 4-minute parts</option>
                  <option value="5">Auto-cut into 5-minute parts</option>
                  <option value="10">Auto-cut into 10-minute parts</option>
                  <option value="15">Auto-cut into 15-minute parts</option>
                </select>
                {videoChunks.length > 0 && (
                  <p className="mt-1 text-[10px] text-slate-400">
                    {videoChunks.length} parts · {Math.round((chunkSeconds ?? 0) / 60)} min each · timeline shows the part markers
                  </p>
                )}
                {videoChunks.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {videoChunks.map((chunk) => {
                      const dubbed = dubAudioSegments.some((segment) => segment.start >= chunk.start && segment.start < chunk.start + chunk.duration);
                      return (
                        <span
                          key={`chip-${chunk.index}`}
                          title={`Part ${chunk.index} · ${formatTimecode(chunk.start)}${dubbed ? ' · dubbed' : ' · not dubbed yet'}`}
                          className={`rounded border px-1.5 py-0.5 text-[10px] font-medium ${dubbed
                            ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-200'
                            : 'border-white/10 bg-slate-900 text-slate-400'
                            }`}
                        >
                          P{chunk.index}{dubbed ? ' ✓' : ''}
                        </span>
                      );
                    })}
                  </div>
                )}
                <button
                  onClick={() => void handleChunk()}
                  disabled={busy || apiAvailable !== true || !hasMedia}
                  className="mt-2 w-full rounded-lg border border-white/10 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-200 hover:border-sky-500/50 hover:bg-slate-700 transition disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? 'Working...' : videoChunks.length === 0
                    ? chunkSelection ? 'Split video now' : 'No split needed'
                    : chunkSelection ? 'Re-split with new length' : 'Remove split'}
                </button>
                <p className="mt-1 text-[10px] leading-4 text-slate-500">
                  Long videos are cut into parts before dubbing, so transcription and dubbing run chunk by chunk with the original timeline preserved. START DUBBING splits automatically when a length is selected.
                </p>
              </div>
              <div className="border-t border-white/10 pt-3">
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Audio separation</label>
                {separated ? (
                  <p className="text-xs leading-4 text-emerald-300">
                    ✓ Original dialogue is replaced by the dub. Choose which layers stay in the mix — they duck automatically during dialogue and recover after.
                  </p>
                ) : (
                  <p className="text-[10px] leading-4 text-slate-500">
                    Split the original audio into voice vs. music vs. sound effects/ambience (runs locally with Demucs). First run downloads the model; separation takes a while on CPU.
                  </p>
                )}
                {separated && (
                  <div className="mt-2 space-y-1.5">
                    <label className="flex items-center gap-2 text-xs text-slate-300">
                      <input
                        type="checkbox"
                        checked={keepSfx}
                        onChange={(event) => {
                          setKeepSfx(event.target.checked);
                          localStorage.setItem('dubvibe-keep-sfx', String(event.target.checked));
                          setExportFiles(null);
                        }}
                        className="h-4 w-4 accent-sky-500"
                      />
                      Original sound effects &amp; ambience
                    </label>
                    <label className="flex items-center gap-2 text-xs text-slate-300">
                      <input
                        type="checkbox"
                        checked={keepMusic}
                        onChange={(event) => {
                          setKeepMusic(event.target.checked);
                          localStorage.setItem('dubvibe-keep-music', String(event.target.checked));
                          setExportFiles(null);
                        }}
                        className="h-4 w-4 accent-sky-500"
                      />
                      Original music
                    </label>
                    <p className="text-[10px] text-slate-500">Original dialogue: OFF — replaced by the AI dubbed voice.</p>
                  </div>
                )}
                <button
                  onClick={() => void handleSeparate()}
                  disabled={busy || apiAvailable !== true || !hasMedia}
                  className="mt-2 w-full rounded-lg border border-white/10 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-200 hover:border-sky-500/50 hover:bg-slate-700 transition disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? 'Working...' : separated ? 'Re-extract voice & effects' : 'Extract voice & effects'}
                </button>
              </div>
              <div className="border-t border-white/10 pt-3">
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Characters</label>
                {characters.speakers.length === 0 ? (
                  <p className="text-[10px] text-slate-500">Transcribe first — every detected speaker becomes a character with its own voice.</p>
                ) : (
                  <div className="space-y-2">
                    {characters.speakers.map((name) => {
                      const saved = (characters.characters[name] ?? {}) as { provider?: string; voice?: string; gender?: string };
                      const edit = charEdits[name] ?? {};
                      const charProvider = edit.provider ?? saved.provider ?? 'auto';
                      const currentVoice = edit.voice ?? saved.voice ?? '';
                      const options = voiceOptionsFor(charProvider, targetLanguage);
                      // Gender auto-assign stores neural ids; always show the
                      // saved voice even when it is not in the standard list.
                      if (currentVoice && !options.some((voice) => voice.id === currentVoice)) {
                        options.unshift({ id: currentVoice, label: currentVoice });
                      }
                      return (
                        <div key={name} className="rounded-lg border border-white/10 bg-slate-950/70 p-2">
                          <p className="text-xs font-medium text-slate-200">
                            {name}
                            {saved.gender && (
                              <span className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-semibold align-middle ${saved.gender === 'Female' ? 'bg-pink-500/20 text-pink-200' : 'bg-sky-500/20 text-sky-200'}`}>
                                {saved.gender}
                              </span>
                            )}
                          </p>
                          <div className="mt-1 flex gap-1.5">
                            <select
                              value={charProvider}
                              onChange={(event) => setCharEdits((prev) => ({ ...prev, [name]: { ...prev[name], provider: event.target.value } }))}
                              className="w-1/2 rounded border border-white/10 bg-slate-950 px-1.5 py-1 text-[10px] text-slate-200"
                            >
                              <option value="auto">Auto</option>
                              <option value="gemini">Gemini</option>
                              <option value="openai">OpenAI</option>
                              <option value="edge">Edge</option>
                              <option value="sapi5">System</option>
                            </select>
                            <select
                              value={currentVoice}
                              onChange={(event) => setCharEdits((prev) => ({ ...prev, [name]: { ...prev[name], voice: event.target.value } }))}
                              className="min-w-0 flex-1 rounded border border-white/10 bg-slate-950 px-1.5 py-1 text-[10px] text-slate-200"
                            >
                              <option value="">Default voice</option>
                              {options.map((voice) => <option key={voice.id} value={voice.id}>{voice.label}</option>)}
                            </select>
                          </div>
                        </div>
                      );
                    })}
                    {Object.keys(charEdits).length > 0 && (
                      <div className="flex gap-1.5">
                        <button
                          onClick={() => void handleSaveCharacters()}
                          disabled={busy}
                          className="rounded border border-emerald-500/40 bg-emerald-500/15 px-2 py-1 text-[10px] font-medium text-emerald-200 hover:bg-emerald-500/25 transition disabled:opacity-40"
                        >
                          Save character voices
                        </button>
                        <button onClick={() => setCharEdits({})} className="rounded border border-white/10 px-2 py-1 text-[10px] text-slate-400 hover:text-slate-200">
                          Discard
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div className="border-t border-white/10 pt-3">
                <label className="flex items-center gap-2 text-sm text-slate-200">
                  <input
                    type="checkbox"
                    checked={preserveOriginalAudio}
                    onChange={(event) => {
                      setPreserveOriginalAudio(event.target.checked);
                      localStorage.setItem('dubvibe-preserve-audio', String(event.target.checked));
                      setExportFiles(null);
                    }}
                    className="h-4 w-4 accent-sky-500"
                  />
                  {separated ? 'Keep sound & music under dub (original voice removed)' : 'Keep original audio under dub'}
                </label>
                {preserveOriginalAudio && (
                  <label className="mt-2 flex items-center gap-2 text-xs text-slate-300" title="Original audio stays at full volume in gaps and dips only while a dubbed line is playing">
                    <input
                      type="checkbox"
                      checked={autoDuck}
                      onChange={(event) => {
                        setAutoDuck(event.target.checked);
                        localStorage.setItem('dubvibe-auto-duck', String(event.target.checked));
                        setExportFiles(null);
                      }}
                      className="h-4 w-4 accent-sky-500"
                    />
                    Auto-duck during dub voice (movie feel)
                  </label>
                )}
                {preserveOriginalAudio && <label className="mt-3 block text-xs text-slate-400">
                  {separated
                    ? `SFX/music duck level during dialogue: ${Math.round(originalAudioVolume * 100)}%${originalAudioVolume === 0 ? ' — bed fully ducks, recovers between lines' : ''}`
                    : `Original audio level: ${Math.round(originalAudioVolume * 100)}%${originalAudioVolume === 0 ? ' — dub voice only, original muted' : ''}`}
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={originalAudioVolume}
                    onChange={(event) => {
                      const volume = Number(event.target.value);
                      setOriginalAudioVolume(volume);
                      localStorage.setItem('dubvibe-original-volume', String(volume));
                      setExportFiles(null);
                    }}
                    className="mt-2 block w-full accent-sky-500"
                  />
                </label>}
              </div>
              <div className="border-t border-white/10 pt-3">
                <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-slate-500">Export as</label>
                <select
                  value={exportMode}
                  onChange={(event) => {
                    setExportMode(event.target.value);
                    localStorage.setItem('dubvibe-export-mode', event.target.value);
                    setExportFiles(null);
                  }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  <option value="single">One combined video</option>
                  <option value="both">Combined video + separate parts</option>
                  <option value="parts">Separate parts only (Part 1…N)</option>
                </select>
                <p className="mt-1 text-[10px] leading-4 text-slate-500">
                  Separate parts follow the video chunking above (Part 1, Part 2, …). Without chunking, export is always one combined video.
                </p>
                <label className="mb-1 mt-3 block text-xs uppercase tracking-[0.18em] text-slate-500">Platform style</label>
                <select
                  value={platformStyle}
                  onChange={(event) => { setPlatformStyle(event.target.value); localStorage.setItem('dubvibe-platform-style', event.target.value); setExportFiles(null); }}
                  className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                >
                  <option value="original">Original — no conversion</option>
                  <option value="vertical">TikTok / Reels / Shorts (9:16 vertical)</option>
                  <option value="standard">Standard YouTube / Facebook (16:9)</option>
                  <option value="square">Instagram / FB Feed (1:1 square)</option>
                </select>
                {platformStyle !== 'original' && (
                  <>
                    <label className="mb-1 mt-2 block text-[10px] uppercase tracking-[0.14em] text-slate-500">Fit style</label>
                    <select
                      value={platformFit}
                      onChange={(event) => { setPlatformFit(event.target.value); localStorage.setItem('dubvibe-platform-fit', event.target.value); setExportFiles(null); }}
                      className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="blur">Blurred background, video centered</option>
                      <option value="crop">Center-crop to fill</option>
                    </select>
                  </>
                )}
                <label className="mb-1 mt-3 block text-xs uppercase tracking-[0.18em] text-slate-500">Export folder</label>
                <div className="flex gap-2">
                  <input
                    value={pickedFolder ? '' : exportFolderInput}
                    onChange={(event) => {
                      setExportFolderInput(event.target.value);
                      localStorage.setItem('dubvibe-export-folder', event.target.value);
                    }}
                    placeholder={defaultExportFolder ?? 'Default export folder'}
                    disabled={Boolean(pickedFolder)}
                    spellCheck={false}
                    className="min-w-0 flex-1 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 font-mono text-[11px] text-slate-200 disabled:opacity-40"
                  />
                  {folderPickerSupported && (
                    <button
                      onClick={() => void handlePickFolder()}
                      className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition"
                    >
                      Choose…
                    </button>
                  )}
                </div>
                {pickedFolder && (
                  <div className="mt-1 flex items-center justify-between gap-2">
                    <span
                      className="inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-[10px] font-medium text-emerald-200"
                      title={`Files will be saved directly into the selected folder "${pickedFolder.name}"`}
                    >
                      Folder: {pickedFolder.name} ✓
                    </span>
                    <button
                      onClick={() => setPickedFolder(null)}
                      className="text-[10px] text-sky-400 hover:text-sky-300"
                    >
                      Clear selection
                    </button>
                  </div>
                )}
                <div className="mt-1 flex items-center justify-between gap-2">
                  <p className="text-[10px] leading-4 text-slate-500">
                    {folderPickerSupported
                      ? 'Use Choose… to pick any folder on disk — files are written there directly, no zip, no download.'
                      : 'Folder picker not supported in this browser — type the folder path above instead.'}
                  </p>
                  {exportFolderInput && !pickedFolder && (
                    <button
                      onClick={() => { setExportFolderInput(''); localStorage.removeItem('dubvibe-export-folder'); }}
                      className="shrink-0 text-[10px] text-sky-400 hover:text-sky-300"
                    >
                      Use default
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="mb-5 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">Media library</p>
              <Search size={14} className="text-slate-400" />
            </div>
            <div className="space-y-3">
              {hasMedia && (
                <div className="rounded-xl border border-white/10 bg-slate-950/80 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-200 truncate mr-2">
                      <FileVideo size={14} className="text-sky-300 shrink-0" />
                      <span className="truncate">{importedFile?.name ?? project.mediaName ?? 'Imported video'}</span>
                    </div>
                    <button
                      onClick={handleClearProject}
                      title="Remove video and reset project"
                      className="rounded p-1 text-slate-500 hover:bg-rose-500/10 hover:text-rose-400 transition shrink-0"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400">
                    <span>Video</span>
                    <span>Local preview</span>
                    <span>{((importedFile?.size ?? project.mediaSize ?? 0) / (1024 * 1024)).toFixed(1)} MB</span>
                    <span>{project.mediaResolution ?? 'Imported'}</span>
                  </div>
                </div>
              )}
              {mediaLibrary.map((item) => (
                <div key={item.id} className="rounded-xl border border-white/10 bg-slate-950/80 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="rounded-md bg-sky-500/15 p-1.5 text-sky-300"><FileVideo size={12} /></div>
                      <span className="text-sm font-medium text-slate-200">{item.name}</span>
                    </div>
                    <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-slate-300">{item.badge}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400">
                    <span>{item.type}</span>
                    <span>{item.duration}</span>
                    <span>{item.size}</span>
                    <span>{item.resolution}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mb-5 rounded-2xl border border-white/10 bg-slate-900/80 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] uppercase tracking-[0.22em] text-slate-500">Provider health</p>
              <button
                onClick={() => void handleRefresh()}
                disabled={isRefreshing}
                title="Refresh Providers & AI Models"
                className="flex items-center gap-1.5 text-xs text-sky-400 hover:text-sky-300 transition"
              >
                <RefreshCw size={12} className={isRefreshing ? 'animate-spin-fast' : ''} />
                <span>Refresh</span>
              </button>
            </div>
            <div className="space-y-2">
              {providers.map((provider) => {
                const downloading = downloadingModel(provider.id);
                const working = workingProviderIds.has(provider.id);
                return (
                  <div key={provider.id} className="flex items-center justify-between rounded-lg border border-white/10 bg-slate-950/80 px-2 py-2 text-sm">
                    <div>
                      <p className="font-medium text-slate-200">{provider.name}</p>
                      <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{provider.mode}</p>
                    </div>
                    {downloading ? (
                      <span className="flex items-center gap-1.5 rounded-full bg-amber-500/20 px-2 py-1 text-[10px] font-semibold text-amber-200">
                        <RefreshCw size={10} className="animate-spin-fast" />
                        {downloading.status === 'validating' ? 'Validating…' : `Downloading ${downloading.progress}%`}
                      </span>
                    ) : working ? (
                      <span className="flex items-center gap-1.5 rounded-full bg-sky-500/20 px-2 py-1 text-[10px] font-semibold text-sky-200">
                        <RefreshCw size={10} className="animate-spin-fast" />
                        Working…
                      </span>
                    ) : (
                      <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${provider.connected ? 'bg-emerald-500/20 text-emerald-200' : 'bg-orange-500/20 text-orange-200'}`}>
                        {provider.connected ? 'Connected' : 'Standby'}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

        </aside>
      </main>

      {editingSegment && editDraft && (
        <div className="fixed bottom-6 right-6 z-40 w-96 rounded-2xl border border-indigo-500/30 bg-slate-950 p-4 shadow-2xl">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-white">Edit segment</p>
              <p className="text-[10px] text-slate-500">
                {editingSegment.speaker} · {editDraft.start.toFixed(2)}s → {editDraft.end.toFixed(2)}s ({(editDraft.end - editDraft.start).toFixed(2)}s)
              </p>
            </div>
            <button onClick={() => setEditingSegmentId(null)} className="rounded p-1 text-slate-500 hover:text-white transition">✕</button>
          </div>
          <label className="mb-1 block text-[10px] uppercase tracking-[0.14em] text-slate-500">Original text</label>
          <textarea
            value={editDraft.text}
            onChange={(event) => { setEditDraft({ ...editDraft, text: event.target.value }); patchSegmentLocal(editingSegment.id, { text: event.target.value }); }}
            rows={2}
            className="w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
          />
          <label className="mb-1 mt-2 block text-[10px] uppercase tracking-[0.14em] text-slate-500">Translation (spoken by the dub)</label>
          <textarea
            value={editDraft.translation}
            onChange={(event) => { setEditDraft({ ...editDraft, translation: event.target.value }); patchSegmentLocal(editingSegment.id, { translation: event.target.value }); }}
            rows={2}
            className="w-full rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
          />
          <div className="mt-2 grid grid-cols-2 gap-2">
            <label className="text-[10px] text-slate-500">
              Start (s)
              <input
                type="number"
                min={0}
                step={0.1}
                value={editDraft.start}
                onChange={(event) => { const v = Math.max(0, Number(event.target.value) || 0); setEditDraft({ ...editDraft, start: v }); patchSegmentLocal(editingSegment.id, { start: v }); }}
                className="mt-0.5 w-full rounded border border-white/10 bg-slate-950 px-2 py-1 font-mono text-[11px] text-slate-200"
              />
            </label>
            <label className="text-[10px] text-slate-500">
              End (s)
              <input
                type="number"
                min={0}
                step={0.1}
                value={editDraft.end}
                onChange={(event) => { const v = Math.max(editDraft.start + 0.1, Number(event.target.value) || 0); setEditDraft({ ...editDraft, end: v }); patchSegmentLocal(editingSegment.id, { end: v }); }}
                className="mt-0.5 w-full rounded border border-white/10 bg-slate-950 px-2 py-1 font-mono text-[11px] text-slate-200"
              />
            </label>
          </div>
          <div className="mt-3 flex flex-wrap gap-1.5">
            <button
              onClick={() => void handleSaveSegmentEdit()}
              disabled={busy}
              className="rounded-lg bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-sky-400 disabled:opacity-40"
            >
              Save
            </button>
            <button
              onClick={() => void handleSplitSegmentEdit()}
              disabled={busy}
              className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
            >
              Split in two
            </button>
            <button
              onClick={() => void handleMergeSegmentEdit()}
              disabled={busy || !canMergeEditing}
              title={canMergeEditing ? 'Merge with the next segment' : 'This is the last segment'}
              className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Merge with next
            </button>
            <button
              onClick={() => void handleDeleteSegmentEdit()}
              disabled={busy}
              className="ml-auto rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 transition hover:bg-rose-500/20 disabled:opacity-40"
            >
              Delete
            </button>
          </div>
          <p className="mt-2 text-[10px] text-slate-500">
            After editing, press Dub — only changed lines are re-synthesized.
          </p>
        </div>
      )}

      {showApiKeys && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setShowApiKeys(false)}>
          <div
            className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-950 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div>
                <h2 className="text-lg font-semibold">Online AI Providers</h2>
                <p className="text-xs text-slate-500">
                  Keys are stored in Windows Credential Manager — never in the browser. Gemini and OpenAI keys enable
                  <span className="text-slate-300"> dubbing (TTS)</span> and
                  <span className="text-slate-300"> translation</span>; the CodeCraft key adds
                  <span className="text-slate-300"> free-tier translation</span>.
                </p>
              </div>
              <button onClick={() => setShowApiKeys(false)} className="rounded p-1 text-slate-500 hover:text-white transition">✕</button>
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {([
                {
                  id: 'gemini', label: 'Google Gemini',
                  placeholder: 'Paste your Google AI API key (AI Studio)',
                  value: geminiKeyInput, setValue: setGeminiKeyInput,
                },
                {
                  id: 'openai', label: 'OpenAI',
                  placeholder: 'Paste your OpenAI API key (sk-...)',
                  value: openaiKeyInput, setValue: setOpenaiKeyInput,
                },
                {
                  id: 'codecraft', label: 'CodeCraft',
                  placeholder: 'Paste your CodeCraft API key (cc_...)',
                  value: codecraftKeyInput, setValue: setCodecraftKeyInput,
                },
              ] as const).map(({ id: provider, label, placeholder, value, setValue }) => {
                const configured = providerKeys[provider]?.has_key;
                return (
                  <div key={provider} className="mb-4 rounded-xl border border-white/10 bg-slate-900/80 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <p className="text-sm font-medium text-slate-200">{label}</p>
                      <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${configured ? 'bg-emerald-500/20 text-emerald-200' : 'bg-slate-700/60 text-slate-300'}`}>
                        {configured ? '●— Key stored' : '○ Not configured'}
                      </span>
                    </div>
                    <input
                      type="password"
                      value={value}
                      onChange={(event) => setValue(event.target.value)}
                      placeholder={placeholder}
                      spellCheck={false}
                      className="w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-200"
                    />
                    <div className="mt-2 flex items-center gap-2">
                      <button
                        onClick={() => void handleSaveKey(provider, value)}
                        disabled={keyBusy !== null}
                        className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
                      >
                        {keyBusy === provider ? 'Saving…' : 'Save key'}
                      </button>
                      <button
                        onClick={() => void handleTestKey(provider)}
                        disabled={keyBusy !== null || !configured}
                        className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                      >
                        {keyBusy === provider ? 'Testing…' : 'Test connection'}
                      </button>
                      {configured && (
                        <button
                          onClick={() => { setValue(''); void removeProviderKey(provider); }}
                          className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                    {provider === 'gemini' && (
                      <p className="mt-1.5 text-[10px] leading-4 text-slate-500">
                        Get a key at aistudio.google.com/apikey. Translation into any app language (Khmer included)
                        works; Gemini <span className="text-slate-400">TTS</span> does not officially list Khmer, so Khmer
                        dubbing falls back to Edge TTS.
                      </p>
                    )}
                    {provider === 'openai' && (
                      <p className="mt-1.5 text-[10px] leading-4 text-slate-500">
                        This is the OpenAI <span className="text-slate-400">API</span> key from platform.openai.com — your ChatGPT subscription is not used here.
                        It powers OpenAI voice dubbing, translation, acting analysis, and the translation review pass.
                      </p>
                    )}
                    {provider === 'codecraft' && (
                      <p className="mt-1.5 text-[10px] leading-4 text-slate-500">
                        Get a key at codecraftapi.com — the free tier includes 1M tokens/month with no card required.
                        CodeCraft powers <span className="text-slate-400">translation only</span> (OpenAI-compatible
                        chat models); voice dubbing still uses Edge, Gemini, or OpenAI. The engine is only used when you
                        pick it in the Translation engine list.
                      </p>
                    )}
                  </div>
                );
              })}
              {keyTestMessage && (
                <p className={`rounded-lg border px-3 py-2 text-xs ${keyTestMessage.includes('connected') ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200' : 'border-amber-500/30 bg-amber-500/10 text-amber-200'}`}>
                  {keyTestMessage}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setDeleteTarget(null)}>
          <div
            className="w-full max-w-md rounded-2xl border border-white/10 bg-slate-950 p-5 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="shrink-0 rounded-xl border border-rose-500/30 bg-rose-500/10 p-2 text-rose-300">
                <Trash2 size={18} />
              </div>
              <div className="min-w-0">
                <h2 className="text-lg font-semibold text-white">Delete project</h2>
                <p className="mt-1 break-words text-sm text-slate-300">
                  Delete <span className="font-medium text-white">{deleteTarget.name}</span>?
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  This permanently removes its video, dub audio, and exports. This cannot be undone.
                </p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setDeleteTarget(null)}
                className="rounded-lg border border-white/10 bg-slate-800 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-700"
              >
                Cancel
              </button>
              <button
                onClick={() => void handleDeleteProject()}
                disabled={busy}
                className="rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-400 disabled:opacity-40"
              >
                {busy ? 'Deleting...' : 'Delete project'}
              </button>
            </div>
          </div>
        </div>
      )}

      {showClearAllConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setShowClearAllConfirm(false)}>
          <div
            className="w-full max-w-md rounded-2xl border border-white/10 bg-slate-950 p-5 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="shrink-0 rounded-xl border border-rose-500/30 bg-rose-500/10 p-2 text-rose-300">
                <Trash2 size={18} />
              </div>
              <div className="min-w-0">
                <h2 className="text-lg font-semibold text-white">Delete ALL projects?</h2>
                <p className="mt-1 text-sm text-slate-300">
                  Clear the whole project history — <span className="font-medium text-white">{projectList.length} {projectList.length === 1 ? 'project' : 'projects'}</span> will be removed.
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  This permanently removes every project's video, transcript, dub audio, and exports from disk. This cannot be undone.
                </p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setShowClearAllConfirm(false)}
                className="rounded-lg border border-white/10 bg-slate-800 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-700"
              >
                Cancel
              </button>
              <button
                onClick={() => void handleClearAllProjects()}
                disabled={busy}
                className="rounded-lg bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-400 disabled:opacity-40"
              >
                {busy ? 'Deleting...' : `Delete all ${projectList.length}`}
              </button>
            </div>
          </div>
        </div>
      )}

      {showModels && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setShowModels(false)}>
          <div
            className="flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-950 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div>
                <h2 className="text-lg font-semibold">AI Model Manager</h2>
                <p className="text-xs text-slate-500">Faster-Whisper transcription models · one shared store for web and desktop</p>
              </div>
              <button onClick={() => setShowModels(false)} className="rounded p-1 text-slate-500 hover:text-white transition">✕</button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              {hardware && (
                <div className="mb-4 rounded-xl border border-white/10 bg-slate-900/80 p-3 text-xs text-slate-400">
                  <p>
                    <span className="text-slate-500">Detected:</span>{' '}
                    {hardware.gpu ? `${hardware.gpu} · VRAM ${hardware.vram_gb} GB` : 'CPU'} · {hardware.cpu_cores} cores · {hardware.ram_gb} GB RAM
                  </p>
                  <p className="mt-0.5">
                    <span className="text-slate-500">Using:</span> device {hardware.resolved_device} · compute {hardware.resolved_compute} (safe CPU fallback)
                  </p>
                </div>
              )}

              <div className="space-y-3">
                {models.map((model) => (
                  <div key={model.id} className={`rounded-xl border p-4 ${model.selected ? 'border-sky-500/40 bg-sky-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                    <div className="flex items-start gap-3">
                      <input
                        type="radio"
                        name="selected-whisper-model"
                        checked={model.selected}
                        onChange={() => void runModelAction(`select-${model.id}`, () => selectModel(model.id))}
                        disabled={modelBusy !== null}
                        className="mt-1 h-4 w-4 accent-sky-500"
                        title="Use this model for transcription"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                          <p className="text-sm font-medium text-slate-200">{model.name}</p>
                          <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${model.status === 'loaded' ? 'bg-emerald-500/20 text-emerald-200'
                              : model.status === 'installed' ? 'bg-sky-500/20 text-sky-200'
                                : model.status === 'downloading' || model.status === 'validating' ? 'bg-amber-500/20 text-amber-200'
                                  : model.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                                    : 'bg-slate-700/60 text-slate-300'
                            }`}>
                            {model.status === 'not_installed' ? 'Not installed'
                              : model.status === 'downloading' ? `Downloading ${model.progress}%`
                                : model.status === 'validating' ? 'Validating…'
                                  : model.status === 'installed' ? '✓ Installed'
                                    : model.status === 'loaded' ? '✓ Loaded'
                                      : 'Error'}
                          </span>
                        </div>
                        <p className="mt-1 break-all text-xs leading-relaxed text-slate-500">
                          {model.installed && model.size_mb ? `${model.size_mb.toFixed(0)} MB · ${model.path}` : `~${model.total_mb} MB download`}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        {model.status === 'not_installed' && (
                          <button
                            onClick={() => void runModelAction(`dl-${model.id}`, () => downloadModel(model.id))}
                            disabled={modelBusy !== null}
                            className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                          >
                            Download
                          </button>
                        )}
                        {model.status === 'installed' && (
                          <button
                            onClick={() => void runModelAction(`load-${model.id}`, () => loadModel(model.id))}
                            disabled={modelBusy !== null}
                            className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
                          >
                            Load
                          </button>
                        )}
                        {model.status === 'loaded' && (
                          <button
                            onClick={() => void runModelAction(`unload-${model.id}`, () => unloadModel(model.id))}
                            disabled={modelBusy !== null}
                            className="rounded-lg border border-white/10 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
                          >
                            Unload
                          </button>
                        )}
                        {(model.status === 'installed' || model.status === 'error' || model.status === 'not_installed') && model.path && (
                          <button
                            onClick={() => void runModelAction(`del-${model.id}`, () => deleteModel(model.id))}
                            disabled={modelBusy !== null}
                            className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>
                    {model.status === 'downloading' && (
                      <div className="mt-3">
                        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                          <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${model.progress}%` }} />
                        </div>
                        <p className="mt-1 text-[10px] text-slate-500">
                          {model.downloaded_mb.toFixed(0)} MB / ~{model.total_mb} MB
                        </p>
                      </div>
                    )}
                    {model.status === 'error' && model.error && (
                      <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{model.error}</p>
                    )}
                  </div>
                ))}
              </div>

              {cudaPack && (
                <div className={`mb-3 rounded-xl border p-3 ${cudaPack.status === 'installed' ? 'border-emerald-500/30 bg-emerald-500/5'
                    : cudaPack.status === 'unavailable' ? 'border-white/10 bg-slate-900/60'
                      : 'border-white/10 bg-slate-900/80'
                  }`}>
                  <div className="flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-slate-200">{cudaPack.name}</p>
                      <p className="text-[10px] text-slate-500">
                        {cudaPack.gpu
                          ? `NVIDIA GPU detected: ${cudaPack.gpu.name} · VRAM ${cudaPack.gpu.vram_gb} GB`
                          : 'No NVIDIA GPU detected on this machine'}
                        {cudaPack.installed ? ' · pack installed' : ''}
                      </p>
                      <p className="mt-0.5 text-[10px] text-slate-500">
                        Accelerates Faster-Whisper transcription only. VoxCPM uses the GPU when a CUDA build of torch is installed; MMS, NLLB, and Demucs run on CPU.
                      </p>
                    </div>
                    <span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold ${cudaPack.status === 'installed' ? 'bg-emerald-500/20 text-emerald-200'
                        : cudaPack.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                          : cudaPack.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                            : cudaPack.status === 'unavailable' ? 'bg-slate-700/60 text-slate-400'
                              : 'bg-sky-500/20 text-sky-200'
                      }`}>
                      {cudaPack.status === 'unavailable' ? 'No NVIDIA GPU'
                        : cudaPack.status === 'downloading' ? `Downloading ${cudaPack.progress}%`
                          : cudaPack.status === 'installed' ? '✓ Installed'
                            : cudaPack.status === 'error' ? 'Error'
                              : 'Ready to download'}
                    </span>
                    {cudaPack.status === 'available' && (
                      <button
                        onClick={() => void runModelAction('dl-gpupack', downloadCudaPack)}
                        disabled={modelBusy !== null}
                        className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-2.5 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                      >
                        Download
                      </button>
                    )}
                    {cudaPack.installed && (
                      <button
                        onClick={() => void runModelAction('del-gpupack', deleteCudaPack)}
                        disabled={modelBusy !== null}
                        className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-2.5 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                  {cudaPack.status === 'downloading' && (
                    <div className="mt-2">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${cudaPack.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">
                        {cudaPack.downloaded_mb.toFixed(0)} MB / ~{cudaPack.total_mb} MB · speeds up transcription ~5-10x on NVIDIA
                      </p>
                    </div>
                  )}
                  {cudaPack.status === 'error' && cudaPack.error && (
                    <p className="mt-2 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{cudaPack.error}</p>
                  )}
                  {cudaPack.status === 'unavailable' && (
                    <p className="mt-2 text-[10px] leading-4 text-slate-500">
                      The pack only helps on machines with an NVIDIA GPU (CUDA 12 driver required). Transcription runs on CPU otherwise.
                    </p>
                  )}
                </div>
              )}

              {qwenModel && (
                <div className={`mt-3 rounded-xl border p-4 ${qwenModel.installed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <p className="text-sm font-medium text-slate-200">{qwenModel.name}</p>
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${qwenModel.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                            : qwenModel.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                              : qwenModel.installed ? 'bg-emerald-500/20 text-emerald-200'
                                : 'bg-slate-700/60 text-slate-300'
                          }`}>
                          {qwenModel.status === 'downloading' ? `Downloading ${qwenModel.progress}%`
                            : qwenModel.status === 'error' ? 'Error'
                              : qwenModel.installed ? '✓ Installed' : 'Not installed'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-500">
                        {qwenModel.installed
                          ? `${qwenModel.size_mb?.toFixed(0)} MB · local translation, works offline`
                          : `~${qwenModel.total_mb} MB download · local translation engine, works offline`}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {qwenModel.status === 'not_installed' && (
                        <button
                          onClick={() => void runModelAction('dl-qwen', downloadQwen)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                        >
                          Download
                        </button>
                      )}
                      {qwenModel.installed && (
                        <button
                          onClick={() => void runModelAction('del-qwen', deleteQwen)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                  {qwenModel.status === 'downloading' && (
                    <div className="mt-3">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${qwenModel.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">{qwenModel.downloaded_mb.toFixed(0)} MB / ~{qwenModel.total_mb} MB</p>
                    </div>
                  )}
                  {qwenModel.status === 'error' && qwenModel.error && (
                    <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{qwenModel.error}</p>
                  )}
                </div>
              )}

              {nllbModel && (
                <div className={`mt-3 rounded-xl border p-4 ${nllbModel.installed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <p className="text-sm font-medium text-slate-200">{nllbModel.name}</p>
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${nllbModel.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                            : nllbModel.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                              : nllbModel.installed ? 'bg-emerald-500/20 text-emerald-200'
                                : 'bg-slate-700/60 text-slate-300'
                          }`}>
                          {nllbModel.status === 'downloading' ? `Downloading ${nllbModel.progress}%`
                            : nllbModel.status === 'error' ? 'Error'
                              : nllbModel.installed ? '✓ Installed' : 'Not installed'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-500">
                        {nllbModel.installed
                          ? `${nllbModel.size_mb?.toFixed(0)} MB · offline translation. Choose it in Translation engine.`
                          : `~${nllbModel.total_mb} MB download · offline translation, 200 languages. Research model, non-commercial license (cc-by-nc-4.0).`}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {nllbModel.status === 'not_installed' && !nllbConfirm && (
                        <button
                          onClick={() => setNllbConfirm(true)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                        >
                          Download
                        </button>
                      )}
                      {nllbModel.status === 'not_installed' && nllbConfirm && (
                        <>
                          <button
                            onClick={() => { setNllbConfirm(false); void runModelAction('dl-nllb', downloadNllb); }}
                            disabled={modelBusy !== null}
                            className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/20 px-3 py-1.5 text-xs font-medium text-sky-100 hover:bg-sky-500/30 transition disabled:opacity-40"
                          >
                            Confirm download (~{(nllbModel.total_mb / 1024).toFixed(1)} GB)
                          </button>
                          <button
                            onClick={() => setNllbConfirm(false)}
                            className="shrink-0 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-white/5 transition"
                          >
                            Cancel
                          </button>
                        </>
                      )}
                      {nllbModel.installed && (
                        <button
                          onClick={() => void runModelAction('del-nllb', deleteNllb)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                  {nllbModel.status === 'downloading' && (
                    <div className="mt-3">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${nllbModel.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">{nllbModel.downloaded_mb.toFixed(0)} MB / ~{nllbModel.total_mb} MB</p>
                    </div>
                  )}
                  {nllbModel.status === 'error' && nllbModel.error && (
                    <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{nllbModel.error}</p>
                  )}
                </div>
              )}
              {genderModel && (
                <div className={`mt-3 rounded-xl border p-4 ${genderModel.installed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <p className="text-sm font-medium text-slate-200">{genderModel.name}</p>
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${genderModel.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                            : genderModel.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                              : genderModel.installed ? 'bg-emerald-500/20 text-emerald-200'
                                : 'bg-slate-700/60 text-slate-300'
                          }`}>
                          {genderModel.status === 'downloading' ? `Downloading ${genderModel.progress}%`
                            : genderModel.status === 'error' ? 'Error'
                              : genderModel.installed ? '✓ Installed' : 'Not installed'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-500">
                        {genderModel.installed
                          ? `${genderModel.size_mb?.toFixed(0)} MB · hears male/female in Detect Speakers, so Mixed voice and Character voices pick correctly every time.`
                          : `~${genderModel.total_mb} MB download · classifies male/female per speaker in Detect Speakers, so Mixed voice dubs each speaker in the correct voice (pitch fallback untouched). MIT license.`}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {genderModel.status === 'not_installed' && !genderConfirm && (
                        <button
                          onClick={() => setGenderConfirm(true)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                        >
                          Download
                        </button>
                      )}
                      {genderModel.status === 'not_installed' && genderConfirm && (
                        <>
                          <button
                            onClick={() => { setGenderConfirm(false); void runModelAction('dl-gender', downloadGender); }}
                            disabled={modelBusy !== null}
                            className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/20 px-3 py-1.5 text-xs font-medium text-sky-100 hover:bg-sky-500/30 transition disabled:opacity-40"
                          >
                            Confirm download (~{genderModel.total_mb} MB)
                          </button>
                          <button
                            onClick={() => setGenderConfirm(false)}
                            className="shrink-0 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-white/5 transition"
                          >
                            Cancel
                          </button>
                        </>
                      )}
                      {genderModel.installed && (
                        <button
                          onClick={() => void runModelAction('del-gender', deleteGender)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                  {genderModel.status === 'downloading' && (
                    <div className="mt-3">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${genderModel.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">{genderModel.downloaded_mb.toFixed(0)} MB / ~{genderModel.total_mb} MB</p>
                    </div>
                  )}
                  {genderModel.status === 'error' && genderModel.error && (
                    <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{genderModel.error}</p>
                  )}
                </div>
              )}
              {demucsModel && (
                <div className={`mt-3 rounded-xl border p-4 ${demucsModel.installed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <p className="text-sm font-medium text-slate-200">{demucsModel.name}</p>
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${demucsModel.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                            : demucsModel.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                              : demucsModel.installed ? 'bg-emerald-500/20 text-emerald-200'
                                : 'bg-slate-700/60 text-slate-300'
                          }`}>
                          {demucsModel.status === 'downloading' ? `Downloading ${demucsModel.progress}%`
                            : demucsModel.status === 'error' ? 'Error'
                              : demucsModel.installed ? '✓ Installed' : 'Not installed'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-500">
                        {demucsModel.installed
                          ? `${demucsModel.size_mb?.toFixed(0)} MB · splits voice, music and sound effects, works offline`
                          : `~${demucsModel.total_mb} MB download · required for "Extract voice & effects", works offline after install`}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {demucsModel.status === 'not_installed' && (
                        <button
                          onClick={() => void runModelAction('dl-demucs', downloadDemucs)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                        >
                          Download
                        </button>
                      )}
                      {demucsModel.installed && (
                        <button
                          onClick={() => void runModelAction('del-demucs', deleteDemucs)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                  {demucsModel.status === 'downloading' && (
                    <div className="mt-3">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${demucsModel.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">{demucsModel.downloaded_mb.toFixed(0)} MB / ~{demucsModel.total_mb} MB</p>
                    </div>
                  )}
                  {demucsModel.status === 'error' && demucsModel.error && (
                    <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{demucsModel.error}</p>
                  )}
                </div>
              )}

              {voxcpmModel && (
                <div className={`mt-3 rounded-xl border p-4 ${voxcpmModel.installed ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-white/10 bg-slate-900/80'}`}>
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                        <p className="text-sm font-medium text-slate-200">{voxcpmModel.name}</p>
                        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${voxcpmModel.status === 'downloading' ? 'bg-amber-500/20 text-amber-200'
                            : voxcpmModel.status === 'error' ? 'bg-rose-500/20 text-rose-200'
                              : voxcpmModel.installed ? 'bg-emerald-500/20 text-emerald-200'
                                : 'bg-slate-700/60 text-slate-300'
                          }`}>
                          {voxcpmModel.status === 'downloading' ? `Downloading ${voxcpmModel.progress}%`
                            : voxcpmModel.status === 'error' ? 'Error'
                              : voxcpmModel.installed ? '✓ Installed' : 'Not installed'}
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-slate-500">
                        {voxcpmModel.installed
                          ? `${voxcpmModel.size_mb?.toFixed(0)} MB · Offline English voice (Apache-2.0, commercial use allowed). CPU works but is slow; a CUDA GPU is much faster.`
                          : `~${(voxcpmModel.total_mb / 1024).toFixed(1)} GB download · offline English voice. Apache-2.0 license. Slow on CPU; a CUDA GPU is recommended.`}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {voxcpmModel.status === 'not_installed' && !voxcpmConfirm && (
                        <button
                          onClick={() => setVoxcpmConfirm(true)}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-500/20 transition disabled:opacity-40"
                        >
                          Download
                        </button>
                      )}
                      {voxcpmModel.status === 'not_installed' && voxcpmConfirm && (
                        <>
                          <button
                            onClick={() => { setVoxcpmConfirm(false); void runModelAction('dl-voxcpm2', () => downloadVoxcpm()); }}
                            disabled={modelBusy !== null}
                            className="shrink-0 rounded-lg border border-sky-500/30 bg-sky-500/20 px-3 py-1.5 text-xs font-medium text-sky-100 hover:bg-sky-500/30 transition disabled:opacity-40"
                          >
                            Confirm download (~{(voxcpmModel.total_mb / 1024).toFixed(1)} GB)
                          </button>
                          <button
                            onClick={() => setVoxcpmConfirm(false)}
                            className="shrink-0 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-white/5 transition"
                          >
                            Cancel
                          </button>
                        </>
                      )}
                      {voxcpmModel.installed && (
                        <button
                          onClick={() => void runModelAction('del-voxcpm2', () => deleteVoxcpm())}
                          disabled={modelBusy !== null}
                          className="shrink-0 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-40"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                  {voxcpmModel.status === 'downloading' && (
                    <div className="mt-3">
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-violet-500" style={{ width: `${voxcpmModel.progress}%` }} />
                      </div>
                      <p className="mt-1 text-[10px] text-slate-500">{voxcpmModel.downloaded_mb.toFixed(0)} MB / ~{voxcpmModel.total_mb} MB</p>
                    </div>
                  )}
                  {voxcpmModel.status === 'error' && voxcpmModel.error && (
                    <p className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200">{voxcpmModel.error}</p>
                  )}
                </div>
              )}


              <div className="mt-4 rounded-xl border border-white/10 bg-slate-900/80 p-3">
                <p className="mb-2 text-[10px] uppercase tracking-[0.18em] text-slate-500">Transcription settings</p>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="text-xs text-slate-400">
                    Processor
                    <select
                      value={hardware?.device ?? 'auto'}
                      onChange={(event) => { void runModelAction(`device-${event.target.value}`, () => setDevice(event.target.value as 'auto' | 'cuda' | 'cpu')); }}
                      className="mt-1 block w-44 rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                    >
                      <option value="auto">Auto — use GPU if available</option>
                      <option value="cuda">GPU (CUDA)</option>
                      <option value="cpu">CPU only</option>
                    </select>
                  </label>
                  <label className="text-xs text-slate-400">
                    Language
                    <select
                      value={transcribeLanguage}
                      onChange={(event) => { setTranscribeLanguage(event.target.value); localStorage.setItem('dubvibe-transcribe-language', event.target.value); }}
                      className="mt-1 block w-40 rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                    >
                      <option value="">Auto-detect</option>
                      <option value="en">English</option>
                      <option value="km">Khmer</option>
                      <option value="fr">French</option>
                      <option value="es">Spanish</option>
                      <option value="zh">Chinese</option>
                      <option value="ja">Japanese</option>
                      <option value="ko">Korean</option>
                      <option value="th">Thai</option>
                      <option value="vi">Vietnamese</option>
                      <option value="de">German</option>
                      <option value="pt">Portuguese</option>
                      <option value="ru">Russian</option>
                      <option value="ar">Arabic</option>
                      <option value="hi">Hindi</option>
                      <option value="id">Indonesian</option>
                    </select>
                  </label>
                  <label className="text-xs text-slate-400">
                    Beam size
                    <input
                      type="number"
                      min={1}
                      max={10}
                      value={beamSize}
                      onChange={(event) => {
                        const size = Math.max(1, Math.min(10, Number(event.target.value) || 5));
                        setBeamSize(size);
                        localStorage.setItem('dubvibe-beam-size', String(size));
                      }}
                      className="mt-1 block w-20 rounded-lg border border-white/10 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
                    />
                  </label>
                  <p className="flex-1 text-[10px] leading-4 text-slate-500">
                    Processor applies to transcription, audio separation, and translation. VAD, temperature fallback, and hallucination filtering are always on. The selected model is used by both the web UI and the desktop app.
                  </p>
                </div>
              </div>

              {(() => {
                const storage = useStudioStore.getState().modelStorage;
                if (!storage) return null;
                return (
                  <p className="mt-4 text-[10px] text-slate-500">
                    Storage: {storage.root} · {storage.installed_mb.toFixed(0)} MB used · {storage.free_gb} GB free
                  </p>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Auto-Fix Overlaps: Preview Modal */}
      {fixOverlapsState === 'preview' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-slate-900 p-6 shadow-2xl">
            <h2 className="mb-4 text-base font-semibold text-white">
              Auto-Fix Speech Overlaps
            </h2>

            {fixProposals.length === 0 ? (
              <p className="mb-6 text-sm text-slate-400">No overlaps found — timeline looks good!</p>
            ) : (
              <div className="mb-4 max-h-72 overflow-y-auto space-y-2">
                {fixProposals.map((p) => (
                  <div key={p.segment_id} className="rounded-lg border border-white/10 bg-slate-800/60 px-3 py-2 text-xs">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-slate-300">{p.segment_id}</span>
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${p.type === 'speed_adjust' ? 'bg-sky-500/20 text-sky-300' :
                          p.type === 'silence_absorb' ? 'bg-amber-500/20 text-amber-300' :
                            'bg-rose-500/20 text-rose-300'
                        }`}>
                        {p.type === 'speed_adjust' ? 'Speed' : p.type === 'silence_absorb' ? 'Shift' : 'Retranslate'}
                      </span>
                    </div>
                    <p className="text-slate-400">{p.description}</p>
                    {p.type === 'speed_adjust' && (
                      <p className="mt-0.5 text-slate-500">
                        {p.old_start?.toFixed(2)}s–{p.old_end?.toFixed(2)}s · speed {p.old_speed ?? 1.0} → {p.new_speed}
                      </p>
                    )}
                    {p.type === 'silence_absorb' && (
                      <p className="mt-0.5 text-slate-500">
                        Start: {p.old_start?.toFixed(2)}s → {p.new_start?.toFixed(2)}s
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setFixOverlapsState('idle')}
                className="rounded-lg border border-white/10 bg-slate-800 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 transition"
              >
                {fixProposals.length === 0 ? 'Close' : 'Cancel'}
              </button>
              {fixProposals.filter((p) => p.type !== 'needs_retranslation').length > 0 && (
                <button
                  onClick={() => void handleApplyFix()}
                  className="rounded-lg bg-gradient-to-r from-sky-500 to-violet-500 px-4 py-2 text-sm font-semibold text-white shadow-glow transition"
                >
                  Apply {fixProposals.filter((p) => p.type !== 'needs_retranslation').length} fix
                  {fixProposals.filter((p) => p.type !== 'needs_retranslation').length !== 1 ? 'es' : ''}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Auto-Fix Overlaps: Applying Overlay */}
      {fixOverlapsState === 'applying' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="rounded-2xl border border-white/10 bg-slate-900 px-8 py-6 text-sm text-white">
            Applying fixes…
          </div>
        </div>
      )}

      {/* Auto-Fix Overlaps: Undo Banner */}
      {canUndoOverlaps && fixOverlapsState === 'idle' && (
        <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2 flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-slate-900 px-4 py-2.5 shadow-2xl text-sm text-emerald-100">
          <span>
            {fixOverlapsFound > 0
              ? `${fixOverlapsFound} overlap${fixOverlapsFound !== 1 ? 's' : ''} fixed`
              : 'Overlap fixes applied'}
          </span>
          <button
            onClick={() => void handleUndoFix()}
            className="text-sky-400 hover:text-sky-300 font-medium"
          >
            Undo
          </button>
          <button onClick={() => setCanUndoOverlaps(false)} className="text-slate-500 hover:text-slate-400">✕</button>
        </div>
      )}

      {/* Auto-Fix Overlaps: Error Banner */}
      {fixError && fixOverlapsState === 'idle' && (
        <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2 flex items-center gap-3 rounded-xl border border-rose-500/30 bg-slate-900 px-4 py-2.5 shadow-2xl text-sm text-rose-200">
          <span>Auto-Fix failed: {fixError}</span>
          <button onClick={() => setFixError(null)} className="text-slate-500 hover:text-slate-400">✕</button>
        </div>
      )}
    </div>
  );
}

export default App;
