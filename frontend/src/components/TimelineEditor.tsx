import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  AudioLines,
  Eye,
  EyeOff,
  Film,
  Languages,
  Layers,
  Lock,
  Magnet,
  Maximize2,
  Mic,
  Pause,
  Play,
  RefreshCw,
  Scissors,
  SkipBack,
  SkipForward,
  SplitSquareVertical,
  Unlock,
  Volume2,
  VolumeX,
  Wand2,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useStudioStore } from '../store/useStudioStore';
import type { ProjectMode, TimelineSegment } from '../types';

const apiBaseUrl = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';

const MIN_PPS = 0.08;
const MAX_PPS = 400;
const DEFAULT_PPS = 60;
const RULER_HEIGHT = 32;
const SNAP_PX = 8;
const MIN_CLIP_SECONDS = 0.2;

type TrackName = 'Video' | 'Original Speech' | 'Translation' | 'Dubbed Voice' | 'Subtitle';
type DragKind = 'move' | 'trim-l' | 'trim-r';
type ClipVariant = 'speech' | 'translation' | 'dub' | 'subtitle';

export interface TimelineTrackState {
  muted: boolean;
  hidden: boolean;
  locked: boolean;
}

interface VideoChunk {
  index: number;
  start: number;
  duration: number;
}

interface ThumbnailSprite {
  url: string;
  count: number;
  cols: number;
  rows: number;
  step: number;
}

interface TimelineEditorProps {
  projectId: string;
  mode: ProjectMode;
  busy: boolean;
  segments: TimelineSegment[];
  dubSegments: TimelineSegment[];
  dubAudioVersion: number;
  hasMedia: boolean;
  mediaName: string;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  videoChunks: VideoChunk[];
  chunkSeconds: number | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onSeek: (time: number) => void;
  onTogglePlay: () => void;
  onOpenEditor: (id: string) => void;
  onNotice: (message: string) => void;
  trackStates: Record<string, TimelineTrackState>;
  onToggleMute: (track: string) => void;
  onToggleHide: (track: string) => void;
  onToggleLock: (track: string) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  onFixOverlaps: () => void;
  fixOverlapsState: 'idle' | 'analyzing' | 'preview' | 'applying';
  onCutRange: (range: { start: number; end: number }, cutMode: 'remove' | 'keep') => Promise<boolean>;
  cutBusy: boolean;
  targetLanguage: string;
}

const TRACKS: Array<{ name: TrackName; height: number; icon: LucideIcon; color: string }> = [
  { name: 'Video', height: 62, icon: Film, color: 'text-sky-400' },
  { name: 'Original Speech', height: 50, icon: Mic, color: 'text-indigo-400' },
  { name: 'Translation', height: 40, icon: Languages, color: 'text-emerald-400' },
  { name: 'Dubbed Voice', height: 58, icon: AudioLines, color: 'text-amber-400' },
  { name: 'Subtitle', height: 40, icon: SplitSquareVertical, color: 'text-rose-400' },
];

const TRACK_HEIGHT = TRACKS.reduce((total, track) => total + track.height, 0);

const CLIP_STYLES: Record<ClipVariant, { idle: string; selected: string; handle: string; text: string }> = {
  speech: {
    idle: 'border-indigo-400/25 bg-gradient-to-b from-indigo-500/35 to-indigo-700/25 hover:border-indigo-300/60 hover:from-indigo-500/45',
    selected: 'border-indigo-200 bg-indigo-500/55 shadow-lg shadow-indigo-950/60',
    handle: 'bg-indigo-200/80',
    text: 'text-indigo-50',
  },
  translation: {
    idle: 'border-emerald-400/25 bg-gradient-to-b from-emerald-500/30 to-emerald-700/20 hover:border-emerald-300/60 hover:from-emerald-500/40',
    selected: 'border-emerald-200 bg-emerald-500/50 shadow-lg shadow-emerald-950/60',
    handle: 'bg-emerald-200/80',
    text: 'text-emerald-50',
  },
  dub: {
    idle: 'border-amber-400/25 bg-gradient-to-b from-amber-500/25 to-amber-700/20 hover:border-amber-300/60 hover:from-amber-500/35',
    selected: 'border-amber-200 bg-amber-500/45 shadow-lg shadow-amber-950/60',
    handle: 'bg-amber-200/80',
    text: 'text-amber-50',
  },
  subtitle: {
    idle: 'border-rose-400/25 bg-gradient-to-b from-rose-500/30 to-rose-700/20 hover:border-rose-300/60 hover:from-rose-500/40',
    selected: 'border-rose-200 bg-rose-500/50 shadow-lg shadow-rose-950/60',
    handle: 'bg-rose-200/80',
    text: 'text-rose-50',
  },
};

/** Audio levels bucket used by the ruler (labels need to stay readable). */
const RULER_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const round2 = (value: number) => Math.round(value * 100) / 100;

function pad2(value: number) {
  return String(value).padStart(2, '0');
}

function formatTimecode(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const secs = (safe - minutes * 60).toFixed(2).padStart(5, '0');
  return `${pad2(minutes)}:${secs}`;
}

function formatRulerLabel(seconds: number, tenths: boolean): string {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor(safe / 60) % 60;
  const secs = safe % 60;
  if (hours > 0) return `${hours}:${pad2(minutes)}:${pad2(Math.floor(secs))}`;
  if (tenths) return `${pad2(minutes)}:${secs.toFixed(1).padStart(4, '0')}`;
  return `${pad2(minutes)}:${pad2(Math.floor(secs))}`;
}

function rulerSteps(pixelsPerSecond: number) {
  const major = RULER_STEPS.find((step) => step * pixelsPerSecond >= 88) ?? RULER_STEPS[RULER_STEPS.length - 1];
  return { major, minor: major / 5 };
}

function buildWaveformPath(peaks: number[]): string {
  if (peaks.length === 0) return '';
  const top: string[] = [];
  const bottom: string[] = [];
  for (let index = 0; index < peaks.length; index += 1) {
    const amplitude = Math.max(0.015, Math.min(1, peaks[index] ?? 0)) * 44;
    top.push(`${index + 0.5} ${50 - amplitude}`);
    bottom.unshift(`${index + 0.5} ${50 + amplitude}`);
  }
  return `M ${top.join(' L ')} L ${bottom.join(' L ')} Z`;
}

function Waveform({ peaks, className }: { peaks: number[]; className?: string }) {
  const path = useMemo(() => buildWaveformPath(peaks), [peaks]);
  if (!path) return null;
  return (
    <svg
      viewBox={`0 0 ${peaks.length} 100`}
      preserveAspectRatio="none"
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 h-full w-full ${className ?? ''}`}
    >
      <path d={path} fill="currentColor" />
    </svg>
  );
}

function readSnapPreference(): boolean {
  try {
    return localStorage.getItem('dubvibe-timeline-snap') !== 'off';
  } catch {
    return true;
  }
}

export function TimelineEditor({
  projectId,
  mode,
  busy,
  segments,
  dubSegments,
  dubAudioVersion,
  hasMedia,
  mediaName,
  duration,
  currentTime,
  isPlaying,
  videoRef,
  videoChunks,
  chunkSeconds,
  selectedId,
  onSelect,
  onSeek,
  onTogglePlay,
  onOpenEditor,
  onNotice,
  trackStates,
  onToggleMute,
  onToggleHide,
  onToggleLock,
  onRefresh,
  isRefreshing,
  onFixOverlaps,
  fixOverlapsState,
  onCutRange,
  cutBusy,
  targetLanguage,
}: TimelineEditorProps) {
  const editable = mode === 'EDIT' && !busy;

  const [pixelsPerSecond, setPixelsPerSecond] = useState(DEFAULT_PPS);
  const [snapEnabled, setSnapEnabled] = useState(readSnapPreference);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [viewportWidth, setViewportWidth] = useState(1200);
  const [scrub, setScrub] = useState<'playhead' | 'range' | null>(null);
  const [rangeSelectMode, setRangeSelectMode] = useState(false);
  const [selectedRange, setSelectedRange] = useState<{ start: number; end: number } | null>(null);
  const [snapGuide, setSnapGuide] = useState<number | null>(null);
  const [peaks, setPeaks] = useState<Record<string, { peaks: number[]; seconds: number }>>({});
  const [sprite, setSprite] = useState<ThumbnailSprite | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const rulerRef = useRef<HTMLDivElement | null>(null);
  const playheadRef = useRef<HTMLDivElement | null>(null);
  const pendingAnchorRef = useRef<{ time: number; offset: number } | null>(null);
  const scrollRafRef = useRef(0);
  const fittedProjectRef = useRef('');

  // Handlers live in refs so memoized track rows do not re-render on every
  // timeupdate tick coming from the player.
  const callbacksRef = useRef({ onSeek, onSelect, onOpenEditor, onNotice });
  callbacksRef.current = { onSeek, onSelect, onOpenEditor, onNotice };

  const stateRef = useRef({ pixelsPerSecond, currentTime, duration, viewportWidth });
  stateRef.current = { pixelsPerSecond, currentTime, duration, viewportWidth };

  const snapPoints = useMemo(() => {
    const points = new Set<number>([0, round2(duration)]);
    for (const segment of segments) {
      points.add(round2(segment.start));
      points.add(round2(segment.end));
    }
    for (const segment of dubSegments) {
      points.add(round2(segment.start));
      points.add(round2(segment.end));
    }
    for (const chunk of videoChunks) {
      points.add(round2(chunk.start));
    }
    return Array.from(points).sort((a, b) => a - b);
  }, [segments, dubSegments, videoChunks, duration]);
  const snapPointsRef = useRef(snapPoints);
  snapPointsRef.current = snapPoints;

  const maxTime = Math.max(duration, viewportWidth / Math.max(pixelsPerSecond, MIN_PPS));
  const contentWidth = Math.max(viewportWidth, duration * pixelsPerSecond + 80);

  const timeFromClientX = useCallback((clientX: number) => {
    const ruler = rulerRef.current;
    const pps = stateRef.current.pixelsPerSecond;
    if (!ruler) return 0;
    const rect = ruler.getBoundingClientRect();
    return clamp((clientX - rect.left) / pps, 0, stateRef.current.duration);
  }, []);

  // --- viewport + zoom ------------------------------------------------------
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const update = () => setViewportWidth(element.clientWidth || 1200);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const applyZoom = useCallback((nextPixelsPerSecond: number, anchorClientX?: number) => {
    const element = scrollRef.current;
    const previous = stateRef.current.pixelsPerSecond;
    const next = clamp(nextPixelsPerSecond, MIN_PPS, MAX_PPS);
    if (!Number.isFinite(next) || Math.abs(next - previous) < 1e-6) return;
    if (!element || element.clientWidth === 0) {
      setPixelsPerSecond(next);
      return;
    }
    const rect = element.getBoundingClientRect();
    const offset = anchorClientX == null
      ? clamp(stateRef.current.currentTime * previous - element.scrollLeft, 0, element.clientWidth)
      : clamp(anchorClientX - rect.left, 0, element.clientWidth);
    pendingAnchorRef.current = {
      time: Math.max(0, (element.scrollLeft + offset) / previous),
      offset,
    };
    setPixelsPerSecond(next);
  }, []);

  const fitToView = useCallback(() => {
    const element = scrollRef.current;
    if (!element || element.clientWidth === 0) return;
    const total = stateRef.current.duration;
    if (total <= 0) return;
    pendingAnchorRef.current = null;
    setPixelsPerSecond(clamp((element.clientWidth - 32) / total, MIN_PPS, MAX_PPS));
    element.scrollLeft = 0;
    setScrollLeft(0);
  }, []);

  // Fit a project the first time its duration is known.
  useEffect(() => {
    if (!projectId || duration <= 1 || viewportWidth <= 0) return;
    if (fittedProjectRef.current === projectId) return;
    fittedProjectRef.current = projectId;
    fitToView();
  }, [projectId, duration, viewportWidth, fitToView]);

  useEffect(() => {
    const element = scrollRef.current;
    const anchor = pendingAnchorRef.current;
    if (!element || !anchor) return;
    pendingAnchorRef.current = null;
    element.scrollLeft = Math.max(0, anchor.time * pixelsPerSecond - anchor.offset);
  }, [pixelsPerSecond]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        applyZoom(stateRef.current.pixelsPerSecond * Math.exp(-event.deltaY * 0.0022), event.clientX);
      } else if (event.shiftKey && !event.altKey) {
        event.preventDefault();
        element.scrollLeft += event.deltaY;
      }
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, [applyZoom]);

  // Reset the horizontal position when switching projects.
  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollLeft = 0;
    setScrollLeft(0);
    setSelectedRange(null);
    setRangeSelectMode(false);
  }, [projectId]);

  useEffect(() => {
    if (mode !== 'EDIT') {
      setRangeSelectMode(false);
      setSelectedRange(null);
    }
  }, [mode]);

  useEffect(() => {
    try {
      localStorage.setItem('dubvibe-timeline-snap', snapEnabled ? 'on' : 'off');
    } catch {
      // Storage can be unavailable in private windows; snapping still works.
    }
  }, [snapEnabled]);

  const handleScroll = useCallback(() => {
    if (scrollRafRef.current) return;
    scrollRafRef.current = window.requestAnimationFrame(() => {
      scrollRafRef.current = 0;
      const element = scrollRef.current;
      if (element) setScrollLeft(element.scrollLeft);
    });
  }, []);

  useEffect(() => () => {
    if (scrollRafRef.current) window.cancelAnimationFrame(scrollRafRef.current);
  }, []);

  // --- playhead -------------------------------------------------------------
  // The player only emits timeupdate a few times a second. While playing, the
  // playhead is painted straight from the video element every animation frame
  // (and the timeline follows it) without re-rendering the clip tree.
  useLayoutEffect(() => {
    const node = playheadRef.current;
    if (!node) return;
    let frame = 0;
    const paint = (time: number) => {
      const x = clamp(time, 0, Math.max(duration, 0)) * pixelsPerSecond;
      node.style.transform = `translateX(${x}px)`;
    };
    const follow = (time: number) => {
      const element = scrollRef.current;
      if (!element || element.clientWidth === 0 || pendingAnchorRef.current) return;
      const x = clamp(time, 0, Math.max(duration, 0)) * pixelsPerSecond;
      if (x > element.scrollLeft + element.clientWidth * 0.78 || x < element.scrollLeft) {
        element.scrollLeft = Math.max(0, x - element.clientWidth * 0.25);
      }
    };
    const playing = isPlaying && Boolean(videoRef.current);
    if (!playing) {
      paint(currentTime);
      const element = scrollRef.current;
      if (element && element.clientWidth > 0 && !pendingAnchorRef.current) {
        const x = currentTime * pixelsPerSecond;
        if (x < element.scrollLeft || x > element.scrollLeft + element.clientWidth - 24) {
          element.scrollLeft = Math.max(0, x - element.clientWidth * 0.25);
        }
      }
      return;
    }
    const tick = () => {
      const time = videoRef.current?.currentTime ?? currentTime;
      paint(time);
      follow(time);
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [isPlaying, currentTime, pixelsPerSecond, duration, videoRef]);

  // --- data for clip visuals ------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    if (!projectId || dubSegments.length === 0) {
      setPeaks({});
      return;
    }
    void fetch(`${apiBaseUrl}/api/projects/${projectId}/timeline/peaks`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: { clips?: Array<{ index: number; id: string; seconds?: number; peaks: number[] }> } | null) => {
        if (cancelled || !data) return;
        const next: Record<string, { peaks: number[]; seconds: number }> = {};
        for (const clip of data.clips ?? []) {
          const value = {
            peaks: Array.isArray(clip.peaks) ? clip.peaks : [],
            seconds: Number(clip.seconds) || 0,
          };
          if (clip.id) next[clip.id] = value;
          next[`#${clip.index}`] = value;
        }
        setPeaks(next);
      })
      .catch(() => {
        // A missing waveform must never break the timeline.
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, dubSegments.length, dubAudioVersion]);

  useEffect(() => {
    let cancelled = false;
    if (!projectId || !hasMedia) {
      setSprite(null);
      return;
    }
    void fetch(`${apiBaseUrl}/api/projects/${projectId}/timeline/thumbnails`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: ThumbnailSprite | null) => {
        if (!cancelled && data && typeof data.url === 'string') setSprite(data);
      })
      .catch(() => {
        // No thumbnails: the clip falls back to a labelled gradient.
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, hasMedia, mediaName]);

  // --- dragging clips -------------------------------------------------------
  const beginSegmentDrag = useCallback((event: React.PointerEvent, segment: TimelineSegment, kind: DragKind, locked: boolean) => {
    if (!editable || locked || event.button !== 0) return;
    event.stopPropagation();
    event.preventDefault();
    const pps = stateRef.current.pixelsPerSecond;
    const total = stateRef.current.duration;
    const info = {
      id: segment.id,
      kind,
      startX: event.clientX,
      origStart: segment.start,
      origEnd: segment.end,
      moved: false,
    };
    document.body.style.cursor = kind === 'move' ? 'grabbing' : 'ew-resize';

    const snapEdge = (value: number): number | null => {
      let best: number | null = null;
      let distance = SNAP_PX / pps;
      for (const point of snapPointsRef.current) {
        const delta = Math.abs(point - value);
        if (delta <= distance) {
          distance = delta;
          best = point;
        }
      }
      return best;
    };

    const onMove = (moveEvent: PointerEvent) => {
      if (moveEvent.pointerId !== event.pointerId) return;
      const delta = (moveEvent.clientX - info.startX) / pps;
      if (Math.abs(moveEvent.clientX - info.startX) > 3) {
        info.moved = true;
      }
      let start = info.origStart;
      let end = info.origEnd;
      let guide: number | null = null;
      const useSnap = snapEnabled && !moveEvent.altKey;

      if (kind === 'move') {
        const length = info.origEnd - info.origStart;
        start = clamp(info.origStart + delta, 0, Math.max(0, total - length));
        end = start + length;
        if (useSnap) {
          const snappedStart = snapEdge(start);
          const snappedEnd = snapEdge(end);
          if (snappedStart != null && (snappedEnd == null || Math.abs(snappedStart - start) <= Math.abs(snappedEnd - end))) {
            const shift = snappedStart - start;
            start = clamp(start + shift, 0, Math.max(0, total - length));
            end = start + length;
            guide = snappedStart;
          } else if (snappedEnd != null) {
            const shift = snappedEnd - end;
            start = clamp(start + shift, 0, Math.max(0, total - length));
            end = start + length;
            guide = snappedEnd;
          }
        }
      } else if (kind === 'trim-l') {
        start = clamp(info.origStart + delta, 0, info.origEnd - MIN_CLIP_SECONDS);
        if (useSnap) {
          const snapped = snapEdge(start);
          if (snapped != null && snapped <= info.origEnd - MIN_CLIP_SECONDS) {
            start = snapped;
            guide = snapped;
          }
        }
      } else {
        end = clamp(info.origEnd + delta, info.origStart + MIN_CLIP_SECONDS, total);
        if (useSnap) {
          const snapped = snapEdge(end);
          if (snapped != null && snapped >= info.origStart + MIN_CLIP_SECONDS) {
            end = snapped;
            guide = snapped;
          }
        }
      }
      setSnapGuide(guide);
      useStudioStore.getState().patchSegmentLocal(info.id, {
        start: round2(start),
        end: round2(end),
      });
    };

    const onUp = (upEvent: PointerEvent) => {
      if (upEvent.pointerId !== event.pointerId) return;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      document.body.style.cursor = '';
      setSnapGuide(null);
      const fresh = useStudioStore.getState().transcript.find((item) => item.id === info.id);
      if (info.moved && fresh) {
        void useStudioStore.getState().updateSegment(projectId, info.id, {
          start: fresh.start,
          end: fresh.end,
        }).catch((error: unknown) => {
          callbacksRef.current.onNotice(error instanceof Error ? error.message : 'Could not save the new timing.');
        });
      } else if (!info.moved && kind === 'move') {
        callbacksRef.current.onSeek(fresh ? fresh.start : info.origStart);
        callbacksRef.current.onSelect(info.id);
        callbacksRef.current.onOpenEditor(info.id);
      }
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  }, [editable, projectId, snapEnabled]);

  // --- ruler scrubbing / range selection -----------------------------------
  const handleRulerPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const time = timeFromClientX(event.clientX);
    if (editable && rangeSelectMode && !cutBusy) {
      setSelectedRange({ start: time, end: time });
      setScrub('range');
      return;
    }
    setScrub('playhead');
    callbacksRef.current.onSeek(time);
  };

  useEffect(() => {
    if (!scrub) return;
    const onMove = (event: PointerEvent) => {
      const time = timeFromClientX(event.clientX);
      if (scrub === 'range') {
        setSelectedRange((current) => (current ? { ...current, end: time } : { start: time, end: time }));
      } else {
        callbacksRef.current.onSeek(time);
      }
    };
    const onUp = () => {
      if (scrub === 'range') {
        setSelectedRange((current) => {
          if (!current) return null;
          const start = Math.min(current.start, current.end);
          const end = Math.max(current.start, current.end);
          return end - start < 0.15 ? null : { start, end };
        });
      }
      setScrub(null);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [scrub, timeFromClientX]);

  const applyCut = async (cutMode: 'remove' | 'keep') => {
    if (!selectedRange || cutBusy) return;
    const ok = await onCutRange(selectedRange, cutMode);
    if (ok) setSelectedRange(null);
  };

  // --- keyboard shortcuts ---------------------------------------------------
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)) {
        return;
      }
      const total = stateRef.current.duration;
      if (event.code === 'Space') {
        event.preventDefault();
        onTogglePlay();
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        callbacksRef.current.onSeek(Math.max(0, stateRef.current.currentTime - (event.shiftKey ? 5 : 1)));
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        callbacksRef.current.onSeek(Math.min(total, stateRef.current.currentTime + (event.shiftKey ? 5 : 1)));
      } else if (event.key === 'Home') {
        event.preventDefault();
        callbacksRef.current.onSeek(0);
      } else if (event.key === 'End') {
        event.preventDefault();
        callbacksRef.current.onSeek(total);
      } else if (event.key === 'Escape') {
        callbacksRef.current.onSelect(null);
        setSelectedRange(null);
        setRangeSelectMode(false);
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        applyZoom(stateRef.current.pixelsPerSecond * 1.4);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        applyZoom(stateRef.current.pixelsPerSecond / 1.4);
      } else if (event.key.toLowerCase() === 'f') {
        fitToView();
      } else if (event.key.toLowerCase() === 's') {
        setSnapEnabled((value) => !value);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [applyZoom, fitToView, onTogglePlay]);

  // --- ruler ticks (virtualized to the visible window) ----------------------
  const ticks = useMemo(() => {
    if (duration <= 0 || pixelsPerSecond <= 0) return [];
    const { major, minor } = rulerSteps(pixelsPerSecond);
    const start = Math.max(0, (scrollLeft - 120) / pixelsPerSecond);
    const end = Math.min(maxTime, (scrollLeft + viewportWidth + 120) / pixelsPerSecond);
    const result: Array<{ time: number; x: number; major: boolean; label: string }> = [];
    const firstIndex = Math.floor(start / minor);
    for (let index = firstIndex; ; index += 1) {
      const time = round2(index * minor);
      if (time > end + 0.0001) break;
      if (time < 0 || time > maxTime) continue;
      const isMajor = Math.abs(time / major - Math.round(time / major)) < 0.0001;
      result.push({
        time,
        x: time * pixelsPerSecond,
        major: isMajor,
        label: isMajor ? formatRulerLabel(time, major < 1) : '',
      });
    }
    return result;
  }, [duration, pixelsPerSecond, scrollLeft, viewportWidth, maxTime]);

  // --- clip rows ------------------------------------------------------------
  const trackRows = useMemo(() => {
    const selectedFor = (id: string) => selectedId === id;
    const dubById = new Map(dubSegments.map((segment) => [segment.id, segment]));

    const renderSegment = (
      track: TrackName,
      kind: 'speech' | 'translation' | 'dub' | 'subtitle',
      index: number,
      segment: TimelineSegment,
      start: number,
      end: number,
      label: string,
      sublabel: string,
      waveform: { peaks: number[]; seconds: number } | null,
    ) => {
      const width = Math.max(4, (end - start) * pixelsPerSecond);
      const state = trackStates[track];
      const locked = Boolean(state?.locked);
      const canDrag = editable && !locked;
      const isSelected = selectedFor(segment.id) && canDrag;
      const styles = CLIP_STYLES[kind];
      const showLabel = width >= 42;
      const showSub = width >= 96 && TRACKS.find((item) => item.name === track)!.height >= 46;
      return (
        <div
          key={`${kind}-${segment.id}-${index}`}
          onPointerDown={canDrag ? (event) => beginSegmentDrag(event, segment, 'move', locked) : undefined}
          onClick={canDrag ? undefined : () => {
            callbacksRef.current.onSeek(start);
            callbacksRef.current.onSelect(segment.id);
          }}
          title={`${formatTimecode(start)} – ${formatTimecode(end)}${label ? `\n${label}` : ''}${mode === 'EDIT' ? '\nDrag to move · drag an edge to trim · click to edit text' : ''}`}
          role="button"
          tabIndex={-1}
          className={`group absolute top-[3px] bottom-[3px] overflow-hidden rounded-[5px] border text-left transition-colors ${isSelected ? styles.selected : styles.idle} ${canDrag ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer'}`}
          style={{ left: start * pixelsPerSecond, width }}
        >
          {waveform && (() => {
            const slot = Math.max(0.01, end - start);
            // Dub clips keep their speech slot; draw the spoken part only so a
            // short take reads as audio followed by room tone, like a real NLE.
            const filled = waveform.seconds > 0 ? clamp(waveform.seconds / slot, 0.02, 1) : 1;
            return (
              <div
                className="pointer-events-none absolute inset-y-0 left-0 overflow-hidden opacity-50"
                style={{ width: `${filled * 100}%` }}
              >
                <Waveform peaks={waveform.peaks} />
              </div>
            );
          })()}
          <div className={`relative z-10 flex h-full flex-col justify-between px-1.5 py-[3px] ${styles.text}`}>
            {showLabel ? (
              <span className="truncate text-[10px] font-medium leading-tight drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">{label}</span>
            ) : null}
            {showSub ? (
              <span className="truncate font-mono text-[8px] leading-tight text-white/45">{sublabel}</span>
            ) : null}
          </div>
          {canDrag && width >= 18 && (
            <>
              <div
                onPointerDown={(event) => beginSegmentDrag(event, segment, 'trim-l', locked)}
                title="Trim start"
                className={`absolute inset-y-0 left-0 z-20 w-1.5 cursor-ew-resize opacity-0 transition-opacity group-hover:opacity-100 ${styles.handle}`}
              />
              <div
                onPointerDown={(event) => beginSegmentDrag(event, segment, 'trim-r', locked)}
                title="Trim end"
                className={`absolute inset-y-0 right-0 z-20 w-1.5 cursor-ew-resize opacity-0 transition-opacity group-hover:opacity-100 ${styles.handle}`}
              />
            </>
          )}
        </div>
      );
    };

    return TRACKS.map((track) => {
      const state = trackStates[track.name];
      const hidden = Boolean(state?.hidden);
      const height = track.height;
      return (
        <div key={track.name} className="relative border-b border-white/[0.045]" style={{ height }}>
          {hidden ? null : track.name === 'Video' ? (
            <>
              {hasMedia && duration > 0 && (
                <div
                  onClick={() => callbacksRef.current.onSeek(0)}
                  title={mediaName || 'Imported Video'}
                  className="absolute inset-y-[3px] left-0 cursor-pointer overflow-hidden rounded-md border border-sky-400/30 bg-sky-950/60 transition-colors hover:border-sky-400/50"
                  style={{ width: Math.max(10, duration * pixelsPerSecond) }}
                >
                  {sprite ? (
                    Array.from({ length: sprite.count }).map((_, tileIndex) => {
                      const cellWidth = sprite.step * pixelsPerSecond;
                      const column = tileIndex % sprite.cols;
                      const row = Math.floor(tileIndex / sprite.cols);
                      return (
                        <div
                          key={`tile-${tileIndex}`}
                          className="absolute inset-y-0"
                          style={{
                            left: cellWidth * tileIndex,
                            width: cellWidth + 0.5,
                            backgroundImage: `url(${apiBaseUrl}${sprite.url})`,
                            backgroundSize: `${sprite.cols * cellWidth}px ${sprite.rows * height}px`,
                            backgroundPosition: `${-column * cellWidth}px ${-row * height}px`,
                            backgroundRepeat: 'no-repeat',
                          }}
                        />
                      );
                    })
                  ) : (
                    <div className="absolute inset-0 bg-gradient-to-r from-sky-900/70 via-sky-800/40 to-sky-900/70" />
                  )}
                  <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-black/60 via-black/10 to-black/50" />
                  <button
                    type="button"
                    onClick={() => callbacksRef.current.onSeek(0)}
                    title={mediaName}
                    className="absolute left-2 top-1/2 z-10 flex max-w-[calc(100%-16px)] -translate-y-1/2 items-center gap-1.5 rounded bg-black/35 px-1.5 py-0.5 text-[10px] font-medium text-sky-50 backdrop-blur-[2px] hover:bg-black/50"
                  >
                    <Film size={11} className="shrink-0 text-sky-300" />
                    <span className="truncate">{mediaName || 'Imported Video'}</span>
                    {videoChunks.length > 1 && (
                      <span className="shrink-0 rounded bg-sky-500/25 px-1 text-[9px] text-sky-100">{videoChunks.length} parts</span>
                    )}
                  </button>
                </div>
              )}
              {hasMedia && !hidden && videoChunks.filter((chunk) => chunk.start > 0).map((chunk) => (
                <div
                  key={`chunk-${chunk.index}`}
                  className="pointer-events-none absolute inset-y-0 z-10"
                  style={{ left: chunk.start * pixelsPerSecond }}
                  title={`Part ${chunk.index} starts at ${formatTimecode(chunk.start)}`}
                >
                  <div className="h-full w-px bg-sky-300/60" />
                  <span className="absolute left-1 top-1 rounded-sm border border-sky-400/30 bg-sky-950/90 px-1 text-[8px] font-semibold text-sky-200">
                    P{chunk.index}
                  </span>
                </div>
              ))}
            </>
          ) : track.name === 'Original Speech' ? (
            segments.map((segment, index) => renderSegment(
              track.name,
              'speech',
              index,
              segment,
              segment.start,
              segment.end,
              segment.text,
              `${formatTimecode(segment.start)} · ${(segment.end - segment.start).toFixed(1)}s`,
              null,
            ))
          ) : track.name === 'Translation' ? (
            segments.map((segment, index) => {
              const dub = dubById.get(segment.id);
              const start = dub?.start ?? segment.start;
              const end = dub?.end ?? segment.end;
              return renderSegment(
                track.name,
                'translation',
                index,
                segment,
                start,
                end,
                segment.translation || segment.text,
                `${targetLanguage || 'Translation'} · ${formatTimecode(start)}`,
                null,
              );
            })
          ) : track.name === 'Dubbed Voice' ? (
            dubSegments.map((segment, index) => renderSegment(
              track.name,
              'dub',
              index,
              segment,
              segment.start,
              segment.end,
              segment.translation || segment.text || segment.voice || 'Dubbed line',
              segment.voice || 'Dubbed Voice',
              peaks[segment.id] ?? peaks[`#${index + 1}`] ?? null,
            ))
          ) : (
            segments.map((segment, index) => {
              const dub = dubById.get(segment.id);
              const start = dub?.start ?? segment.start;
              const end = dub?.end ?? segment.end;
              return renderSegment(
                track.name,
                'subtitle',
                index,
                segment,
                start,
                end,
                segment.translation || segment.text,
                `${formatTimecode(start)} – ${formatTimecode(end)}`,
                null,
              );
            })
          )}
        </div>
      );
    });
  }, [
    beginSegmentDrag,
    dubSegments,
    duration,
    editable,
    hasMedia,
    mediaName,
    mode,
    peaks,
    pixelsPerSecond,
    segments,
    selectedId,
    sprite,
    targetLanguage,
    trackStates,
    videoChunks,
  ]);

  const rangeBounds = selectedRange
    ? { start: Math.min(selectedRange.start, selectedRange.end), end: Math.max(selectedRange.start, selectedRange.end) }
    : null;

  const sliderValue = Math.round(1000 * Math.log(pixelsPerSecond / MIN_PPS) / Math.log(MAX_PPS / MIN_PPS));
  const pixelLabel = pixelsPerSecond >= 10 ? `${Math.round(pixelsPerSecond)} px/s` : `${pixelsPerSecond.toFixed(2)} px/s`;

  return (
    <div className="timeline-container overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0c1220] shadow-2xl">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] bg-[#0e1526] px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Scissors size={16} className="text-sky-400" />
            <span>Timeline</span>
          </div>
          <span className="rounded-full border border-sky-500/15 bg-sky-500/10 px-2.5 py-0.5 text-[10px] font-medium text-sky-300">
            {segments.length} clips
          </span>
          {videoChunks.length > 0 && (
            <span className="rounded-full border border-emerald-500/15 bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-medium text-emerald-300">
              {videoChunks.length} parts × {Math.round((chunkSeconds ?? 0) / 60)} min
            </span>
          )}

          <div className="ml-2 flex items-center gap-1.5 border-l border-white/[0.06] pl-3">
            <button
              onClick={() => onSeek(0)}
              title="Jump to start (Home)"
              className="rounded p-1 text-slate-500 transition hover:bg-white/5 hover:text-white"
            >
              <SkipBack size={14} />
            </button>
            <button
              onClick={onTogglePlay}
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
              className="rounded-md bg-white/[0.06] p-1.5 text-white transition hover:bg-white/10"
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              onClick={() => onSeek(currentTime + 2)}
              title="Step forward 2s"
              className="rounded p-1 text-slate-500 transition hover:bg-white/5 hover:text-white"
            >
              <SkipForward size={14} />
            </button>

            <div className="ml-1 flex items-center gap-2 rounded-md border border-white/[0.06] bg-black/30 px-2 py-1 font-mono text-[11px]">
              <span className="font-semibold text-white">{formatTimecode(currentTime)}</span>
              <span className="text-slate-600">/</span>
              <span className="text-slate-500">{formatTimecode(duration)}</span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setSnapEnabled((value) => !value)}
            title={`Magnetic snapping ${snapEnabled ? 'on' : 'off'} (S) · hold Alt to bypass`}
            className={`flex items-center gap-1.5 rounded-md border px-2 py-1.5 text-[11px] transition ${snapEnabled
              ? 'border-cyan-400/40 bg-cyan-500/15 text-cyan-200'
              : 'border-white/[0.06] bg-white/[0.03] text-slate-500 hover:text-white'
              }`}
          >
            <Magnet size={12} />
            <span>Snap</span>
          </button>

          <div className="flex items-center gap-1 rounded-md border border-white/[0.06] bg-black/20 px-2 py-1 text-xs text-slate-400">
            <button
              onClick={() => applyZoom(pixelsPerSecond / 1.4)}
              title="Zoom out (-)"
              className="p-0.5 transition hover:text-white"
            >
              <ZoomOut size={12} />
            </button>
            <input
              type="range"
              min="0"
              max="1000"
              step="1"
              value={sliderValue}
              onChange={(event) => {
                const value = Number(event.target.value);
                applyZoom(MIN_PPS * Math.pow(MAX_PPS / MIN_PPS, value / 1000));
              }}
              className="timeline-zoom-slider h-1 w-20 cursor-pointer"
              title={`Timeline zoom: ${pixelLabel}`}
            />
            <button
              onClick={() => applyZoom(pixelsPerSecond * 1.4)}
              title="Zoom in (+)"
              className="p-0.5 transition hover:text-white"
            >
              <ZoomIn size={12} />
            </button>
            <button
              onClick={fitToView}
              title="Fit project to view (F)"
              className="ml-0.5 p-0.5 text-slate-500 transition hover:text-white"
            >
              <Maximize2 size={11} />
            </button>
          </div>

          <button
            onClick={() => void onRefresh()}
            disabled={isRefreshing}
            title="Refresh studio state"
            className="flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.03] px-2.5 py-1.5 text-[11px] text-slate-400 transition hover:bg-white/[0.06] hover:text-white disabled:opacity-40"
          >
            <RefreshCw size={12} className={isRefreshing ? 'animate-spin-fast text-sky-400' : ''} />
            <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>

          {mode === 'EDIT' && dubSegments.length > 0 && (
            <button
              onClick={() => void onFixOverlaps()}
              disabled={fixOverlapsState !== 'idle'}
              title="Detect and fix speech overlaps automatically"
              className="flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.03] px-2.5 py-1.5 text-[11px] text-slate-400 transition hover:bg-white/[0.06] hover:text-white disabled:opacity-40"
            >
              <Wand2 size={12} className={fixOverlapsState === 'analyzing' ? 'animate-spin-fast text-sky-400' : ''} />
              <span>{fixOverlapsState === 'analyzing' ? 'Analyzing…' : 'Auto-Fix Overlaps'}</span>
            </button>
          )}

          {mode === 'EDIT' && (
            <button
              onClick={() => {
                setRangeSelectMode((value) => !value);
                setSelectedRange(null);
              }}
              disabled={cutBusy}
              title="Drag on the ruler to select a time range, then cut it out or keep only it"
              className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[11px] transition disabled:opacity-40 ${rangeSelectMode
                ? 'border-amber-400/40 bg-amber-500/20 text-amber-200'
                : 'border-white/[0.06] bg-white/[0.03] text-slate-400 hover:bg-white/[0.06] hover:text-white'
                }`}
            >
              <Scissors size={12} />
              <span>{rangeSelectMode ? 'Selecting range…' : 'Cut range'}</span>
            </button>
          )}

          {rangeBounds && (
            <div className="flex items-center gap-1.5 rounded-md border border-amber-400/30 bg-amber-500/10 px-2 py-1">
              <span className="font-mono text-[10px] text-amber-200">
                {formatTimecode(rangeBounds.start)} → {formatTimecode(rangeBounds.end)} ({(rangeBounds.end - rangeBounds.start).toFixed(1)}s)
              </span>
              <button
                onClick={() => void applyCut('remove')}
                disabled={cutBusy}
                title="Delete this range from the video; the rest shifts back"
                className="rounded bg-rose-500/80 px-2 py-1 text-[10px] font-semibold text-white transition hover:bg-rose-500 disabled:opacity-40"
              >
                {cutBusy ? 'Cutting…' : 'Cut out'}
              </button>
              <button
                onClick={() => void applyCut('keep')}
                disabled={cutBusy}
                title="Trim the video to only this range"
                className="rounded bg-sky-500/80 px-2 py-1 text-[10px] font-semibold text-white transition hover:bg-sky-500 disabled:opacity-40"
              >
                {cutBusy ? 'Trimming…' : 'Keep only'}
              </button>
              <button
                onClick={() => setSelectedRange(null)}
                disabled={cutBusy}
                className="rounded px-1.5 py-1 text-[10px] text-slate-400 transition hover:text-white disabled:opacity-40"
              >
                ✕
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="flex">
        {/* Track headers */}
        <div className="z-20 w-44 shrink-0 select-none border-r border-white/[0.06] bg-[#0b1018]">
          <div
            className="flex items-center gap-1.5 border-b border-white/[0.06] bg-black/20 px-3 text-[9px] font-semibold uppercase tracking-widest text-slate-600"
            style={{ height: RULER_HEIGHT }}
          >
            <Layers size={10} className="text-slate-600" />
            Tracks
          </div>
          {TRACKS.map((track) => {
            const state = trackStates[track.name];
            const audio = track.name === 'Original Speech' || track.name === 'Dubbed Voice';
            return (
              <div
                key={track.name}
                className="group flex items-center justify-between border-b border-white/[0.04] px-2.5 transition-colors hover:bg-white/[0.02]"
                style={{ height: track.height }}
              >
                <div className="flex min-w-0 items-center gap-1.5">
                  <track.icon size={13} className={`${track.color} shrink-0`} />
                  <span className={`truncate text-[11px] font-medium transition-colors ${state?.hidden ? 'text-slate-600 line-through' : 'text-slate-300 group-hover:text-white'}`}>
                    {track.name}
                  </span>
                </div>
                <div className="flex items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
                  {audio ? (
                    <button
                      onClick={() => onToggleMute(track.name)}
                      title={state?.muted ? 'Unmute' : 'Mute'}
                      className={`rounded p-0.5 transition hover:bg-white/10 ${state?.muted ? 'text-rose-400 opacity-100' : 'text-slate-500'}`}
                    >
                      {state?.muted ? <VolumeX size={11} /> : <Volume2 size={11} />}
                    </button>
                  ) : (
                    <button
                      onClick={() => onToggleHide(track.name)}
                      title={state?.hidden ? 'Show' : 'Hide'}
                      className={`rounded p-0.5 transition hover:bg-white/10 ${state?.hidden ? 'text-amber-400 opacity-100' : 'text-slate-500'}`}
                    >
                      {state?.hidden ? <EyeOff size={11} /> : <Eye size={11} />}
                    </button>
                  )}
                  <button
                    onClick={() => onToggleLock(track.name)}
                    title={state?.locked ? 'Unlock' : 'Lock'}
                    className={`rounded p-0.5 transition hover:bg-white/10 ${state?.locked ? 'text-rose-400 opacity-100' : 'text-slate-600'}`}
                  >
                    {state?.locked ? <Lock size={10} /> : <Unlock size={10} />}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Scrollable canvas */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="timeline-scrollbar relative flex-1 overflow-x-auto overflow-y-hidden bg-[#080d17]"
        >
          <div style={{ width: contentWidth }} className="relative select-none">
            {/* Ruler */}
            <div
              ref={rulerRef}
              onPointerDown={handleRulerPointerDown}
              className={`relative overflow-hidden border-b border-white/[0.06] ${editable && rangeSelectMode ? 'cursor-crosshair bg-[#1a1408]' : 'cursor-pointer bg-[#0a1020]'}`}
              style={{ height: RULER_HEIGHT }}
              title={editable && rangeSelectMode ? 'Drag to select a range, then Cut out or Keep only' : 'Click or drag to scrub · Ctrl+wheel to zoom'}
            >
              {ticks.map((tick) => (
                <div key={`tick-${tick.time}`} className="pointer-events-none absolute top-0 flex flex-col items-center" style={{ left: tick.x }}>
                  <div className={`w-px ${tick.major ? 'h-3 bg-slate-500' : 'h-1.5 bg-slate-800'}`} />
                  {tick.major && (
                    <span className="mt-0.5 -translate-x-1/2 font-mono text-[8px] text-slate-500">{tick.label}</span>
                  )}
                </div>
              ))}
              {rangeBounds && (
                <div
                  className="pointer-events-none absolute inset-y-0 border-x-2 border-amber-400 bg-amber-400/20"
                  style={{ left: rangeBounds.start * pixelsPerSecond, width: Math.max(1, (rangeBounds.end - rangeBounds.start) * pixelsPerSecond) }}
                />
              )}
            </div>

            {/* Track canvas */}
            <div className="relative" style={{ height: TRACK_HEIGHT }}>
              {/* Grid */}
              {ticks.filter((tick) => tick.major).map((tick) => (
                <div
                  key={`grid-${tick.time}`}
                  className="pointer-events-none absolute top-0 bottom-0 w-px bg-white/[0.035]"
                  style={{ left: tick.x }}
                />
              ))}

              {trackRows}

              {/* Content past the end of the media */}
              {duration > 0 && duration * pixelsPerSecond < contentWidth && (
                <div
                  className="pointer-events-none absolute inset-y-0 right-0 z-10 bg-[#05080f]/75"
                  style={{ left: duration * pixelsPerSecond }}
                />
              )}

              {rangeBounds && (
                <div
                  className="pointer-events-none absolute inset-y-0 z-20 border-x-2 border-amber-400/90 bg-amber-400/[0.12]"
                  style={{ left: rangeBounds.start * pixelsPerSecond, width: Math.max(1, (rangeBounds.end - rangeBounds.start) * pixelsPerSecond) }}
                >
                  <div className="absolute inset-y-0 left-0 w-0.5 -translate-x-1/2 bg-amber-300" />
                  <div className="absolute inset-y-0 right-0 w-0.5 translate-x-1/2 bg-amber-300" />
                </div>
              )}

              {snapGuide != null && (
                <div
                  className="pointer-events-none absolute top-0 bottom-0 z-20 w-px bg-cyan-300"
                  style={{ left: snapGuide * pixelsPerSecond }}
                >
                  <div className="absolute -top-0.5 left-1/2 h-1.5 w-1.5 -translate-x-1/2 rotate-45 bg-cyan-300" />
                </div>
              )}

              {segments.length === 0 && dubSegments.length === 0 && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <div className="rounded-lg border border-white/10 bg-slate-900/80 px-4 py-2 text-[11px] text-slate-500">
                    Import a video and run Transcribe to populate the timeline.
                  </div>
                </div>
              )}
            </div>

            {/* Playhead (painted imperatively while playing) */}
            <div
              ref={playheadRef}
              className="pointer-events-none absolute top-0 bottom-0 z-40 will-change-transform"
            >
              <div className="absolute left-0 top-0 h-0 w-0 -translate-x-1/2" style={{ borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '8px solid #fb7185' }} />
              <div className="absolute bottom-0 top-2 left-0 w-px -translate-x-1/2 bg-rose-400/90 shadow-[0_0_6px_rgba(251,113,133,0.55)]" />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] bg-[#0e1526] px-4 py-1.5 text-[10px] text-slate-500">
        <div className="flex items-center gap-3">
          <span className="font-mono text-slate-400">{formatTimecode(currentTime)}</span>
          <span>{pixelLabel}</span>
          <span>{segments.length} speech · {dubSegments.length} dubbed{videoChunks.length > 0 ? ` · ${videoChunks.length} parts` : ''}</span>
          {isPlaying && <span className="text-emerald-400">Playing</span>}
        </div>
        <div className="hidden items-center gap-3 text-slate-600 lg:flex">
          <span><kbd className="rounded border border-white/10 bg-black/30 px-1 py-0.5 text-slate-400">Space</kbd> play</span>
          <span><kbd className="rounded border border-white/10 bg-black/30 px-1 py-0.5 text-slate-400">←/→</kbd> 1s</span>
          <span><kbd className="rounded border border-white/10 bg-black/30 px-1 py-0.5 text-slate-400">Ctrl</kbd>+wheel zoom</span>
          <span><kbd className="rounded border border-white/10 bg-black/30 px-1 py-0.5 text-slate-400">S</kbd> snap</span>
          <span><kbd className="rounded border border-white/10 bg-black/30 px-1 py-0.5 text-slate-400">F</kbd> fit</span>
          <span>{mode === 'EDIT' ? 'Drag clips to move · edges to trim · click to edit' : 'Click ruler to scrub · click a clip to select'}</span>
        </div>
      </div>
    </div>
  );
}
