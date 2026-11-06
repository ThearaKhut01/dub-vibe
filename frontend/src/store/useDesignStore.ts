import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Canvas/design state for the preview and, through it, the export.
 *
 * Every geometry value is normalised so the preview and FFmpeg can use the same
 * numbers without knowing each other's pixels:
 *
 * - `video.x` / `video.y` — the video's centre as an offset from the frame centre,
 *   in frame widths/heights (-1 = left/top edge, 0 = centred, 1 = right/bottom).
 * - `video.size` — the video's width as a fraction of the frame width.
 * - `subtitle.x` / `subtitle.y` — 0..1 across/inside the frame.
 * - `subtitle.size` — font size as a fraction of the frame height.
 * - `background.blur` — the slider's 0..50 intensity (a radius, not a layout size).
 * - `video.rotation` — degrees, because an angle has no pixel or frame unit.
 */

export type DesignPlatform = 'original' | 'vertical' | 'standard' | 'square';
export type BackgroundMode = 'blur' | 'color' | 'gradient';

export interface DesignPlatformInfo {
  label: string;
  hint: string;
  /** null = follow the source video's own shape. */
  ratio: number | null;
}

export const DESIGN_PLATFORMS: Record<DesignPlatform, DesignPlatformInfo> = {
  original: { label: 'Original', hint: 'The video’s own aspect ratio', ratio: null },
  vertical: { label: '9:16', hint: 'TikTok, Reels, Shorts', ratio: 9 / 16 },
  standard: { label: '16:9', hint: 'YouTube, Facebook', ratio: 16 / 9 },
  square: { label: '1:1', hint: 'Instagram feed', ratio: 1 },
};

export const BACKGROUND_MODES: Record<BackgroundMode, string> = {
  blur: 'Blur',
  color: 'Colour',
  gradient: 'Gradient',
};

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));

const clamp01 = (value: number): number => clamp(value, 0, 1);

interface DesignState {
  platform: DesignPlatform;
  /** Show handles, snap guides and the subtitle box. */
  editing: boolean;
  background: {
    mode: BackgroundMode;
    /** 0–50, the slider's own unit. */
    blur: number;
    color: string;
    from: string;
    to: string;
  };
  video: {
    x: number;
    y: number;
    size: number;
    rotation: number;
    flipH: boolean;
    flipV: boolean;
  };
  subtitle: {
    enabled: boolean;
    x: number;
    y: number;
    size: number;
    /** 0 = keep each cue on one line; 1–12 = wrap the text after that many words. */
    wordsPerLine: number;
  };
  /**
   * A blurred rectangle over the frame, for hiding subtitles or logos that are
   * burned into the source video. Same normalised space as the caption.
   */
  blurBox: {
    enabled: boolean;
    x: number;
    y: number;
    w: number;
    h: number;
    /** 0–50, the same scale as the background blur. */
    strength: number;
  };
  safeZone: boolean;

  setPlatform: (platform: DesignPlatform) => void;
  setEditing: (editing: boolean) => void;
  setBackground: (update: Partial<DesignState['background']>) => void;
  setVideo: (update: Partial<DesignState['video']>) => void;
  setSubtitle: (update: Partial<DesignState['subtitle']>) => void;
  setBlurBox: (update: Partial<DesignState['blurBox']>) => void;
  toggleSafeZone: () => void;
  reset: () => void;
}

const DEFAULT_VIDEO = { x: 0, y: 0, size: 1, rotation: 0, flipH: false, flipV: false };
const DEFAULT_SUBTITLE = { enabled: true, x: 0.5, y: 0.88, size: 0.06, wordsPerLine: 0 };
// Sits over the letterbox bar where burned-in subtitles usually are; wide and
// short, so it reads as a strip rather than a blank rectangle.
const DEFAULT_BLUR_BOX = { enabled: false, x: 0.05, y: 0.7, w: 0.9, h: 0.12, strength: 20 };
const DEFAULT_BACKGROUND = {
  mode: 'blur' as BackgroundMode,
  blur: 20,
  color: '#0b1220',
  from: '#1e293b',
  to: '#0ea5e9',
};

export const useDesignStore = create<DesignState>()(
  persist(
    (set) => ({
      platform: 'original',
      editing: true,
      background: { ...DEFAULT_BACKGROUND },
      video: { ...DEFAULT_VIDEO },
      subtitle: { ...DEFAULT_SUBTITLE },
      blurBox: { ...DEFAULT_BLUR_BOX },
      safeZone: false,

      setPlatform: (platform) => set({ platform }),
      setEditing: (editing) => set({ editing }),
      setBackground: (update) => set((state) => ({
        background: {
          ...state.background,
          ...update,
          ...(update.blur !== undefined ? { blur: clamp(update.blur, 0, 50) } : {}),
        },
      })),
      setVideo: (update) => set((state) => ({
        video: {
          ...state.video,
          ...update,
          ...(update.x !== undefined ? { x: clamp(update.x, -1, 1) } : {}),
          ...(update.y !== undefined ? { y: clamp(update.y, -1, 1) } : {}),
          ...(update.size !== undefined ? { size: clamp(update.size, 0.05, 1) } : {}),
          ...(update.rotation !== undefined ? { rotation: clamp(update.rotation, -180, 180) } : {}),
        },
      })),
      setSubtitle: (update) => set((state) => ({
        subtitle: {
          ...state.subtitle,
          ...update,
          ...(update.x !== undefined ? { x: clamp01(update.x) } : {}),
          ...(update.y !== undefined ? { y: clamp01(update.y) } : {}),
          ...(update.size !== undefined ? { size: clamp(update.size, 0.02, 0.2) } : {}),
          ...(update.wordsPerLine !== undefined
            ? { wordsPerLine: Math.round(clamp(update.wordsPerLine, 0, 12)) }
            : {}),
        },
      })),
      setBlurBox: (update) => set((state) => {
        const merged = { ...state.blurBox, ...update };
        const strength = clamp(merged.strength, 0, 50);
        // Keep the rectangle inside the frame while it is dragged or resized.
        const x = clamp(merged.x, 0, 0.98);
        const y = clamp(merged.y, 0, 0.98);
        const w = clamp(merged.w, 0.02, 1 - x);
        const h = clamp(merged.h, 0.02, 1 - y);
        return { blurBox: { ...merged, x, y, w, h, strength } };
      }),
      toggleSafeZone: () => set((state) => ({ safeZone: !state.safeZone })),
      // Layout only: the platform and the blur keep their settings, because those
      // describe the output, not where the video sits inside it.
      reset: () => set({
        video: { ...DEFAULT_VIDEO },
        subtitle: { ...DEFAULT_SUBTITLE },
        blurBox: { ...DEFAULT_BLUR_BOX },
      }),
    }),
    {
      name: 'dubvibe-design',
      version: 1,
      partialize: (state) => ({
        platform: state.platform,
        editing: state.editing,
        background: state.background,
        video: state.video,
        subtitle: state.subtitle,
        blurBox: state.blurBox,
        safeZone: state.safeZone,
      }),
    },
  ),
);

/** Cut a line into cues of at most `words` words (0 = the whole line). */
export function splitCueWords(text: string, words: number): string[] {
  const body = String(text ?? '').trim();
  if (!body) return [];
  if (words <= 0) return [body];
  const parts = body.split(/\s+/);
  const chunks: string[] = [];
  for (let index = 0; index < parts.length; index += words) {
    chunks.push(parts.slice(index, index + words).join(' '));
  }
  return chunks;
}

/**
 * The cue that should be on screen at `time`, given the line's own window.
 * Cues are spread evenly over the line, so the text advances while it plays and
 * the last word still lands before the line ends.
 */
export function captionAtTime(text: string, words: number, time: number, start: number, end: number): string {
  const chunks = splitCueWords(text, words);
  if (chunks.length <= 1) return chunks[0] ?? '';
  const duration = Math.max(0.001, end - start);
  const ratio = Math.min(0.999, Math.max(0, (time - start) / duration));
  return chunks[Math.min(chunks.length - 1, Math.floor(ratio * chunks.length))];
}

/** The payload sent to the backend export, with the same normalised numbers. */
export function designPayload(state: DesignState): Record<string, unknown> {
  return {
    platform: state.platform,
    background: { ...state.background },
    video: { ...state.video },
    subtitle: { ...state.subtitle },
    blur_box: { ...state.blurBox },
    safe_zone: state.safeZone,
  };
}
