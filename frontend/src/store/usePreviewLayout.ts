import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Preview layout: the one row that holds the Source and Dubbed players.
 *
 * Everything here is shared by the two cards because they sit side by side:
 * `height` is the row's height (the splitter under it drags this), `share` is
 * the fraction of the row given to the Source card (the vertical splitter drags
 * this), and `size` only records which S/M/L/Fit preset produced the current
 * height so the buttons can show what is active ('custom' after a drag).
 */

export type PreviewSize = 'S' | 'M' | 'L' | 'Fit';
export type PreviewSizeState = PreviewSize | 'custom';

/** The row can never be shorter than this — the header, frame and controls need it. */
export const PREVIEW_MIN_HEIGHT = 180;
/** Each card keeps at least a quarter of the row. */
export const PREVIEW_MIN_SHARE = 0.25;

export const PREVIEW_PRESETS: Record<PreviewSize, number> = { S: 220, M: 300, L: 420, Fit: 0 };

/** 70% of the window height, as required by the splitter's upper bound. */
export const maxPreviewHeight = (): number => {
  if (typeof window === 'undefined') return 480;
  return Math.max(PREVIEW_MIN_HEIGHT, Math.round(window.innerHeight * 0.7));
};

export const clampPreviewHeight = (value: number): number =>
  Math.max(PREVIEW_MIN_HEIGHT, Math.min(Math.round(value), maxPreviewHeight()));

export const clampPreviewShare = (value: number): number =>
  Math.max(PREVIEW_MIN_SHARE, Math.min(value, 1 - PREVIEW_MIN_SHARE));

const DEFAULT_HEIGHT = PREVIEW_PRESETS.M;

interface PreviewLayoutState {
  height: number;
  share: number;
  size: PreviewSizeState;
  syncPlayback: boolean;
  /** Absolute setter; a drag makes the size 'custom' because no preset matches. */
  setHeight: (height: number) => void;
  /** Relative setter for the pointer drag. */
  bumpHeight: (delta: number) => void;
  setShare: (share: number) => void;
  bumpShare: (delta: number) => void;
  applySize: (size: PreviewSize) => void;
  setSyncPlayback: (value: boolean) => void;
  /** Keep the row inside the window after a resize / on restore. */
  fitToViewport: () => void;
  /** Sizes only: the sync toggle is a playback preference, not layout. */
  reset: () => void;
}

export const usePreviewLayout = create<PreviewLayoutState>()(
  persist(
    (set) => ({
      height: DEFAULT_HEIGHT,
      share: 0.5,
      size: 'M',
      syncPlayback: false,

      setHeight: (height) => set({ height: clampPreviewHeight(height), size: 'custom' }),
      bumpHeight: (delta) => set((state) => ({
        height: clampPreviewHeight(state.height + delta),
        size: 'custom',
      })),
      setShare: (share) => set({ share: clampPreviewShare(share) }),
      bumpShare: (delta) => set((state) => ({ share: clampPreviewShare(state.share + delta) })),
      applySize: (size) => set({
        size,
        height: clampPreviewHeight(size === 'Fit' ? maxPreviewHeight() : PREVIEW_PRESETS[size]),
      }),
      setSyncPlayback: (value) => set({ syncPlayback: value }),
      fitToViewport: () => set((state) => ({ height: clampPreviewHeight(state.height) })),
      reset: () => set({ height: DEFAULT_HEIGHT, share: 0.5, size: 'M' }),
    }),
    {
      name: 'dubvibe-preview-layout',
      version: 1,
      // Only the values: the actions are rebuilt on every load.
      partialize: (state) => ({
        height: state.height,
        share: state.share,
        size: state.size,
        syncPlayback: state.syncPlayback,
      }),
    },
  ),
);
