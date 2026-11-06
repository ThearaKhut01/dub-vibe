import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const ZOOM_MIN = 0.75;
export const ZOOM_MAX = 1.5;
export const ZOOM_STEP = 0.05;

const round = (value: number): number => Math.round(value * 100) / 100;
const clamp = (value: number): number => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, round(value)));

interface ZoomState {
  /** 1 = 100 %. Applied as CSS `zoom` on #root (see index.css). */
  zoom: number;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
}

/**
 * Page zoom.
 *
 * CSS `zoom` scales the whole UI — text, spacing and the canvas — while the
 * layout keeps its proportions, which is what a browser's Ctrl +/- does. The
 * shell's two viewport-sized boxes are compensated in index.css so the window
 * still fits exactly one screen at any level.
 */
export const useZoomStore = create<ZoomState>()(
  persist(
    (set, get) => ({
      zoom: 1,
      zoomIn: () => set({ zoom: clamp(get().zoom + ZOOM_STEP) }),
      zoomOut: () => set({ zoom: clamp(get().zoom - ZOOM_STEP) }),
      reset: () => set({ zoom: 1 }),
    }),
    { name: 'dubvibe-zoom', version: 1 },
  ),
);

export function applyZoom(zoom: number): void {
  document.documentElement.style.setProperty('--app-zoom', String(clamp(zoom)));
}
