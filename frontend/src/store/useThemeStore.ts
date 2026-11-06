import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'dark' | 'light';

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggle: () => void;
}

/**
 * The whole UI is written with Tailwind's slate palette; the light theme is a
 * set of overrides in `index.css` keyed on `[data-theme='light']`, so no
 * component had to be rewritten to get it. This store only remembers the choice
 * and writes the attribute the CSS looks for.
 */
export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      setTheme: (theme) => set({ theme }),
      toggle: () => set({ theme: get().theme === 'dark' ? 'light' : 'dark' }),
    }),
    { name: 'dubvibe-theme', version: 1 },
  ),
);

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}
