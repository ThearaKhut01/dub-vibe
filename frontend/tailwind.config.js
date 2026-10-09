/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#101828',
        panelAlt: '#0f172a',
        accent: '#38bdf8',
        accentStrong: '#8b5cf6',
        success: '#34d399',
        warning: '#fbbf24',
        danger: '#f87171',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(56, 189, 248, 0.35), 0 18px 40px rgba(14, 165, 233, 0.25)',
      },
    },
  },
  plugins: [],
};
