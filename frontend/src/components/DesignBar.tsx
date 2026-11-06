import { BACKGROUND_MODES, DESIGN_PLATFORMS, useDesignStore } from '../store/useDesignStore';
import type { BackgroundMode, DesignPlatform } from '../store/useDesignStore';

/** Small pill button used by the segmented controls. */
function Choice({ active, onClick, title, children }: {
  active: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={title}
      className={`rounded px-1.5 py-1 text-[10px] font-semibold transition ${
        active ? 'bg-sky-500/25 text-sky-100' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
      }`}
    >
      {children}
    </button>
  );
}

/**
 * The controls for the canvas that sits over both previews.
 *
 * It only edits normalised values — the preview components and the FFmpeg export
 * both read them — so nothing here depends on the current preview's pixel size.
 */
export function DesignBar() {
  const {
    platform, editing, background, video, subtitle, blurBox, safeZone,
    setPlatform, setEditing, setBackground, setVideo, setSubtitle, setBlurBox, toggleSafeZone, reset,
  } = useDesignStore();

  return (
    <section className="rounded-2xl border border-white/10 bg-slate-900/70 px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Canvas</span>
        <div className="flex items-center overflow-hidden rounded-md border border-white/10" role="group" aria-label="Platform frame">
          {(Object.keys(DESIGN_PLATFORMS) as DesignPlatform[]).map((key) => (
            <Choice key={key} active={platform === key} onClick={() => setPlatform(key)} title={DESIGN_PLATFORMS[key].hint}>
              {DESIGN_PLATFORMS[key].label}
            </Choice>
          ))}
        </div>

        <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Background</span>
        <div className="flex items-center overflow-hidden rounded-md border border-white/10" role="group" aria-label="Background">
          {(Object.keys(BACKGROUND_MODES) as BackgroundMode[]).map((key) => (
            <Choice key={key} active={background.mode === key} onClick={() => setBackground({ mode: key })}>
              {BACKGROUND_MODES[key]}
            </Choice>
          ))}
        </div>

        {background.mode === 'blur' && (
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400">
            Blur
            <input
              type="range" min={0} max={50} step={1} value={background.blur}
              onChange={(event) => setBackground({ blur: Number(event.target.value) })}
              className="h-1 w-20 cursor-pointer" aria-label="Background blur"
            />
            <span className="w-4 text-right tabular-nums text-slate-300">{background.blur}</span>
          </label>
        )}
        {background.mode === 'color' && (
          <input
            type="color" value={background.color}
            onChange={(event) => setBackground({ color: event.target.value })}
            className="h-5 w-7 cursor-pointer rounded border border-white/10 bg-transparent"
            aria-label="Background colour"
          />
        )}
        {background.mode === 'gradient' && (
          <span className="flex items-center gap-1">
            <input
              type="color" value={background.from}
              onChange={(event) => setBackground({ from: event.target.value })}
              className="h-5 w-7 cursor-pointer rounded border border-white/10 bg-transparent"
              aria-label="Gradient start"
            />
            <input
              type="color" value={background.to}
              onChange={(event) => setBackground({ to: event.target.value })}
              className="h-5 w-7 cursor-pointer rounded border border-white/10 bg-transparent"
              aria-label="Gradient end"
            />
          </span>
        )}
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <label className="flex items-center gap-1.5 text-[10px] text-slate-400">
          Rotate
          <input
            type="range" min={-180} max={180} step={1} value={video.rotation}
            onChange={(event) => setVideo({ rotation: Number(event.target.value) })}
            className="h-1 w-24 cursor-pointer" aria-label="Video rotation"
          />
          <span className="w-8 text-right tabular-nums text-slate-300">{video.rotation}°</span>
        </label>

        <div className="flex items-center overflow-hidden rounded-md border border-white/10">
          <Choice active={video.flipH} onClick={() => setVideo({ flipH: !video.flipH })} title="Mirror left to right">Flip H</Choice>
          <Choice active={video.flipV} onClick={() => setVideo({ flipV: !video.flipV })} title="Mirror top to bottom">Flip V</Choice>
        </div>

        <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-slate-300">
          <input
            type="checkbox" checked={subtitle.enabled}
            onChange={(event) => setSubtitle({ enabled: event.target.checked })}
            className="h-3 w-3 accent-sky-500"
          />
          Caption
        </label>
        <label className="flex items-center gap-1.5 text-[10px] text-slate-400" title="Each cue shows at most this many words and advances through the line while it plays (0 = the whole line)">
          Words/cue
          <input
            type="range" min={0} max={12} step={1} value={subtitle.wordsPerLine}
            onChange={(event) => setSubtitle({ wordsPerLine: Number(event.target.value) })}
            className="h-1 w-16 cursor-pointer"
            aria-label="Words per subtitle cue"
          />
          <span className="w-5 text-right tabular-nums text-slate-300">
            {subtitle.wordsPerLine === 0 ? 'all' : subtitle.wordsPerLine}
          </span>
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-slate-300"
               title="Hides subtitles or logos burned into the source video">
          <input
            type="checkbox" checked={blurBox.enabled}
            onChange={(event) => setBlurBox({ enabled: event.target.checked })}
            className="h-3 w-3 accent-rose-500"
          />
          Blur box
        </label>
        {blurBox.enabled && (
          <label className="flex items-center gap-1.5 text-[10px] text-slate-400">
            Strength
            <input
              type="range" min={0} max={50} step={1} value={blurBox.strength}
              onChange={(event) => setBlurBox({ strength: Number(event.target.value) })}
              className="h-1 w-16 cursor-pointer" aria-label="Blur box strength"
            />
            <span className="w-4 text-right tabular-nums text-slate-300">{blurBox.strength}</span>
          </label>
        )}
        <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-slate-300">
          <input
            type="checkbox" checked={safeZone} onChange={toggleSafeZone}
            className="h-3 w-3 accent-amber-500"
          />
          Safe zone
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-slate-300" title="Show the drag handles and snap guides on the previews">
          <input
            type="checkbox" checked={editing} onChange={(event) => setEditing(event.target.checked)}
            className="h-3 w-3 accent-emerald-500"
          />
          Handles
        </label>

        <button
          type="button" onClick={reset}
          className="ml-auto rounded-md border border-white/10 bg-slate-900 px-2 py-1 text-[10px] font-medium text-slate-300 transition hover:bg-slate-800"
          title="Reset the video position, scale, rotation, flips, caption and blur box"
        >
          Reset
        </button>
      </div>
    </section>
  );
}
