import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, Mic, Plus, Trash2, AlertTriangle, Sparkles } from 'lucide-react';
import type { TimelineSegment } from '../types';
import { useStudioStore } from '../store/useStudioStore';

const apiBaseUrl = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001';

type Gender = 'Male' | 'Female';

interface TimingProblem {
  segment: string;
  kind: 'overlap' | 'invalid' | 'beyond_video';
  message: string;
}

interface DubbingTrackEditorProps {
  projectId: string;
  segments: TimelineSegment[];
  audio: TimelineSegment[];
  targetLanguage: string;
  provider: string;
  currentTime: number;
  videoDuration: number;
  /** Ceiling for the panel, so it never grows past the panel it sits beside. */
  maxHeight?: number;
  /** Re-read that ceiling (called when the slider reaches its end). */
  onMeasureMaxHeight?: () => void;
  onSeek: (time: number) => void;
  onNotice: (message: string) => void;
}

function formatTimecode(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const secs = (safe - minutes * 60).toFixed(2).padStart(5, '0');
  return `${String(minutes).padStart(2, '0')}:${secs}`;
}

export function DubbingTrackEditor({
  projectId,
  segments,
  audio,
  targetLanguage,
  provider,
  currentTime,
  videoDuration,
  maxHeight,
  onMeasureMaxHeight,
  onSeek,
  onNotice,
}: DubbingTrackEditorProps) {
  const generateSegmentAudio = useStudioStore((state) => state.generateSegmentAudio);
  const detectSegmentGenders = useStudioStore((state) => state.detectSegmentGenders);
  const updateSegment = useStudioStore((state) => state.updateSegment);
  const deleteSegment = useStudioStore((state) => state.deleteSegment);

  // `height` is the whole card, header included, so the slider and the panel
  // next to it can be compared directly. The ceiling is that panel's height, and
  // the track OPENS at it: the default is the slider dragged all the way right,
  // so both panels line up without the user touching anything. "At the ceiling"
  // is remembered, so a re-measured Workflow card keeps being matched instead of
  // leaving the track a few pixels short.
  const [height, setHeight] = useState(480);
  const [atCeiling, setAtCeiling] = useState(true);
  const ceiling = Math.max(260, Math.round(maxHeight ?? Math.min(720, window.innerHeight * 0.62)));
  const cardHeight = atCeiling ? ceiling : Math.max(220, Math.min(height, ceiling));
  const [collapsed, setCollapsed] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [genders, setGenders] = useState<Record<string, Gender>>({});
  const [busyId, setBusyId] = useState<string | null>(null);
  const [problems, setProblems] = useState<TimingProblem[]>([]);

  const audioById = useMemo(() => new Map(audio.map((item) => [String(item.id), item])), [audio]);
  const ordered = useMemo(() => [...segments].sort((a, b) => a.start - b.start), [segments]);
  const activeId = useMemo(() => {
    const line = ordered.find((segment) => currentTime >= segment.start && currentTime < segment.end);
    return line ? line.id : null;
  }, [ordered, currentTime]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rowRefs = useRef<Map<string, HTMLTableRowElement>>(new Map());
  const listRef = useRef<HTMLDivElement>(null);
  const lastScrolledRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeId || activeId === lastScrolledRef.current) return;
    lastScrolledRef.current = activeId;
    const row = rowRefs.current.get(activeId);
    const list = listRef.current;
    if (!row || !list) return;
    const rowBox = row.getBoundingClientRect();
    const listBox = list.getBoundingClientRect();
    // Already fully visible inside the track: leave everything alone.
    if (rowBox.top >= listBox.top + 4 && rowBox.bottom <= listBox.bottom - 4) return;
    // Scroll THIS list by hand. `scrollIntoView` also scrolls every scrollable
    // ancestor — it used to jump the whole page down to the track while the
    // preview at the top was playing.
    const offset = rowBox.top < listBox.top
      ? rowBox.top - listBox.top - 8
      : rowBox.bottom - listBox.bottom + 8;
    list.scrollTo({ top: list.scrollTop + offset, behavior: 'smooth' });
  }, [activeId]);
  const totalSeconds = Math.max(videoDuration, ...ordered.map((item) => item.end), 1);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/projects/${projectId}/segment-timing`);
        if (!response.ok || cancelled) return;
        const data = await response.json();
        if (!cancelled) setProblems(data.problems ?? []);
      } catch {
        // Timing warnings are advisory; a failed check must not block editing.
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [projectId, segments, audio]);

  useEffect(() => () => {
    audioRef.current?.pause();
  }, []);

  const [detectingAll, setDetectingAll] = useState(false);

  const detectAll = async () => {
    setDetectingAll(true);
    try {
      const result = await detectSegmentGenders(projectId);
      onNotice(result.message);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'Voice detection failed.');
    } finally {
      setDetectingAll(false);
    }
  };

  const detectAllButton = (
    <button
      onClick={() => void detectAll()}
      disabled={detectingAll || busyId !== null || segments.length === 0}
      className="inline-flex items-center gap-1 rounded-lg border border-violet-500/40 bg-violet-600/20 px-2 py-1 text-[11px] font-medium text-violet-200 hover:bg-violet-600/40 disabled:opacity-40"
      title="Listen to every line, save whether the speaker is male or female, and auto-select the voice buttons"
    >
      <Sparkles size={12} className={detectingAll ? 'animate-pulse' : ''} />
      {detectingAll ? 'Detecting…' : 'Auto-detect all'}
    </button>
  );

  const genderFor = (segment: TimelineSegment): Gender => genders[segment.id] ?? (segment.gender ?? 'Male');

  const rowStatus = (segment: TimelineSegment): 'ready' | 'stale' | 'missing' | 'unsaved' => {
    const draft = drafts[segment.id];
    if (draft !== undefined && draft.trim() !== segment.translation.trim()) return 'unsaved';
    const clip = audioById.get(segment.id);
    if (!clip) return 'missing';
    if (clip.translation.trim() !== segment.translation.trim()) return 'stale';
    return 'ready';
  };

  const saveText = async (segment: TimelineSegment) => {
    const next = (drafts[segment.id] ?? segment.translation).trim();
    if (next === segment.translation.trim()) return;
    setBusyId(segment.id);
    try {
      await updateSegment(projectId, segment.id, { translation: next });
      setDrafts((current) => {
        const copy = { ...current };
        delete copy[segment.id];
        return copy;
      });
      onNotice('Dubbed text saved. Generate audio to update this line.');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'Could not save the text.');
    } finally {
      setBusyId(null);
    }
  };

  const generate = async (segment: TimelineSegment, gender?: Gender | 'auto') => {
    const text = (drafts[segment.id] ?? segment.translation).trim();
    if (!text) {
      onNotice('Write the dubbed text before generating audio.');
      return;
    }
    setBusyId(segment.id);
    try {
      if (text !== segment.translation.trim()) {
        await updateSegment(projectId, segment.id, { translation: text });
        setDrafts((current) => {
          const copy = { ...current };
          delete copy[segment.id];
          return copy;
        });
      }
      const detected = await generateSegmentAudio(projectId, segment.id, {
        gender: gender ?? genderFor(segment),
        provider,
        targetLanguage,
      });
      onNotice(gender === 'auto' && detected
        ? `Audio generated for ${formatTimecode(segment.start)} — detected a ${detected.gender.toLowerCase()} voice.`
        : `Audio generated for ${formatTimecode(segment.start)}.`);
    } catch (error) {
      // The store only replaces the clip after success, so the previous audio is still in place.
      onNotice(error instanceof Error ? error.message : 'Audio generation failed. The previous audio was kept.');
    } finally {
      setBusyId(null);
    }
  };

  const runStructural = async (action: () => Promise<void>, success: string) => {
    setBusyId('structure');
    try {
      await action();
      onNotice(success);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : 'Edit failed.');
    } finally {
      setBusyId(null);
    }
  };

  const addSegmentAfter = (segment: TimelineSegment | null) => {
    const start = segment ? segment.end : 0;
    const nextStart = ordered.find((item) => item.start >= start + 0.1)?.start ?? totalSeconds;
    const end = Math.min(nextStart, start + 2);
    if (end - start < 0.2) {
      onNotice('No room to add a segment here. Shorten or move a neighbouring segment first.');
      return;
    }
    void runStructural(async () => {
      await useStudioStore.getState().createSegment(projectId, { start, end });
    }, 'Segment added. Write its dubbed text, then generate audio.');
  };

  const remove = (segment: TimelineSegment) => {
    void runStructural(() => deleteSegment(projectId, segment.id), 'Segment deleted.');
  };

  return (
    <section
      className="timeline-container flex flex-col rounded-2xl border border-white/[0.08] bg-[#0c1220] shadow-2xl overflow-hidden"
      style={collapsed ? undefined : { height: cardHeight }}
    >
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] bg-[#0e1526] px-4 py-2.5">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Mic size={16} className="text-emerald-400" />
            <span>Dubbing Track</span>
          </div>
          <span className="text-xs text-slate-500">{segments.length} segments</span>
          {problems.length > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-200">
              <AlertTriangle size={12} /> {problems.length} timing {problems.length === 1 ? 'warning' : 'warnings'}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label
            className="flex items-center gap-2 text-[11px] text-slate-400"
            title={`Drag to resize. All the way right it matches the panel beside it (${ceiling} px).`}
          >
            Height
            <input
              type="range"
              min={220}
              max={ceiling}
              step={1}
              value={cardHeight}
              onChange={(event) => {
                const next = Number(event.target.value);
                setHeight(next);
                // The end of the track means "same height as the panel next to me".
                // Re-measure it here: the card can have grown since the last
                // ResizeObserver notification (a step finishing, a label wrapping)
                // and the end of the slider must land exactly on it.
                const atEnd = next >= ceiling;
                setAtCeiling(atEnd);
                if (atEnd) onMeasureMaxHeight?.();
              }}
              className="h-1 w-20 cursor-pointer"
              aria-label="Dubbing track height"
            />
          </label>
          {detectAllButton}
          <button
            onClick={() => addSegmentAfter(null)}
            disabled={busyId !== null}
            className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-slate-900 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-800 disabled:opacity-40"
            title="Add a segment at the start"
          >
            <Plus size={12} /> Add
          </button>
          <button
            onClick={() => setCollapsed((value) => !value)}
            className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-slate-900 px-2 py-1 text-[11px] text-slate-200 hover:bg-slate-800"
            aria-expanded={!collapsed}
          >
            {collapsed ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
            {collapsed ? 'Expand' : 'Collapse'}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="flex min-h-0 flex-1 flex-col">
          {problems.length > 0 && (
            <ul className="max-h-20 overflow-y-auto border-b border-amber-500/20 bg-amber-500/5 px-4 py-1.5 text-[11px] text-amber-200">
              {problems.slice(0, 20).map((problem, index) => (
                <li key={`${problem.segment}-${index}`}>
                  {problem.message}
                </li>
              ))}
            </ul>
          )}

          <div className="grid h-full min-h-0 flex-1 grid-rows-[minmax(0,1fr)] overflow-hidden">
            <div ref={listRef} className="h-full min-h-0 overflow-y-auto">
              {ordered.length === 0 && (
                <p className="p-4 text-xs text-slate-500">
                  No segments yet. Transcribe and translate the video, or add a segment.
                </p>
              )}
              <table className="w-full border-collapse text-xs">
                <thead className="sticky top-0 z-10 border-b border-white/[0.12] bg-[#0a1020] text-left text-[10px] uppercase tracking-[0.12em] text-slate-500 shadow-[0_2px_6px_rgba(0,0,0,0.35)]">
                  <tr>
                    <th className="px-2 py-1.5 font-medium">#</th>
                    <th className="px-2 py-1.5 font-medium">Start</th>
                    <th className="px-2 py-1.5 font-medium">End</th>
                    <th className="px-2 py-1.5 font-medium">Dur</th>
                    <th className="px-2 py-1.5 font-medium">Dubbed text</th>

                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.05]">
                {ordered.map((segment) => {
                  const status = rowStatus(segment);
                  const busy = busyId === segment.id || busyId === 'structure';
                  const draft = drafts[segment.id] ?? segment.translation;
                  return (
                    <tr
                      key={segment.id}
                      ref={(node) => { if (node) rowRefs.current.set(segment.id, node); else rowRefs.current.delete(segment.id); }}
                      data-status={status}
                      onClick={() => onSeek(segment.start)}
                      className={`cursor-pointer align-top transition-colors ${activeId === segment.id ? 'bg-sky-500/15 ring-1 ring-inset ring-sky-400/50' : 'hover:bg-white/[0.03]'}`}
                    >
                      <td className="px-2 py-2 font-mono text-slate-500">{ordered.indexOf(segment) + 1}</td>
                      <td className="px-2 py-2 font-mono text-sky-300 whitespace-nowrap">{formatTimecode(segment.start)}</td>
                      <td className="px-2 py-2 font-mono text-slate-400 whitespace-nowrap">{formatTimecode(segment.end)}</td>
                      <td className="px-2 py-2 font-mono text-slate-500 whitespace-nowrap">{(segment.end - segment.start).toFixed(2)}s</td>
                      <td className="px-2 py-2" onClick={(event) => event.stopPropagation()}>
                        <div className="flex items-start gap-2">
                          <textarea
                            value={draft}
                            onChange={(event) => setDrafts((current) => ({ ...current, [segment.id]: event.target.value }))}
                            onBlur={() => void saveText(segment)}
                            rows={2}
                            className="min-w-0 flex-1 resize-y rounded-md border border-white/10 bg-slate-950 px-2 py-1 text-xs text-slate-100 focus:border-sky-500/60 focus:outline-none"
                            aria-label={`Dubbed text ${formatTimecode(segment.start)}`}
                          />
                          <div className="flex shrink-0 flex-col gap-1">
                            <div className="inline-flex overflow-hidden rounded-md border border-white/10" role="group" aria-label="Speaker voice (Auto listens to the line's audio)">
                              <button
                                onClick={() => void generate(segment, 'auto')}
                                disabled={busy}
                                className="inline-flex items-center gap-1 bg-violet-600/80 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-600 disabled:opacity-40"
                                title="Detect the speaker gender from this line's audio, save it, and generate the voice"
                              >
                                <Sparkles size={11} /> Auto
                              </button>
                              {(['Male', 'Female'] as Gender[]).map((gender) => (
                                <button
                                  key={gender}
                                  onClick={() => { setGenders((current) => ({ ...current, [segment.id]: gender })); void updateSegment(projectId, segment.id, { gender }).catch((error) => onNotice(error instanceof Error ? error.message : 'Could not save the voice choice.')); }}
                                  className={`px-2 py-1 text-[11px] ${genderFor(segment) === gender ? 'bg-sky-500 text-slate-950' : 'bg-slate-900 text-slate-400 hover:text-slate-200'}`}
                                  aria-pressed={genderFor(segment) === gender}
                                >
                                  {gender === 'Male' ? 'Male voice' : 'Female voice'}
                                </button>
                              ))}
                            </div>
                            <button
                              onClick={() => void generate(segment)}
                              disabled={busy || !draft.trim()}
                              className="inline-flex items-center justify-center gap-1 rounded-md bg-gradient-to-r from-sky-500 to-violet-500 px-2.5 py-1 text-[11px] font-semibold text-white disabled:opacity-40"
                            >
                              {busyId === segment.id ? <Loader2 size={12} className="animate-spin" /> : <Mic size={12} />}
                              Generate TTS
                            </button>
                            <button
                              onClick={() => remove(segment)}
                              disabled={busy}
                              className="inline-flex items-center justify-center gap-1 rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-[11px] text-rose-200 disabled:opacity-40"
                            >
                              <Trash2 size={11} /> Delete
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
