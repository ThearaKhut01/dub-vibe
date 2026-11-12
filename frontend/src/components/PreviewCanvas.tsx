import { useEffect, useRef, useState } from 'react';
import type React from 'react';
import { DESIGN_PLATFORMS, useDesignStore } from '../store/useDesignStore';

interface PreviewCanvasProps {
  /** The media the canvas shows (empty string = placeholder). */
  src: string;
  /** The source video's width/height. Shared, so both previews frame identically. */
  sourceRatio: number;
  /** The element the rest of the app (timeline, sync, seek) keeps using. */
  videoRef: React.RefObject<HTMLVideoElement>;
  onLoadedMetadata?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  onTimeUpdate?: (event: React.SyntheticEvent<HTMLVideoElement>) => void;
  onPlay?: () => void;
  onPause?: () => void;
  /** Current subtitle line, drawn where the export will draw it. */
  subtitleText?: string;
  /** Rendered when `src` is empty. */
  empty?: React.ReactNode;
}

const SNAP_PX = 8;
/** The blur slider is calibrated at a 720 px-wide frame; both the preview and
 * the export scale it by width / 720 so the bars look the same at any size. */
const BLUR_REFERENCE_WIDTH = 720;

function capture(event: React.PointerEvent<HTMLElement>): void {
  try {
    event.currentTarget.setPointerCapture?.(event.pointerId);
  } catch {
    // A synthetic or already-released pointer cannot be captured; the drag still
    // works while the pointer stays over the element.
  }
}

/**
 * The editing surface: one platform frame (9:16, 16:9, 1:1 or the source's own
 * shape) with the video, its background, the caption and the guide overlays.
 *
 * It is deliberately a DOM/CSS compositor rather than a `<canvas>`: the app
 * already plays a real `<video>` here (timeline seeking, synced playback, the
 * dubbed export), and CSS transforms let the preview and the FFmpeg filter graph
 * consume the same normalised numbers. Nothing is measured into the store — the
 * frame is the only thing that knows about pixels.
 */
export function PreviewCanvas({
  src,
  sourceRatio,
  videoRef,
  onLoadedMetadata,
  onTimeUpdate,
  onPlay,
  onPause,
  subtitleText,
  empty,
}: PreviewCanvasProps) {
  const { platform, editing, background, video, subtitle, blurBox, ocrBox, safeZone, setVideo, setBlurBox, setOcrBox } = useDesignStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const backgroundRef = useRef<HTMLVideoElement>(null);

  const [box, setBox] = useState<{ width: number; height: number } | null>(null);
  const [guides, setGuides] = useState<{ x: boolean; y: boolean }>({ x: false, y: false });

  // A fixed platform frame wins; "original" follows the source's own shape. This
  // ratio is passed in, not measured per card: the dubbed preview has no video
  // until an export exists, and it used to fall back to 16:9 while the source
  // card showed the real shape.
  const platformRatio = DESIGN_PLATFORMS[platform].ratio ?? sourceRatio;

  // Largest frame of this ratio that fits the card — the same "no black bars"
  // rule the preview had before, now driven by the platform.
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const measure = () => {
      const width = element.clientWidth;
      const height = element.clientHeight;
      if (!width || !height) return;
      const fitWidth = Math.min(width, height * platformRatio);
      setBox({ width: Math.floor(fitWidth), height: Math.floor(fitWidth / platformRatio) });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [platformRatio]);

  // The background is the same clip, blurred, muted and kept in step with the
  // main video so the bars never show a frozen or drifting picture.
  const syncBackground = (time: number) => {
    const element = backgroundRef.current;
    if (!element) return;
    if (Math.abs(element.currentTime - time) > 0.35) element.currentTime = time;
  };

  useEffect(() => {
    const element = backgroundRef.current;
    if (!element || !src) return;
    element.load();
  }, [src]);

  const dragRef = useRef<{ startX: number; startY: number; x: number; y: number } | null>(null);

  const onVideoPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!editing || !box) return;
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    dragRef.current = { startX: event.clientX, startY: event.clientY, x: video.x, y: video.y };
  };

  const onVideoPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = dragRef.current;
    if (!start || !box) return;
    const nextX = start.x + (event.clientX - start.startX) / (box.width * 0.5);
    const nextY = start.y + (event.clientY - start.startY) / (box.height * 0.5);
    const snapX = Math.abs(nextX) < SNAP_PX / (box.width * 0.5) ? 0 : nextX;
    const snapY = Math.abs(nextY) < SNAP_PX / (box.height * 0.5) ? 0 : nextY;
    setGuides({ x: snapX === 0 && nextX !== start.x, y: snapY === 0 && nextY !== start.y });
    setVideo({ x: snapX, y: snapY });
  };

  const endVideoDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    setGuides({ x: false, y: false });
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  const scaleRef = useRef<{ distance: number; size: number } | null>(null);

  const onHandlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!box) return;
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    const frame = event.currentTarget.parentElement?.getBoundingClientRect();
    if (!frame) return;
    const centreX = frame.left + frame.width / 2;
    const centreY = frame.top + frame.height / 2;
    scaleRef.current = {
      distance: Math.max(12, Math.hypot(event.clientX - centreX, event.clientY - centreY)),
      size: video.size,
    };
  };

  const onHandlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = scaleRef.current;
    if (!start) return;
    const frame = event.currentTarget.parentElement?.getBoundingClientRect();
    if (!frame) return;
    const centreX = frame.left + frame.width / 2;
    const centreY = frame.top + frame.height / 2;
    const distance = Math.hypot(event.clientX - centreX, event.clientY - centreY);
    setVideo({ size: start.size * (distance / start.distance) });
  };

  const endHandleDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    scaleRef.current = null;
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  const captionRef = useRef<{ startX: number; startY: number; x: number; y: number } | null>(null);
  const onCaptionPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!editing || !box) return;
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    captionRef.current = { startX: event.clientX, startY: event.clientY, x: subtitle.x, y: subtitle.y };
  };

  const onCaptionPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = captionRef.current;
    if (!start || !box) return;
    const store = useDesignStore.getState();
    store.setSubtitle({
      x: start.x + (event.clientX - start.startX) / box.width,
      y: start.y + (event.clientY - start.startY) / box.height,
    });
  };

  const endCaptionDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    captionRef.current = null;
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  const captionScaleRef = useRef<{ startY: number; size: number } | null>(null);

  const onCaptionResizeDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    captionScaleRef.current = { startY: event.clientY, size: subtitle.size };
  };

  const onCaptionResizeMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = captionScaleRef.current;
    if (!start || !box) return;
    // Do not let the parent caption drag run on the same pointer move.
    event.stopPropagation();
    useDesignStore.getState().setSubtitle({
      size: start.size + (event.clientY - start.startY) / box.height,
    });
  };

  const endCaptionResize = (event: React.PointerEvent<HTMLDivElement>) => {
    captionScaleRef.current = null;
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  // --- blur box: dragged and resized in the same normalised frame space ------
  const blurDragRef = useRef<{ startX: number; startY: number; x: number; y: number } | null>(null);

  const onBlurPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!editing || !box) return;
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    blurDragRef.current = { startX: event.clientX, startY: event.clientY, x: blurBox.x, y: blurBox.y };
  };

  const onBlurPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = blurDragRef.current;
    if (!start || !box) return;
    setBlurBox({
      x: start.x + (event.clientX - start.startX) / box.width,
      y: start.y + (event.clientY - start.startY) / box.height,
    });
  };

  const endBlurDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    blurDragRef.current = null;
    const target = event.currentTarget;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  const blurResizeRef = useRef<{
    corner: string; startX: number; startY: number; box: typeof blurBox;
  } | null>(null);

  const onBlurHandleDown = (corner: 'nw' | 'ne' | 'sw' | 'se') => (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    blurResizeRef.current = { corner, startX: event.clientX, startY: event.clientY, box: { ...blurBox } };
  };

  const onBlurHandleMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = blurResizeRef.current;
    if (!start || !box) return;
    const deltaX = (event.clientX - start.startX) / box.width;
    const deltaY = (event.clientY - start.startY) / box.height;
    const next = { ...start.box };
    if (start.corner.includes('e')) next.w = start.box.w + deltaX;
    if (start.corner.includes('s')) next.h = start.box.h + deltaY;
    if (start.corner.includes('w')) { next.x = start.box.x + deltaX; next.w = start.box.w - deltaX; }
    if (start.corner.includes('n')) { next.y = start.box.y + deltaY; next.h = start.box.h - deltaY; }
    setBlurBox(next);
  };

  const endBlurResize = (event: React.PointerEvent<HTMLDivElement>) => {
    blurResizeRef.current = null;
    const target = event.currentTarget;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  // --- OCR box: the region the engine reads, dragged like the blur box -------
  const ocrDragRef = useRef<{ startX: number; startY: number; x: number; y: number } | null>(null);

  const onOcrPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!editing || !box) return;
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    ocrDragRef.current = { startX: event.clientX, startY: event.clientY, x: ocrBox.x, y: ocrBox.y };
  };

  const onOcrPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = ocrDragRef.current;
    if (!start || !box) return;
    setOcrBox({
      x: start.x + (event.clientX - start.startX) / box.width,
      y: start.y + (event.clientY - start.startY) / box.height,
    });
  };

  const endOcrDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    ocrDragRef.current = null;
    const target = event.currentTarget;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  const ocrResizeRef = useRef<{
    corner: string; startX: number; startY: number; box: typeof ocrBox;
  } | null>(null);

  const onOcrHandleDown = (corner: 'nw' | 'ne' | 'sw' | 'se') => (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    capture(event);
    ocrResizeRef.current = { corner, startX: event.clientX, startY: event.clientY, box: { ...ocrBox } };
  };

  const onOcrHandleMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = ocrResizeRef.current;
    if (!start || !box) return;
    const deltaX = (event.clientX - start.startX) / box.width;
    const deltaY = (event.clientY - start.startY) / box.height;
    const next = { ...start.box };
    if (start.corner.includes('e')) next.w = start.box.w + deltaX;
    if (start.corner.includes('s')) next.h = start.box.h + deltaY;
    if (start.corner.includes('w')) { next.x = start.box.x + deltaX; next.w = start.box.w - deltaX; }
    if (start.corner.includes('n')) { next.y = start.box.y + deltaY; next.h = start.box.h - deltaY; }
    setOcrBox(next);
  };

  const endOcrResize = (event: React.PointerEvent<HTMLDivElement>) => {
    ocrResizeRef.current = null;
    const target = event.currentTarget;
    if (target.hasPointerCapture?.(event.pointerId)) target.releasePointerCapture(event.pointerId);
  };

  // Same calibration as the background bars: 0–50 at a 720 px-wide frame.
  const blurPx = (blurBox.strength * (box?.width ?? BLUR_REFERENCE_WIDTH)) / BLUR_REFERENCE_WIDTH;

  // The video box in frame percentages: the frame is 100% × 100%, so the same
  // numbers the export uses describe it here.
  const widthPct = video.size * 100;
  const heightPct = box ? ((video.size * box.width) / sourceRatio / box.height) * 100 : 0;
  const leftPct = 50 + video.x * 50 - widthPct / 2;
  const topPct = 50 + video.y * 50 - heightPct / 2;

  return (
    <div ref={containerRef} className="relative flex h-full w-full items-center justify-center">
      <div
        className="media-surface relative overflow-hidden rounded-xl bg-black shadow-[0_0_0_1px_rgba(255,255,255,0.08)]"
        style={{
          width: box?.width ?? '100%',
          height: box?.height ?? '100%',
          aspectRatio: String(platformRatio),
        }}
      >
        {/* Background: the same clip blurred, or a flat / gradient fill. */}
        {src && background.mode === 'blur' && (
          <video
            ref={backgroundRef}
            src={src}
            muted
            playsInline
            aria-hidden
            tabIndex={-1}
            className="pointer-events-none absolute inset-0 h-full w-full scale-110 object-cover"
            style={{ filter: `blur(${((background.blur * (box?.width ?? BLUR_REFERENCE_WIDTH)) / BLUR_REFERENCE_WIDTH).toFixed(2)}px)` }}
          />
        )}
        {background.mode === 'color' && (
          <div className="absolute inset-0" style={{ background: background.color }} />
        )}
        {background.mode === 'gradient' && (
          <div
            className="absolute inset-0"
            style={{ background: `linear-gradient(135deg, ${background.from}, ${background.to})` }}
          />
        )}

        {/* Foreground video: width is a fraction of the frame, height follows the
            media's own ratio, position is the normalised centre offset. */}
        {src ? (
          <div
            className="absolute"
            style={{
              left: `${50 + video.x * 50}%`,
              top: `${50 + video.y * 50}%`,
              width: `${widthPct}%`,
              transform: 'translate(-50%, -50%)',
            }}
          >
            <video
              ref={videoRef}
              src={src}
              playsInline
              onLoadedMetadata={(event) => {
                // The ratio itself is owned by App (see sourceRatio), so both
                // previews stay in the same frame.
                onLoadedMetadata?.(event);
              }}
              onTimeUpdate={(event) => {
                const element = event.currentTarget;
                syncBackground(element.currentTime);
                onTimeUpdate?.(event);
              }}
              onPlay={() => {
                void backgroundRef.current?.play().catch(() => { });
                onPlay?.();
              }}
              onPause={() => {
                backgroundRef.current?.pause();
                onPause?.();
              }}
              style={{
                transform: `rotate(${video.rotation}deg) scaleX(${video.flipH ? -1 : 1}) scaleY(${video.flipV ? -1 : 1})`,
              }}
              className="block h-auto w-full"
            />
          </div>
        ) : (
          <div className="absolute inset-0">{empty}</div>
        )}

        {/* Blur box: hides subtitles or logos burned into the source. It sits
            above the video but below the caption, so our own text stays sharp. */}
        {src && blurBox.enabled && (
          <>
            <div
              className={`absolute z-20 ${editing ? 'cursor-move border border-dashed border-rose-300/60' : 'pointer-events-none'}`}
              style={{
                left: `${blurBox.x * 100}%`,
                top: `${blurBox.y * 100}%`,
                width: `${blurBox.w * 100}%`,
                height: `${blurBox.h * 100}%`,
                backdropFilter: `blur(${blurPx.toFixed(2)}px)`,
                WebkitBackdropFilter: `blur(${blurPx.toFixed(2)}px)`,
              }}
              onPointerDown={onBlurPointerDown}
              onPointerMove={onBlurPointerMove}
              onPointerUp={endBlurDrag}
              onPointerCancel={endBlurDrag}
              title={editing ? 'Drag to move the blur box' : undefined}
            />
            {editing && (
              <div
                className="absolute z-40 flex h-3.5 w-9 -translate-x-1/2 -translate-y-1/2 cursor-move items-center justify-center rounded-full border border-white bg-rose-500 text-[8px] font-bold leading-none text-white"
                style={{ left: `${(blurBox.x + blurBox.w / 2) * 100}%`, top: `${blurBox.y * 100}%` }}
                onPointerDown={onBlurPointerDown}
                onPointerMove={onBlurPointerMove}
                onPointerUp={endBlurDrag}
                onPointerCancel={endBlurDrag}
                title="Drag to move the blur box"
              >
                ⠿
              </div>
            )}
            {editing && ([
              { key: 'nw', cursor: 'cursor-nwse-resize', style: { left: `${blurBox.x * 100}%`, top: `${blurBox.y * 100}%` } },
              { key: 'ne', cursor: 'cursor-nesw-resize', style: { left: `${(blurBox.x + blurBox.w) * 100}%`, top: `${blurBox.y * 100}%` } },
              { key: 'sw', cursor: 'cursor-nesw-resize', style: { left: `${blurBox.x * 100}%`, top: `${(blurBox.y + blurBox.h) * 100}%` } },
              { key: 'se', cursor: 'cursor-nwse-resize', style: { left: `${(blurBox.x + blurBox.w) * 100}%`, top: `${(blurBox.y + blurBox.h) * 100}%` } },
            ] as const).map((handle) => (
              <div
                key={handle.key}
                className={`absolute z-30 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-sm border border-white bg-rose-400 ${handle.cursor}`}
                style={handle.style}
                onPointerDown={onBlurHandleDown(handle.key)}
                onPointerMove={onBlurHandleMove}
                onPointerUp={endBlurResize}
                onPointerCancel={endBlurResize}
                title="Drag to resize the blur box"
              />
            ))}
          </>
        )}

        {/* OCR box: the region "Test OCR" reads. Amber and dashed so it is never
            confused with the blur box; it draws nothing over the video. */}
        {src && ocrBox.enabled && (
          <>
            <div
              className={`absolute z-20 border-2 border-dashed border-amber-400/80 bg-amber-400/10 ${editing ? 'cursor-move' : 'pointer-events-none'}`}
              style={{
                left: `${ocrBox.x * 100}%`,
                top: `${ocrBox.y * 100}%`,
                width: `${ocrBox.w * 100}%`,
                height: `${ocrBox.h * 100}%`,
              }}
              onPointerDown={onOcrPointerDown}
              onPointerMove={onOcrPointerMove}
              onPointerUp={endOcrDrag}
              onPointerCancel={endOcrDrag}
              title={editing ? 'Drag to move the OCR box' : undefined}
            />
            {editing && (
              <div
                className="absolute z-40 flex h-3.5 w-9 -translate-x-1/2 -translate-y-1/2 cursor-move items-center justify-center rounded-full border border-white bg-amber-500 text-[8px] font-bold leading-none text-white"
                style={{ left: `${(ocrBox.x + ocrBox.w / 2) * 100}%`, top: `${ocrBox.y * 100}%` }}
                onPointerDown={onOcrPointerDown}
                onPointerMove={onOcrPointerMove}
                onPointerUp={endOcrDrag}
                onPointerCancel={endOcrDrag}
                title="Drag to move the OCR box"
              >
                ⠿
              </div>
            )}
            {editing && ([
              { key: 'nw', cursor: 'cursor-nwse-resize', style: { left: `${ocrBox.x * 100}%`, top: `${ocrBox.y * 100}%` } },
              { key: 'ne', cursor: 'cursor-nesw-resize', style: { left: `${(ocrBox.x + ocrBox.w) * 100}%`, top: `${ocrBox.y * 100}%` } },
              { key: 'sw', cursor: 'cursor-nesw-resize', style: { left: `${ocrBox.x * 100}%`, top: `${(ocrBox.y + ocrBox.h) * 100}%` } },
              { key: 'se', cursor: 'cursor-nwse-resize', style: { left: `${(ocrBox.x + ocrBox.w) * 100}%`, top: `${(ocrBox.y + ocrBox.h) * 100}%` } },
            ] as const).map((handle) => (
              <div
                key={handle.key}
                className={`absolute z-30 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-sm border border-white bg-amber-400 ${handle.cursor}`}
                style={handle.style}
                onPointerDown={onOcrHandleDown(handle.key)}
                onPointerMove={onOcrHandleMove}
                onPointerUp={endOcrResize}
                onPointerCancel={endOcrResize}
                title="Drag to resize the OCR box"
              />
            ))}
          </>
        )}

        {/* Editing chrome: outline, corner handles, guides, caption box. */}
        {src && editing && (
          <>
            <div
              className="absolute z-10 cursor-move border border-dashed border-sky-400/50"
              style={{ left: `${leftPct}%`, top: `${topPct}%`, width: `${widthPct}%`, height: `${heightPct}%` }}
              onPointerDown={onVideoPointerDown}
              onPointerMove={onVideoPointerMove}
              onPointerUp={endVideoDrag}
              onPointerCancel={endVideoDrag}
              title="Drag to move the video"
            />
            {([
              { key: 'nw', cursor: 'cursor-nwse-resize', style: { left: `${leftPct}%`, top: `${topPct}%` } },
              { key: 'ne', cursor: 'cursor-nesw-resize', style: { right: `${100 - leftPct - widthPct}%`, top: `${topPct}%` } },
              { key: 'sw', cursor: 'cursor-nesw-resize', style: { left: `${leftPct}%`, bottom: `${100 - topPct - heightPct}%` } },
              { key: 'se', cursor: 'cursor-nwse-resize', style: { right: `${100 - leftPct - widthPct}%`, bottom: `${100 - topPct - heightPct}%` } },
            ]).map((handle) => (
              <div
                key={handle.key}
                className={`absolute z-30 h-3 w-3 rounded-sm border border-white bg-sky-400 ${handle.cursor}`}
                style={handle.style}
                onPointerDown={onHandlePointerDown}
                onPointerMove={onHandlePointerMove}
                onPointerUp={endHandleDrag}
                onPointerCancel={endHandleDrag}
                title="Drag a corner to scale"
              />
            ))}
            {guides.x && <div className="pointer-events-none absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-emerald-400/80" />}
            {guides.y && <div className="pointer-events-none absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-emerald-400/80" />}
          </>
        )}

        {safeZone && (
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute inset-[5%] border border-dashed border-amber-300/50" />
            <div className="absolute inset-x-[5%] bottom-[5%] h-[18%] border border-dashed border-rose-300/40 bg-rose-500/5" />
          </div>
        )}

        {src && subtitle.enabled && (
          <div
            className={`absolute z-40 max-w-[92%] -translate-x-1/2 -translate-y-1/2 whitespace-pre-line text-center font-semibold leading-tight text-white ${editing ? 'cursor-move rounded border border-dashed border-white/40 px-1' : 'pointer-events-none'}`}
            style={{
              left: `${subtitle.x * 100}%`,
              top: `${subtitle.y * 100}%`,
              fontSize: box ? `${Math.max(8, subtitle.size * box.height)}px` : `${subtitle.size * 100}%`,
              textShadow: '0 1px 3px rgba(0,0,0,0.9), 0 0 1px rgba(0,0,0,0.9)',
            }}
            onPointerDown={onCaptionPointerDown}
            onPointerMove={onCaptionPointerMove}
            onPointerUp={endCaptionDrag}
            onPointerCancel={endCaptionDrag}
            title={editing ? 'Drag to place the caption' : undefined}
          >
            {subtitleText || 'Caption preview'}
            {editing && (
              <span
                className="absolute -bottom-1.5 -right-1.5 h-3 w-3 cursor-ns-resize rounded-sm border border-white bg-emerald-400"
                onPointerDown={onCaptionResizeDown}
                onPointerMove={onCaptionResizeMove}
                onPointerUp={endCaptionResize}
                onPointerCancel={endCaptionResize}
                title="Drag to resize the caption"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
