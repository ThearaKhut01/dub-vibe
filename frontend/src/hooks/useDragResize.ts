import { useCallback, useRef, useState } from 'react';
import type React from 'react';

interface DragResize {
  /** Movement since the previous pointer event, in CSS pixels. */
  onDelta: (deltaX: number, deltaY: number) => void;
}

/**
 * Pointer drag for a resize handle.
 *
 * The handle captures the pointer, so a fast drag that leaves the 4 px bar keeps
 * delivering events — no window listeners to leak, and no text selection to
 * clear up afterwards (pointerdown is cancelled). Deltas are incremental: the
 * caller adds them to whatever it is resizing, so one slow drag and one fast
 * drag move the same distance.
 */
export function useDragResize({ onDelta }: DragResize) {
  const last = useRef<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  const stop = useCallback((event?: React.PointerEvent<HTMLElement>) => {
    last.current = null;
    setDragging(false);
    if (!event) return;
    const target = event.currentTarget;
    try {
      if (target?.hasPointerCapture?.(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
      }
    } catch {
      // Nothing to release.
    }
  }, []);

  const handleProps = {
    onPointerDown: (event: React.PointerEvent<HTMLElement>) => {
      if (event.button !== 0) return; // left button / touch / pen only
      event.preventDefault(); // never start a text selection on the timeline
      try {
        event.currentTarget.setPointerCapture?.(event.pointerId);
      } catch {
        // A pointer that is already gone (or a synthetic one) cannot be captured;
        // the drag still works while the pointer stays over the handle.
      }
      last.current = { x: event.clientX, y: event.clientY };
      setDragging(true);
    },
    onPointerMove: (event: React.PointerEvent<HTMLElement>) => {
      const previous = last.current;
      if (!previous) return;
      const deltaX = event.clientX - previous.x;
      const deltaY = event.clientY - previous.y;
      last.current = { x: event.clientX, y: event.clientY };
      if (deltaX !== 0 || deltaY !== 0) onDelta(deltaX, deltaY);
    },
    onPointerUp: stop,
    onPointerCancel: stop,
    onLostPointerCapture: () => {
      last.current = null;
      setDragging(false);
    },
  };

  return { dragging, handleProps };
}
