import { useEffect, useState } from 'react';
import type React from 'react';

export interface FitSize {
  width: number;
  height: number;
}

/**
 * Largest box of `ratio` (width / height) that fits inside the element.
 *
 * A video with `height: 100%; width: auto` grows past the card's width as soon
 * as the card is narrower than the video's shape, and then `object-contain`
 * letterboxes it — the black bars this is meant to remove. Measuring instead of
 * guessing gives a frame that is exactly the video's shape at every size: it
 * follows the card height, never overflows, and centres the leftover space on
 * the card background rather than inside a black rectangle.
 */
export function useFitSize(
  ref: React.RefObject<HTMLElement | null>,
  ratio: number,
): FitSize | null {
  const [size, setSize] = useState<FitSize | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element || !Number.isFinite(ratio) || ratio <= 0) return;

    const measure = () => {
      const boxWidth = element.clientWidth;
      const boxHeight = element.clientHeight;
      if (boxWidth <= 0 || boxHeight <= 0) return;
      const width = Math.min(boxWidth, boxHeight * ratio);
      const height = width / ratio;
      // Floor, so rounding can never push the frame one pixel past the box.
      setSize({ width: Math.max(1, Math.floor(width)), height: Math.max(1, Math.floor(height)) });
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, ratio]);

  return size;
}
