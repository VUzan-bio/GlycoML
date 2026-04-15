export type ScoreScale = (kd: number) => number;

type RGB = { r: number; g: number; b: number };

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));

const lerp = (start: number, end: number, t: number) => start + (end - start) * t;

const mix = (from: RGB, to: RGB, t: number): RGB => ({
  r: Math.round(lerp(from.r, to.r, t)),
  g: Math.round(lerp(from.g, to.g, t)),
  b: Math.round(lerp(from.b, to.b, t)),
});

const BLUE_LOW: RGB = { r: 87, g: 113, b: 254 };
const BLUE_HIGH: RGB = { r: 96, g: 165, b: 250 };
const YELLOW_LOW: RGB = { r: 252, g: 211, b: 77 };
const YELLOW_HIGH: RGB = { r: 245, g: 158, b: 11 };
const RED_LOW: RGB = { r: 248, g: 113, b: 113 };
const RED_HIGH: RGB = { r: 239, g: 68, b: 68 };

export const createScoreScale = (values: number[]): ScoreScale => {
  if (values.length === 0) {
    return () => 0.5;
  }
  const logs = values.filter((v) => v > 0).map((v) => Math.log10(v));
  const min = Math.min(...logs);
  const max = Math.max(...logs);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return () => 0.5;
  }
  return (kd: number) => {
    if (kd <= 0 || !Number.isFinite(kd)) {
      return 0.5;
    }
    const normalized = (Math.log10(kd) - min) / (max - min);
    return clamp(1 - normalized);
  };
};

export const scoreToColor = (score: number): string => {
  const value = clamp(score);
  if (value <= 0.4) {
    const t = value / 0.4;
    const color = mix(BLUE_LOW, BLUE_HIGH, t);
    return `rgb(${color.r}, ${color.g}, ${color.b})`;
  }
  if (value <= 0.7) {
    const t = (value - 0.4) / 0.3;
    const color = mix(YELLOW_LOW, YELLOW_HIGH, t);
    return `rgb(${color.r}, ${color.g}, ${color.b})`;
  }
  const t = (value - 0.7) / 0.3;
  const color = mix(RED_LOW, RED_HIGH, t);
  return `rgb(${color.r}, ${color.g}, ${color.b})`;
};
