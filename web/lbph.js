// Local Binary Patterns Histograms (LBPH) face recognition, ported from OpenCV's
// cv::face::LBPHFaceRecognizer (the recognizer the desktop app uses) with its defaults:
// radius 1, 8 neighbours, 8x8 grid, normalised 256-bin cell histograms, chi-square distance.
// Same maths, so the desktop app's confidence thresholds (known < 50, unknown > 75) still apply.

export const FACE_SIZE = 100;
const RADIUS = 1;
const NEIGHBORS = 8;
const GRID = 8;
const BINS = 256;
const EPS = 1.1920929e-7; // float epsilon, as in OpenCV

/** Extended (circular) LBP with bilinear interpolation: gray w×h → codes (w-2)×(h-2). */
export function elbp(gray, w, h) {
  const ow = w - 2 * RADIUS;
  const oh = h - 2 * RADIUS;
  const out = new Uint8Array(ow * oh);
  for (let n = 0; n < NEIGHBORS; n++) {
    const x = RADIUS * Math.cos((2 * Math.PI * n) / NEIGHBORS);
    const y = -RADIUS * Math.sin((2 * Math.PI * n) / NEIGHBORS);
    const fx = Math.floor(x), fy = Math.floor(y), cx = Math.ceil(x), cy = Math.ceil(y);
    const tx = x - fx, ty = y - fy;
    const w1 = (1 - tx) * (1 - ty), w2 = tx * (1 - ty), w3 = (1 - tx) * ty, w4 = tx * ty;
    for (let i = RADIUS; i < h - RADIUS; i++) {
      for (let j = RADIUS; j < w - RADIUS; j++) {
        const c = gray[i * w + j];
        const t = w1 * gray[(i + fy) * w + j + fx] + w2 * gray[(i + fy) * w + j + cx]
                + w3 * gray[(i + cy) * w + j + fx] + w4 * gray[(i + cy) * w + j + cx];
        if (t > c || Math.abs(t - c) < EPS) out[(i - RADIUS) * ow + (j - RADIUS)] |= 1 << n;
      }
    }
  }
  return { codes: out, width: ow, height: oh };
}

/** Concatenated, per-cell normalised histograms over an 8×8 grid (length 64 × 256). */
export function spatialHistogram({ codes, width, height }) {
  const cw = Math.floor(width / GRID);
  const ch = Math.floor(height / GRID);
  const hist = new Float32Array(GRID * GRID * BINS);
  const cellSize = cw * ch;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const base = (gy * GRID + gx) * BINS;
      for (let y = gy * ch; y < (gy + 1) * ch; y++) {
        for (let x = gx * cw; x < (gx + 1) * cw; x++) hist[base + codes[y * width + x]] += 1;
      }
      for (let b = 0; b < BINS; b++) hist[base + b] /= cellSize;
    }
  }
  return hist;
}

export const describe = (gray, w = FACE_SIZE, h = FACE_SIZE) => spatialHistogram(elbp(gray, w, h));

/** OpenCV HISTCMP_CHISQR_ALT: 2 · Σ (a−b)² / (a+b). */
export function chiSquare(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const s = a[i] + b[i];
    if (s > 1e-12) {
      const d = a[i] - b[i];
      sum += (d * d) / s;
    }
  }
  return 2 * sum;
}

export class LBPHRecognizer {
  constructor() { this.samples = []; }            // [{ label, hist }]
  train(faces) {                                  // faces: [{ label, gray }]
    this.samples = faces.map((f) => ({ label: f.label, hist: describe(f.gray) }));
  }
  get size() { return this.samples.length; }
  /** Nearest neighbour: { label, confidence } where confidence is the chi-square distance (lower = closer). */
  predict(gray) {
    const q = describe(gray);
    let best = { label: null, confidence: Infinity };
    for (const s of this.samples) {
      const d = chiSquare(q, s.hist);
      if (d < best.confidence) best = { label: s.label, confidence: d };
    }
    return best;
  }
}
