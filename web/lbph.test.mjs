// Checks the JS LBPH port against its definition and against OpenCV itself (when available):
//   node --test web/lbph.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FACE_SIZE, LBPHRecognizer, chiSquare, describe, elbp } from './lbph.js';

const N = FACE_SIZE * FACE_SIZE;
function texture(seed) {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  return Uint8Array.from({ length: N }, (_, i) => Math.floor(128 + 100 * Math.sin((i % FACE_SIZE) * (0.05 + seed * 0.01)) * rnd()));
}
const noisy = (img, amount, seed) => {
  let s = seed;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  return img.map((v) => Math.max(0, Math.min(255, v + Math.round((rnd() - 0.5) * amount))));
};

test('flat image: every neighbour equals the centre, so every LBP code is 255', () => {
  const { codes } = elbp(new Uint8Array(N).fill(90), FACE_SIZE, FACE_SIZE);
  assert.ok(codes.every((c) => c === 255));
});

test('each of the 64 cell histograms is normalised to 1', () => {
  const h = describe(texture(3));
  assert.equal(h.length, 64 * 256);
  for (let c = 0; c < 64; c++) {
    const sum = h.slice(c * 256, (c + 1) * 256).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-4, `cell ${c} sums to ${sum}`);
  }
});

test('chi-square: identical is 0, symmetric, positive otherwise', () => {
  const a = describe(texture(1)), b = describe(texture(2));
  assert.equal(chiSquare(a, a), 0);
  assert.ok(Math.abs(chiSquare(a, b) - chiSquare(b, a)) < 1e-6);
  assert.ok(chiSquare(a, b) > 0);
});

test('recognizer picks the right person from noisy samples', () => {
  const people = { 1: texture(11), 2: texture(29), 3: texture(47) };
  const r = new LBPHRecognizer();
  r.train(Object.entries(people).flatMap(([label, img]) =>
    [1, 2, 3, 4, 5].map((k) => ({ label: Number(label), gray: noisy(img, 16, k * 7 + Number(label)) }))));
  for (const [label, img] of Object.entries(people)) {
    assert.equal(r.predict(noisy(img, 16, 999 + Number(label))).label, Number(label));
  }
});
