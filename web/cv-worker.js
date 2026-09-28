// Face detection off the main thread: OpenCV.js is ~10 MB of wasm, and compiling it
// on the page would freeze the UI. The page sends frames and gets back 100×100 face crops.
/* global cv */
// 4.12: OpenCV 5 moved the Haar cascade (CascadeClassifier) out of the core build.
const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js';
const CASCADE_URL = 'haarcascade_frontalface_default.xml';
const FACE_SIZE = 100;

let detector = null;

const ready = (async () => {
  importScripts(OPENCV_URL);
  let c = self.cv;
  if (c instanceof Promise) c = await c;
  else if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
  self.cv = c;
  const xml = new Uint8Array(await (await fetch(CASCADE_URL)).arrayBuffer());
  cv.FS_createDataFile('/', 'cascade.xml', xml, true, false, false);
  detector = new cv.CascadeClassifier();
  if (!detector.load('cascade.xml')) throw new Error('cascade did not load');
  return cv.getBuildInformation().match(/Version control:\s*(\S+)/)?.[1] || '4.12';
})();

ready.then(
  (version) => postMessage({ type: 'ready', version }),
  (err) => postMessage({ type: 'failed', message: err.message }),
);

function detect({ width, height, data }, scale, neighbours, minSize = 60) {
  const src = cv.matFromImageData({ width, height, data });
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const found = new cv.RectVector();
  detector.detectMultiScale(gray, found, scale, neighbours, 0, new cv.Size(minSize, minSize));
  const faces = [];
  for (let i = 0; i < found.size(); i++) {
    const r = found.get(i);
    const roi = gray.roi(r);
    const small = new cv.Mat();
    cv.resize(roi, small, new cv.Size(FACE_SIZE, FACE_SIZE), 0, 0, cv.INTER_LINEAR);
    faces.push({ x: r.x, y: r.y, w: r.width, h: r.height, gray: new Uint8Array(small.data) });
    roi.delete(); small.delete();
  }
  src.delete(); gray.delete(); found.delete();
  return faces;
}

onmessage = async (e) => {
  const { id, frame, scale, neighbours, minSize } = e.data;
  try {
    await ready;
    const faces = detect(frame, scale, neighbours, minSize);
    postMessage({ type: 'faces', id, faces }, faces.map((f) => f.gray.buffer));
  } catch (err) {
    postMessage({ type: 'faces', id, faces: [], error: err.message });
  }
};
