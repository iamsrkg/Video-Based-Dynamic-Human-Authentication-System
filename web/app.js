// Face Gate: the train.py flow (take images → train → track) in the browser.
// OpenCV.js does the Haar-cascade detection; lbph.js is a port of OpenCV's LBPH recognizer.
import { FACE_SIZE, LBPHRecognizer } from './lbph.js';

const SAMPLES = 60;                        // train.py stops after 60 crops

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const worker = new Worker('cv-worker.js');   // OpenCV lives here, off the main thread
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
const people = new Map();                  // id -> { name, faces: [Uint8Array] }
const recognizer = new LBPHRecognizer();
const record = new Map();                  // id -> { id, name, date, time, distance }
const unknown = [];
let stream = null, tracking = false, busy = false;

// ---------------------------------------------------------------- setup
function status(text, state = 'ok') {
  $('status').innerHTML = `<span class="led ${state}"></span> ${esc(text)}`;
}

const pendingDetect = new Map();
let nextId = 0;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'ready') { status(`ready · OpenCV ${m.version} in a worker`); document.body.classList.add('ready'); }
  else if (m.type === 'failed') status(`OpenCV failed to load: ${m.message}`, 'err');
  else if (m.type === 'faces') { pendingDetect.get(m.id)?.(m.faces); pendingDetect.delete(m.id); }
};
worker.onerror = (e) => status(`OpenCV worker error: ${e.message}`, 'err');

// ---------------------------------------------------------------- vision
/** Detect faces in an RGBA ImageData (in the worker). Resolves to [{ x, y, w, h, gray: Uint8Array(100*100) }]. */
function detectFaces(frame, scale, neighbours, minSize = 60) {
  return new Promise((resolve) => {
    const id = ++nextId;
    pendingDetect.set(id, resolve);
    worker.postMessage({ id, frame: { width: frame.width, height: frame.height, data: frame.data }, scale, neighbours, minSize }, [frame.data.buffer]);
  });
}

const grab = document.createElement('canvas');
const gctx = grab.getContext('2d', { willReadFrequently: true });
function frameFrom(source, w, h) {
  const max = 640, k = Math.min(1, max / Math.max(w, h));
  grab.width = Math.round(w * k); grab.height = Math.round(h * k);
  gctx.drawImage(source, 0, 0, grab.width, grab.height);
  return gctx.getImageData(0, 0, grab.width, grab.height);
}

async function imageFromFile(file) {
  const bmp = await createImageBitmap(file);
  const data = frameFrom(bmp, bmp.width, bmp.height);
  bmp.close();
  return data;
}

function faceToDataUrl(gray) {
  const c = document.createElement('canvas');
  c.width = c.height = FACE_SIZE;
  const img = c.getContext('2d').createImageData(FACE_SIZE, FACE_SIZE);
  for (let i = 0; i < gray.length; i++) { img.data.set([gray[i], gray[i], gray[i], 255], i * 4); }
  c.getContext('2d').putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

function drawBoxes(boxes, frameW) {
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  const k = video.videoWidth / frameW;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.lineWidth = 3; octx.font = '600 20px system-ui, sans-serif';
  for (const b of boxes) {
    // The video is shown mirrored (like a mirror), so boxes are flipped here and the text stays readable.
    const x = overlay.width - (b.x + b.w) * k;
    octx.strokeStyle = b.color; octx.fillStyle = b.color;
    octx.strokeRect(x, b.y * k, b.w * k, b.h * k);
    if (b.label) octx.fillText(b.label, x, (b.y + b.h) * k + 24);
  }
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  if (stream) return true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (err) {
    $('placeholder').querySelector('p').textContent = `Camera unavailable (${err.name}). Use the photo buttons instead.`;
    return false;
  }
  video.srcObject = stream;
  await video.play();
  $('placeholder').hidden = true;
  $('cam-state').hidden = false;
  stream.getVideoTracks()[0].addEventListener('ended', stopCamera);   // e.g. permission revoked
  return true;
}

function stopCamera() {
  tracking = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $('placeholder').hidden = false;
  $('cam-state').hidden = true;
}

// ---------------------------------------------------------------- 1 · take images
function readPerson() {
  const id = $('pid').value.trim(), name = $('pname').value.trim();
  // Same validation as train.py: numeric ID, alphabetic name.
  if (!/^\d+$/.test(id)) { $('enroll-msg').textContent = 'Enter a numeric ID'; return null; }
  if (!/^\p{L}+$/u.test(name)) { $('enroll-msg').textContent = 'Enter an alphabetical name'; return null; }
  return { id: Number(id), name };
}

function addFaces(person, faces) {
  const p = people.get(person.id) || { name: person.name, faces: [] };
  p.name = person.name;
  p.faces.push(...faces);
  people.set(person.id, p);
  renderPeople();
  $('train').disabled = false;
  $('train-msg').textContent = 'New images since the last training.';
}

function renderPeople() {
  $('people').innerHTML = [...people].map(([id, p]) => `
    <li><img src="${faceToDataUrl(p.faces[0])}" alt=""><span><b>${esc(id)} · ${esc(p.name)}</b><br>${p.faces.length} images</span></li>`).join('');
}

$('enroll').addEventListener('click', async () => {
  const person = readPerson();
  if (!person || busy || !(await startCamera())) return;
  busy = true; tracking = false;
  const faces = [];
  $('enroll-msg').textContent = 'Look at the camera and move your head a little…';
  const deadline = Date.now() + 40000;                       // don't wait forever for a face
  while (faces.length < SAMPLES && stream && Date.now() < deadline) {
    const frame = frameFrom(video, video.videoWidth, video.videoHeight);
    const found = await detectFaces(frame, 1.3, 5);               // train.py: detectMultiScale(gray, 1.3, 5)
    drawBoxes(found.map((f) => ({ ...f, color: '#5cc8ff' })), grab.width);
    if (found.length === 1) faces.push(found[0].gray);
    $('enroll-bar').style.width = `${(faces.length / SAMPLES) * 100}%`;
    await sleep(100);                                        // train.py: waitKey(100)
  }
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!faces.length) {
    $('enroll-msg').textContent = 'No face seen. Face the camera in good light and try again.';
  } else {
    addFaces(person, faces);
    $('enroll-msg').textContent = faces.length < SAMPLES
      ? `Saved ${faces.length} images for ID ${person.id}, name ${person.name} (stopped after 40 s)`
      : `Images saved for ID ${person.id}, name ${person.name}`;
  }
  busy = false;
});

$('enroll-files').addEventListener('change', async (e) => {
  const person = readPerson();
  if (!person) { e.target.value = ''; return; }
  const faces = [];
  for (const file of e.target.files) {
    const found = await detectFaces(await imageFromFile(file), 1.1, 5, 30);
    if (found.length) faces.push(found.sort((a, b) => b.w - a.w)[0].gray);   // the biggest face in each photo
  }
  e.target.value = '';
  if (!faces.length) { $('enroll-msg').textContent = 'No face found in those photos.'; return; }
  addFaces(person, faces);
  $('enroll-bar').style.width = '100%';
  $('enroll-msg').textContent = `Images saved for ID ${person.id}, name ${person.name} (${faces.length} from photos)`;
});

// ---------------------------------------------------------------- 2 · train
$('train').addEventListener('click', () => {
  const t = performance.now();
  recognizer.train([...people].flatMap(([label, p]) => p.faces.map((gray) => ({ label, gray }))));
  $('train-msg').textContent = `Image trained: ${recognizer.size} images of ${people.size} ${people.size === 1 ? 'person' : 'people'} in ${Math.round(performance.now() - t)} ms`;
  $('track').disabled = false; $('track-file').disabled = false;
});

// ---------------------------------------------------------------- 3 · track
const thresholds = () => ({ known: Number($('known').value), unknown: Number($('unknown').value) });
['known', 'unknown'].forEach((k) => $(k).addEventListener('input', () => { $(`${k}-v`).textContent = $(k).value; }));

/** The decision from TrackImages(): log a match once per person, save the unknowns. */
function judge(face) {
  const { label, confidence } = recognizer.predict(face.gray);
  const d = Math.round(confidence * 10) / 10;
  const t = thresholds();
  if (d < t.known) {
    const name = people.get(label)?.name || 'Unregistered';
    if (!record.has(label)) {
      const now = new Date();
      record.set(label, { id: label, name, date: now.toISOString().slice(0, 10), time: now.toTimeString().slice(0, 8), distance: d });
      renderRecord();
    }
    return { ...face, color: '#3ddc97', label: `${label}-${name} (${d})` };
  }
  if (d > t.unknown && unknown.length < 24) { unknown.push(faceToDataUrl(face.gray)); renderUnknown(); }
  return { ...face, color: '#ff7eb6', label: `Unknown (${d})` };
}

$('track').addEventListener('click', async () => {
  if (tracking) { tracking = false; return; }
  if (busy || !(await startCamera())) return;
  tracking = true;
  $('track').textContent = 'Stop tracking';
  while (tracking && stream) {
    const frame = frameFrom(video, video.videoWidth, video.videoHeight);
    const boxes = (await detectFaces(frame, 1.2, 5)).map(judge);        // train.py: detectMultiScale(gray, 1.2, 5)
    drawBoxes(boxes, grab.width);
    $('track-msg').textContent = boxes.length ? boxes.map((b) => b.label).join(' · ') : 'No face in view';
    await sleep(120);
  }
  $('track').textContent = 'Start tracking';
});

$('track-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  const boxes = (await detectFaces(await imageFromFile(file), 1.1, 5, 30)).map(judge);
  $('track-msg').textContent = boxes.length ? boxes.map((b) => b.label).join(' · ') : 'No face found in that photo.';
});

// ---------------------------------------------------------------- record
function renderRecord() {
  $('record').innerHTML = [...record.values()].map((r) =>
    `<tr><td>${r.id}</td><td>${esc(r.name)}</td><td>${r.date}</td><td>${r.time}</td><td>${r.distance}</td></tr>`).join('');
  $('csv').disabled = false;
}
function renderUnknown() {
  $('unknown-faces').innerHTML = unknown.map((src) => `<img src="${src}" alt="unknown face">`).join('');
}

$('csv').addEventListener('click', () => {
  // Same columns as train.py's record_<date>_<time>.csv
  const csv = 'Id,Name,Date,Time\n' + [...record.values()].map((r) => [r.id, r.name, r.date, r.time].join(',')).join('\n');
  const now = new Date();
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })),
    download: `record_${now.toISOString().slice(0, 10)}_${now.toTimeString().slice(0, 8).replace(/:/g, '-')}.csv`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

$('cam').addEventListener('click', startCamera);
$('cam-off').addEventListener('click', stopCamera);
status('loading OpenCV in a worker…', 'wait');
