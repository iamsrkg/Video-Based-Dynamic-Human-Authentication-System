# Video-Based Dynamic Human Authentication

A desktop app that recognizes people passing a gate from a live webcam feed and **logs who entered and when**. Unknown faces are saved for review.

**▶ Try it live in your browser: https://iamsrkg.github.io/Video-Based-Dynamic-Human-Authentication-System/** (camera or photos; nothing leaves your device)

Published as a research paper in the *International Journal of Advanced Science and Technology* (IJAST, SERSC).

## How it works

1. **Enroll** (*Take Images*): enter a numeric ID and a name. The app captures about 60 face crops from the webcam using a **Haar cascade** detector (`haarcascade_frontalface_default.xml`) and saves them as grayscale images.
2. **Train** (*Train Images*): trains an **LBPH** (Local Binary Pattern Histograms) face recognizer (OpenCV `cv2.face`) on the enrolled faces and saves the model.
3. **Recognize** (*Track Images*): for each detected face in the live feed, the model predicts an ID and a confidence distance:
   - distance **< 50**: a known person. Their ID, name, date and time go into the session record.
   - distance **> 75**: the face crop is saved to `ImagesUnknown/` for review.
   - Pressing `q` ends the session and writes the record to `record/record_<date>_<time>.csv`.

**Why LBPH?** It's fast on a CPU, works with only a few dozen samples per person, and handles lighting changes better than Eigenfaces or Fisherfaces. That suits a low-cost gate camera.

## Run it

Needs Python 3.9+ and a webcam.

```bash
pip install -r requirements.txt
python train.py
```

The app creates its data folders on first run (`TrainingImage/`, `TrainingImageLabel/`, `visitor_detail/`, `ImagesUnknown/`, `record/`). They're git-ignored because they hold **biometric data** of real people.

### Tested
The enrollment → training → recognition pipeline is smoke-tested headlessly, with synthetic face images, against Python 3.12 and OpenCV 5.0 (contrib). The live capture and tracking loops need a physical webcam.

## Web version (`web/`)
The same take images → train → track flow, running entirely in the browser:
- **Detection:** OpenCV.js 4.12 with the same Haar cascade and the same `detectMultiScale` parameters as `train.py`. It runs in a **Web Worker**, so compiling the ~10 MB wasm never freezes the page. (OpenCV 5 moved `CascadeClassifier` out of the core build, which is why it's pinned to 4.12.)
- **Recognition:** `web/lbph.js`, a JavaScript port of OpenCV's LBPH: the extended LBP operator, an 8×8 grid of normalized histograms, and chi-square nearest neighbour. It's unit-tested in `web/lbph.test.mjs`.
- **Same decisions:** distance < 50 is logged once per person, and > 75 is saved as unknown. Both thresholds are adjustable, and the visitor record downloads as the same `record_<date>_<time>.csv`.
- **Privacy:** frames, faces and the model stay in the tab, and a reload wipes them.

```bash
node --test web/lbph.test.mjs          # tests
# serve web/ plus the cascade file with any static server, e.g.
cp haarcascade_frontalface_default.xml web/ && npx serve web
```
GitHub Actions runs the tests and deploys to GitHub Pages on every push.

## Stack
Python · OpenCV (Haar cascade + LBPH) · NumPy · pandas · Pillow · Tkinter

`setup.py` packages the app as a Windows executable with cx_Freeze (`python setup.py build`).
