# Real-Time Face Detection & Recognition System

A consent-based, fully local face detection and recognition app built with **Python, OpenCV (YuNet + SFace), SQLite and Streamlit**.

> **Verification status (please read).** In the environment where this code was written, the 57 unit tests in `tests/` were executed and pass (`python -m unittest`). They use fake detector/recognizer objects and synthetic images. **Not verified there:** the real YuNet/SFace models (the download was blocked), a real webcam, and the Streamlit UI (Streamlit could not be installed). Expect to do a first-run check on your own machine and use the troubleshooting section if something differs.

## Features

- Live webcam detection of multiple faces with bounding boxes, optional landmarks, FPS and face count
- Recognition by cosine similarity with a configurable threshold; "Unknown Person" for non-matches; ambiguous matches are never named
- IoU face tracking + majority-vote smoothing (no label flicker; embeddings computed every N frames, not every frame)
- Consent-gated enrollment with quality checks (size, blur, brightness, frontal pose, exactly one face), duplicate-name and duplicate-face prevention
- Only embeddings are stored, never photos. Admin-only user listing/deletion, retention policy, overwrite-then-delete

## Project structure

```
face_recognition_app/
├── app.py                # Streamlit UI (entry point)
├── config.py             # Settings from environment / .env, privacy notice text
├── database.py           # SQLite metadata + .npy embedding files, retention, secure delete
├── face_detector.py      # YuNet detector wrapper (Detection dataclass)
├── face_recognizer.py    # SFace embeddings + Matcher (cosine similarity)
├── registration.py       # Consented enrollment and validation
├── tracker.py            # IoU tracker + temporal smoothing
├── pipeline.py           # detect -> track -> embed/match -> draw
├── models.py, errors.py  # shared dataclasses and exceptions
├── utils.py              # image helpers, quality checks, camera, password hashing
├── download_models.py    # fetches the ONNX models
├── requirements.txt, .env.example, pytest.ini, .gitignore
├── models/               # ONNX files go here (created by download_models.py)
├── data/                 # created at runtime: faces.db + embeddings/
└── tests/                # unit tests
```

## Installation

Requirements: Python 3.11+, a webcam, internet access for the one-time model download.

**Windows (PowerShell)**
```powershell
cd face_recognition_app
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd face_recognition_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then:
```bash
python download_models.py        # downloads the two ONNX models into models/
cp .env.example .env             # (Windows: copy .env.example .env)
python utils.py                  # choose an admin password; paste the printed ADMIN_PASSWORD_HASH=... line into .env
streamlit run app.py
```
Open the URL Streamlit prints (normally http://localhost:8501).

For Streamlit Community Cloud, `.env` is not deployed. Add the generated hash in the app's
**Settings → Secrets** using TOML syntax:
```toml
ADMIN_PASSWORD_HASH = "pbkdf2_sha256$..."
REQUIRE_ADMIN_FOR_ENROLLMENT = true
```
Save the secrets and reboot the deployed app. Do not commit `.env` or publish the generated hash.

### Models, sources and licences

| Model | Source | Licence |
|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT |
| SFace `face_recognition_sface_2021dec.onnx` | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface | Apache-2.0 |

SFace is used instead of an InsightFace/ArcFace checkpoint because the widely used pretrained ArcFace weights have non-commercial terms and extra dependencies. If you swap models, check their licences.

## Using the app

**Register a user** (needs admin unlock by default): sidebar → enter admin password → *Register* tab → read the notice, type a display name (English letters/digits/spaces), tick the consent box → take 3–8 photos with *Capture with webcam* (or upload photos), varying head angle slightly. Rejected samples tell you why (no face, several faces, blurry, dark, too small, turned). Press **Enroll**. **Stop the live camera before capturing**, so the webcam is free.

**Recognize:** *Live recognition* tab → **Start camera**. Green box = recognised (name + similarity), red = Unknown Person, orange = still analysing / too small.

**Delete a user:** admin unlock → *Manage users* → select the user → tick the confirmation → *Delete user permanently*. The embedding file is overwritten with random bytes and removed, then the DB row is deleted. *Purge expired* applies the retention policy immediately (it also runs at session start).

## Running the tests

```bash
python -m unittest discover -s tests -t . -v     # no extra packages needed
pytest                                           # alternative, if pytest is installed
```
Covered: geometry/quality utilities, image decoding errors, password hashing, matching (match / unknown / ambiguous / threshold), database (add, duplicate, delete, retention, missing files, init failure), tracker + smoothing, registration rules (name, consent, sample counts, no/multiple faces, duplicates, mixed identities), and the pipeline.

## How it works

1. **Detection** – YuNet, a small CNN, outputs a box, 5 landmarks (eyes, nose, mouth corners) and a confidence for every face. Frames are downscaled to `DETECT_INPUT_WIDTH` for speed.
2. **Embedding** – SFace warps the face to a canonical pose using the landmarks and maps it to a 128-number vector. Photos of the same person land close together in this space.
3. **Matching** – both vectors are length-normalised, so cosine similarity is a dot product in [-1, 1]. A user's score is the best over their samples. A face is named only if the best score ≥ threshold (default 0.363, the value suggested for SFace) *and* no other user also passes within `AMBIGUITY_MARGIN`.
4. **Stabilising** – an IoU tracker follows each face; recognition runs every N frames and the label is a majority vote over the last few results.

**Scores are not probabilities.** A similarity of 0.6 does not mean "60 % sure"; the scale depends on the model, camera and conditions.

## Limitations and responsible use

- Accuracy varies with lighting, pose, resolution, masks/glasses and the individuals involved. **No accuracy or demographic-fairness claim is made for this project**; it has not been evaluated on a benchmark or on your users.
- Before relying on results, measure **false acceptance rate (FAR)** and **false rejection rate (FRR)** on consented data from your own camera setup and choose the threshold accordingly: raising it lowers FAR and raises FRR.
- Do not use it as the sole factor for security decisions; there is no liveness/anti-spoofing, so a photo of a person may be accepted.
- Use only with the knowledge and consent of the people involved. No covert or hidden-camera mode exists.
- Admin lockout counters are per browser session, not a hardened authentication system. Embeddings are stored unencrypted in `data/`; use full-disk encryption and restrict OS file permissions. Overwrite-before-delete cannot guarantee erasure on SSDs.
- Local-only: nothing is sent to external services (only the model download uses the internet). Logs never contain images, embeddings or names.

## Configuration

All options are documented in `.env.example` (thresholds, retention, camera, quality gates). Key ones: `RECOGNITION_THRESHOLD`, `RETENTION_DAYS` (0 = keep until deleted), `REQUIRE_ADMIN_FOR_ENROLLMENT`, `ADMIN_PASSWORD_HASH`. The threshold can also be adjusted live in the sidebar.

## Troubleshooting

| Problem | What to do |
|---|---|
| "Face detector model not found" | Run `python download_models.py`; or download the files manually (links above) into `models/`. |
| "Could not load the YuNet/SFace model" | Upgrade OpenCV (`pip install -U opencv-python`, needs ≥ 4.9); delete the `.onnx` file and download again. |
| "Could not open camera" / no frames | Close other apps using the webcam (Zoom, Teams, another tab); try `CAMERA_INDEX=1`; allow camera access for your terminal/Python (Windows: Settings → Privacy → Camera; macOS: System Settings → Privacy & Security → Camera). |
| Browser camera widget is black in *Register* | Press **Stop camera** on the live tab; allow camera permission for `localhost` in the browser. |
| "A required Python package is missing" | Activate the venv and run `pip install -r requirements.txt`. |
| Admin features disabled | Local app: set `ADMIN_PASSWORD_HASH` in the project `.env` (see `python utils.py`) and restart Streamlit. Deployed app: set `ADMIN_PASSWORD_HASH` under the app's Streamlit Secrets settings and reboot it; local `.env` files are not deployed. |
| Samples keep being rejected as blurry/dark | Improve lighting and focus, move closer, or lower `BLUR_THRESHOLD`/`MIN_FACE_SIZE` slightly. |
| Known people shown as Unknown | Re-enroll with more varied samples; lower the threshold a little (this raises false accepts). |
| Strangers matched to someone | Raise the threshold; evaluate FAR on test data. |
| Database errors | Make sure `data/` is writable; delete `data/faces.db` only if you accept losing all enrollments. |

## Future improvements

- Evaluation script computing FAR/FRR/ROC from a labelled, consented image set
- Liveness / anti-spoofing; encrypted embedding storage; per-user admin roles with real authentication
- Browser-side streaming (e.g. `streamlit-webrtc`) so the camera works when the app runs on a remote server
- GPU/ONNX Runtime backends and stronger models (ArcFace-class) where licences allow
- Audit-log viewer, CSV export of non-biometric metadata, internationalised names (Unicode font rendering)
