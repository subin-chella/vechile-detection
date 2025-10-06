# Vehicle Detection & License Plate Recognition

Streamlit application for end-to-end vehicle detection, tracking, and license plate recognition powered by YOLOv8, DeepSORT, and EasyOCR. The pipeline processes uploaded videos, annotates detections frame-by-frame, and renders an annotated MP4 for download.

## ✨ Highlights

* Streamlit-first experience—launch one app and manage everything from the browser.
* YOLOv8 + DeepSORT for real-time vehicle tracking with robust plate association.
* EasyOCR with custom normalization tuned for Indian license formats.
* Shared-cache, in-memory SQLite storage replaces CSV artifacts for consistent state handling.
* Clean overlays that show at most one plate per vehicle and automatically clear when vehicles leave the scene.
* Outputs (videos, logs, CSV exports for debugging) collected under `outputs/` for easy retrieval.

## 🧱 Key Modules

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, upload handling, and pipeline orchestration. |
| `main.py` | Simple launcher that delegates to `streamlit run app.py`. |
| `processing.py` | Runs YOLOv8 detection, DeepSORT tracking, and stores raw detections in SQLite. |
| `interpolation.py` | Interpolates missing detections and writes enhanced tracks back into SQLite. |
| `visualization.py` | Builds overlays and produces the annotated MP4 using DB-backed data. |
| `database.py` | Creates the in-memory shared SQLite schema and exposes CRUD helpers. |
| `sort/` | DeepSORT tracker implementation. |
| `models/` | YOLOv8 weights (`yolov8n.pt`) and the license plate detector (`license_plate_detector.pt`). |
| `config.py` | Runtime configuration (confidence thresholds, retention windows, etc.). |
| `requirements.txt` | Python dependencies. |

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* `pip`
* GPU is optional; CPU mode works but is slower for large videos.

### Installation

```bash
git clone https://github.com/subin-chella/vechile_detection.git
cd vechile_detection
pip install -r requirements.txt
```

Ensure the YOLO weights exist in `models/`. If you need to re-download them, consult the links documented in `TECHNICAL_DOCUMENTATION.md`.

## ▶️ Usage

You can start the app via either command:

```bash
python main.py
# or
streamlit run app.py
```

Then:

1. Open the Streamlit URL shown in the terminal (defaults to `http://localhost:8501`).
2. Upload an MP4 video.
3. Press **Process Video**.
4. Monitor progress indicators for detection, interpolation, and visualization.
5. Download the annotated video once processing finishes. All artifacts (output video, CSV snapshots for debugging, logs) live in `outputs/`.

### Data Flow Overview

1. **Processing** – `processing.py` runs YOLOv8 detections, DeepSORT tracking, and records each frame’s detections into the shared SQLite database.
2. **Interpolation** – `interpolation.py` fills in missed detections to keep overlays smooth and consistent.
3. **Visualization** – `visualization.py` reads detections/interpolations from SQLite to draw plate overlays and produce the final MP4.

The database exists in memory (shared-cache mode) for the lifetime of the Streamlit session. Each video upload receives a unique UUID, keeping results isolated even during concurrent sessions.

## 📦 Outputs

All generated files live under `outputs/`:

* `<video-id>_output.mp4` – annotated video ready for download.
* `results.csv`, `results_interpolated.csv` – diagnostic exports retained for debugging.
* `app.log` – Streamlit-side logging, useful when debugging OCR or detections.

Clean up the folder whenever you want to reclaim disk space.

## 🛠 Troubleshooting

* **Streamlit not found** – ensure `pip install -r requirements.txt` ran successfully and the `streamlit` executable is on your PATH.
* **CUDA errors** – either install appropriate GPU drivers or force CPU mode by editing `config.py` to disable GPU usage.
* **OCR misreads** – adjust normalization rules in `util.py` or fine-tune EasyOCR thresholds in `config.py`.
* **Low detection confidence** – tweak `VEHICLE_CONFIDENCE_THRESHOLD` or `PLATE_CONFIDENCE_THRESHOLD` in `config.py`.

## 🧰 Technologies

* Streamlit
* YOLOv8 (Ultralytics)
* DeepSORT
* EasyOCR
* OpenCV
* SQLite (in-memory, shared-cache)
* NumPy / Pandas

For deeper architectural notes, refer to `TECHNICAL_DOCUMENTATION.md`.
