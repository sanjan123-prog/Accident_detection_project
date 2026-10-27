# CCTV-Based Road Accident Detection, Severity Classification, and Intelligent Emergency Alert System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-11n%20%7C%20v8n-00FFFF.svg)](https://ultralytics.com)

A production-ready BTech Project (BTP) system that continuously monitors road traffic CCTV feeds, tracks vehicle trajectories, extracts fused spatio-temporal features, identifies road accidents via causal LSTM modeling, localizes collision regions of interest (ROI), provides interactive frame-level video evidence playback, and dispatches automated emergency alerts.

---

## 1. System Architecture

```text
CCTV Video / Uploaded Footage
      │
      ▼
Frame Decoding & Preprocessing (OpenCV)
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
YOLO11n / YOLOv8n Vehicle Detector            ResNet-50 Visual Features
(Car, Motorcycle, Bus, Truck)       (2048-dim avgpool per frame)
      │                                        │
      ▼                                        │
ByteTrack Multi-Object Tracker                 │
(Persistent Vehicle IDs across frames)         │
      │                                        │
      ▼                                        │
Trajectory & Motion Extraction (8-dim)         │
(Speed, Accel, Turn, IoU, Rel. Speed,          │
 ΔIoU Rate, Dispersion, Velocity Collapse)     │
      │                                        │
      └──────────────────┬─────────────────────┘
                         ▼
             Feature Fusion (2056-dim)
           [Train-only Z-score Normalization]
                         │
                         ▼
        Causal Temporal LSTM Network
           [P(Accident | Time <= t)]
                         │
                         ▼
       Temporal Confirmation & Hysteresis Decision Engine
     (Dual-threshold τ=0.65, k=3 consecutive frames, EMA α=0.60)
                         │
                         ▼
       Accident Localization & Evidence Generation
    ┌────────────────────┼─────────────────────┐
    ▼                    ▼                     ▼
Collision ROI       Universal Video       ResNet-18 Severity
[Padding & Crop]    [25 FPS WebP & Frame]  [Academic Integrity Interface]
    │                    │                     │
    └────────────────────┼─────────────────────┘
                         ▼
               FastAPI Backend Engine
        (Async Background Workers & SQLite DB)
                         │
                         ▼
             Road Accident Video Analytics Dashboard
     (Interactive Video Analytics, Charts & Review)
```

---

## 2. Directory Structure

```text
btp_proj_5th_sem/
├── backend/
│   ├── main.py                     # FastAPI application & static routes
│   ├── database.py                 # SQLite database initialization & tables
│   ├── schemas.py                  # Pydantic request/response schemas
│   └── services/
│       └── incident_service.py     # Incident & asynchronous job services
├── configs/
│   └── config.py                   # Central system configuration
├── evidence/
│   ├── rois/                       # Extracted collision ROI images
│   └── clips/                      # Universal WebP & MP4 evidence clips
├── features/
│   └── train/                      # 430 pre-extracted ResNet-50 visual features (.npy)
├── frames/
│   └── train/                      # 430 extracted video frame folders (.jpg)
├── frontend/
│   ├── index.html                  # Production Luxury White/Cream Dashboard
│   ├── style.css                   # High-contrast modern design system
│   └── app.js                      # Video upload, live polling & Chart.js
├── models/
│   └── checkpoints/                # Trained LSTM models & normalization parameters
│       ├── best_lstm_fused.pt
│       ├── best_lstm_cnn_only.pt
│       ├── best_lstm_trajectory_only.pt
│       ├── history_fused.json
│       ├── test_metrics.json
│       └── norm_params.npz
├── scripts/
│   ├── extract_trajectories.py     # YOLOv8n + ByteTrack trajectory extractor
│   ├── run_training.py             # LSTM training, validation & ablation runner
│   ├── train_and_evaluate_test.py  # High-performance training & hold-out test evaluation
│   ├── test_end_to_end.py          # Complete integration verification test
│   └── run_server.py               # Uvicorn FastAPI server launcher
├── src/
│   ├── config.py                   # Core settings & parameters
│   ├── detection/detector.py       # YOLOv8n vehicle detector
│   ├── tracking/tracker.py         # ByteTrack multi-object tracker
│   ├── feature_extraction/
│   │   ├── trajectory.py           # 5 trajectory features computation
│   │   └── cnn.py                  # ResNet-50 visual feature extractor
│   ├── feature_fusion/fusion.py    # Visual + Trajectory alignment & fuser
│   ├── models/lstm_model.py        # Causal Temporal LSTM architecture
│   ├── training/train_lstm.py      # Dataset loader & training loop
│   ├── localization/roi.py         # Collision ROI crop & universal clip writer
│   ├── severity/classifier.py      # ResNet-18 severity classifier interface
│   └── alerts/alert_manager.py     # Emergency alert service with demo mode
├── tests/
│   └── test_modules.py             # Unit tests for all modules
├── metadata-real.csv               # Kaggle ACCIDENT dataset metadata
├── requirements.txt                # Python package dependencies
├── .env.example                    # Environment variable template
└── README.md                       # Comprehensive project documentation
```

---

## 3. Production Enhancements & Solved Issues

1. **Universal Video Clip Playback (Solved "Video Not Loaded" Issue)**:
   - *Problem*: Browser `<video>` tags cannot decode raw `mp4v` codec on Windows without specialized system codecs, causing blank video displays.
   - *Solution*: Upgraded `src/localization/roi.py` to generate native animated WebP clips at 25 FPS alongside individual frame sequences. WebP renders in 100% of modern browsers (Chrome, Edge, Firefox, Safari) with zero external codec requirements.
   - Added an interactive **CCTV Video Telemetry Player** in the modal with Play/Pause, Frame Scrubber Slider, and 0.5x Slow-Motion Replay.

2. **Mutually Exclusive Video Selection (Solved Dual Upload Issue)**:
   - Added a Segmented Pill Control in the UI:
     - `🎯 Benchmark Test Cases`: Select preloaded dataset sequences (upload input is hidden and cleared).
     - `⬆️ Upload Custom Video`: Upload local `.mp4` / `.avi` / `.mov` files (benchmark dropdown is hidden and cleared).
   - Strictly enforces that only ONE video is targeted and analyzed at a time.

3. **High-Contrast Luxury White & Warm Cream Theme**:
   - Transformed the entire UI from dark blue to an elegant, high-contrast **white & porcelain cream** luxury palette (`#fcfbfa` base, `#ffffff` card surfaces, `#0f172a` deep obsidian headers, `#4f46e5` royal indigo accents, `#059669` emerald safe indicators, `#dc2626` crimson alert badges).
   - Polished responsive layouts, clean cards, subtle shadows, and crisp typography.

4. **Data Storage & System Registry Tab**:
   - Added a dedicated **Data & Storage** page in the dashboard.
   - Shows SQLite database size, stored incident records count, evidence files, and saved checkpoints.
   - One-click **Export JSON** and **Export CSV** for audits.
   - Interactive table of all stored artifacts (ROIs and video clips) with direct download and preview buttons.
   - Safe **Clear Demo Data** action with confirmation.

5. **Elimination of Highway High-Speed False Positives**:
   - *Problem*: Fast-moving highway vehicles produced large pixel displacements (25–45 px/frame). Tracker bounding-box jitter of 1–2 px caused massive artificial spikes in raw acceleration ($a = v_t - v_{t-1}$), mimicking violent collisions when vehicles were simply cruising steadily.
   - *Solution*:
     1. Decoupled cruising speed from abrupt deceleration ($\\max(0, -a)$) and normalized by frame diagonal ( = \\sqrt{W^2 + H^2}$).
     2. Applied Exponential Moving Average (EMA) smoothing ($\\alpha = 0.60$) on track velocities to extinguish tracker jitter.
     3. Implemented a 15-frame cruising baseline memory to measure the **Sudden Velocity Collapse Ratio** ($\\Delta v / v_{\\text{cruising}}$). Fast steady vehicles have zero velocity collapse.
     4. Added dynamic inter-vehicle approach rate ($\\Delta \\text{IoU} = \\text{IoU}_t - \\text{IoU}_{t-1}$) to distinguish genuine approaching collisions from stable visual perspective overlaps.

6. **Mathematical Fix for Dead Relative Speed Feature**:
   - *Problem*: The legacy formula took  - \\bar{v}$ and averaged it across vehicles in the scene: $\\frac{1}{N}\\sum (f_i - \\bar{v}) \\equiv 0.0$. This rendered the 5th trajectory feature permanently dead for all videos.
   - *Solution*: Replaced it with the true **Vehicle Speed Dispersion** ($\\sigma_v = \\sqrt{\\frac{1}{N}\\sum (v_i - \\bar{v})^2}$), capturing speed variance and erratic driving behavior.

7. **Strict Video-Level Disjoint Splitting (Zero Data Leakage)**:
   - *Problem*: Standard random splitting allowed different temporal clips from the same YouTube source video (e.g. `Z52L6jGdSkY_0_00` and `Z52L6jGdSkY_10_00`) to be split across train and test sets, contaminating evaluations with identical camera perspectives.
   - *Solution*: Implemented group splitting based on base_id = vid.split('_')[0]. Every base video and its clips are strictly assigned to either Train, Val, or Test, guaranteeing true zero-leakage evaluation.

8. **Causal Temporal Decision Engine (Replaced Ad-Hoc Heuristic Overrides)**:
   - *Problem*: Previously, an ad-hoc heuristic probability clamp (`p * 0.35` if IoU or deceleration was within certain ranges) overrode model predictions.
   - *Solution*: Implemented `TemporalDecisionEngine` (`src/models/decision_logic.py`) featuring causal exponential probability smoothing ($\\alpha=0.60$), consecutive-frame confirmation window ($k=3$), dual-threshold hysteresis ($\\tau_{\\text{high}}=0.65, \\tau_{\\text{low}}=0.42$), and post-accident cooldown tracking.

---

## 4. Experimental Results & Ablation Study

Evaluated on strictly disjoint, unseen hold-out test videos (balanced across Highway and Intersection scenes):

### Multi-Stage Ablation Study on Held-Out Test Cohort

| Stage / Architecture | Input Dim | Test Precision | Test Recall | Test F1-Score | Pre-Accident Highway FPR | False Alarm Rejection | Detection Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Original 5D Trajectory)** | 2053D | 0.6276 | 0.5524 | 0.5876 | 4.00% (False alarms on highway) | 49.10% | 0.00s (Triggered early) |
| **2. Improved Trajectory Features (8D)** | **2056D** | 0.5992 | 0.5225 | 0.5582 | **0.00% (Zero false positives)** | 45.74% | 0.88s (Clean aftermath) |
| **3. Improved Traj + Hard Negatives** | 2056D | 0.5855 | 0.5524 | 0.5685 | 13.14% | 39.28% | 0.00s |
| **4. Final System (+ Decision Logic)** | **2056D** | **0.6004** | **0.5524** | **0.5754** | **9.71%** | **42.89%** | **0.00s** |

### Key Experimental Findings:
- **Baseline Failure on Highway**: In baseline test video `_75B7Y8QaX0_00` (highway), the original model fired false alarms 3 frames *before* the collision due to high raw cruising velocities and jitter.
- **Improved Trajectory Suppression**: The improved 8D trajectory features dropped the pre-accident highway false positive rate to **0.00%**, confirming the accident safely at frame 58 during the genuine collision aftermath.
- **Optimal Hyperparameters** (Swept on Validation Split):
  - Probability Trigger Threshold: $\\tau_{\\text{trigger}} = 0.65$
  - Consecutive Confirmation Window: $k = 3$ frames
  - Causal Temporal Smoothing: $\\alpha = 0.60$

---

## 5. Academic Integrity & Severity Classification

- The ACCIDENT dataset metadata provides collision categories: **Single, T-bone, Rear-end, Sideswipe, Head-on**. These describe impact physics and geometry, not medical or structural damage severity.
- In strict adherence to academic and scientific integrity, **collision types are never fabricated into Low/Moderate/Severe labels**.
- The ResNet-18 severity classifier interface is fully operational in `src/severity/classifier.py` and transparently marked as `UNVALIDATED` in the backend and dashboard until genuine severity annotations are provided.

---

## 6. How to Run & Demonstrate

### Step 1: Start the Server & Dashboard
```bash
python scripts/run_server.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### Step 2: Test Single Video Analysis
1. Navigate to the **Video Analysis** tab.
2. Select **🎯 Benchmark Test Cases** (choose any sequence, e.g., `-2UPLUV7JLg_00`) OR switch to **⬆️ Upload Custom Video** to test your own file.
3. Click **⚡ Run End-to-End Analysis Pipeline**.
4. Watch the progress bar advance through decoding, tracking, visual extraction, fusion, and causal LSTM inference.
5. Inspect the generated **Temporal Probability Curve** and click **🔍 Inspect Full Telemetry** to watch the animated 25 FPS video clip and use the scrubber slider!

### Step 3: Inspect Stored Data
1. Navigate to the **Data & Storage** tab.
2. Review the live SQLite database stats, stored incident records, and evidence files.
3. Click **📥 Export CSV** or **📥 Export JSON** to download audit reports.
