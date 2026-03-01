# Sign Language Interpreter (ML + FIR Pipeline)

This repository now includes a complete scikit-learn based workflow for:

1. **Dataset capture** (63 MediaPipe landmark features + label)
2. **Model training** (KNN and RandomForest comparison)
3. **Model persistence** (`joblib` save/load)
4. **Live webcam prediction**
5. **Prediction smoothing** (majority vote over last 5 predictions)
6. **Gesture → FIR template mapping**

## Project Structure

- `sign_pipeline/capture_dataset.py` — capture hand-landmark samples for a given label.
- `sign_pipeline/train_model.py` — train and evaluate KNN + RandomForest, save best model.
- `sign_pipeline/live_fir.py` — live prediction with stable output + FIR text rendering.
- `sign_pipeline/utils.py` — shared MediaPipe extraction, smoothing, drawing, FIR templates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Capture Dataset

Capture at least ~50 samples for each gesture (e.g., `HELP`, `THEFT`, `ACCIDENT`, `FIRE`):

```bash
python sign_pipeline/capture_dataset.py --label HELP --samples 50
python sign_pipeline/capture_dataset.py --label THEFT --samples 50
python sign_pipeline/capture_dataset.py --label ACCIDENT --samples 50
python sign_pipeline/capture_dataset.py --label FIRE --samples 50
```

Controls:
- Press `c` to capture one sample when hand is detected.
- Press `q` to quit.

## 2) Train Model

```bash
python sign_pipeline/train_model.py --dataset data/gestures.csv --model-out models/gesture_model.joblib
```

This script:
- validates dataset shape (`63 features + 1 label`)
- compares KNN vs RandomForest
- prints accuracy + classification report
- saves the best model

## 3) Run Live Prediction + FIR Mapping

```bash
python sign_pipeline/live_fir.py --model models/gesture_model.joblib --window-size 5
```

Pipeline:

`Webcam -> MediaPipe landmarks (63) -> scikit-learn model -> smoothed prediction -> FIR template`

## FIR Template Mapping

Current mapping lives in `sign_pipeline/utils.py`:
- `THEFT` → Theft template
- `HELP` → Emergency Assistance template
- `ACCIDENT` → Accident template
- `FIRE` → Fire template

Extend `FIR_TEMPLATES` to add new gestures.
