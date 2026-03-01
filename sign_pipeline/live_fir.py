from __future__ import annotations

import argparse

import cv2 as cv
import joblib
import numpy as np

from utils import FIR_TEMPLATES, HandLandmarkExtractor, PredictionSmoother, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live sign prediction with FIR mapping.")
    parser.add_argument("--model", default="models/gesture_model.joblib", help="Path to trained model")
    parser.add_argument("--window-size", type=int, default=5, help="Prediction smoothing window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)

    extractor = HandLandmarkExtractor()
    smoother = PredictionSmoother(window_size=args.window_size)

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features, results = extractor.extract(frame)
        draw_landmarks(frame, results)

        stable_prediction = "NO_HAND"
        if features is not None:
            pred = model.predict(np.array(features).reshape(1, -1))[0]
            stable_prediction = smoother.add(str(pred).upper())

        fir_text = FIR_TEMPLATES.get(
            stable_prediction,
            "Incident Type: Unknown\nSummary: No mapped FIR template for current prediction.",
        )

        cv.putText(frame, f"Prediction: {stable_prediction}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y = 65
        for line in fir_text.split("\n"):
            cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 25
        cv.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow("Sign to FIR Live", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    extractor.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
