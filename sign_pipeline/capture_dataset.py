from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2 as cv

from utils import HandLandmarkExtractor, LANDMARK_FEATURES, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture labeled hand-landmark samples from webcam.")
    parser.add_argument("--label", required=True, help="Gesture label, e.g. HELP, THEFT, ACCIDENT, FIRE")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to capture")
    parser.add_argument("--output", default="data/gestures.csv", help="CSV file for dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = HandLandmarkExtractor()
    cap = cv.VideoCapture(0)

    header = [f"f_{i}" for i in range(LANDMARK_FEATURES)] + ["label"]
    write_header = not output_path.exists()

    captured = 0
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(header)

        while cap.isOpened() and captured < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            features, results = extractor.extract(frame)
            draw_landmarks(frame, results)
            cv.putText(
                frame,
                f"Label: {args.label.upper()}  Captured: {captured}/{args.samples}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.imshow("Dataset Capture", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and features is not None:
                writer.writerow(features + [args.label.upper()])
                captured += 1

    cap.release()
    extractor.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
