from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils import LANDMARK_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture classifier (KNN / RandomForest).")
    parser.add_argument("--dataset", default="data/gestures.csv", help="Path to CSV dataset")
    parser.add_argument("--model-out", default="models/gesture_model.joblib", help="Where to save model")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)

    expected_columns = LANDMARK_FEATURES + 1
    if df.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns (63 features + label), got {df.shape[1]}")

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1].str.upper()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=args.random_state),
    }

    best_name = ""
    best_model = None
    best_accuracy = -1.0

    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_name = name
            best_model = model
            best_accuracy = acc

    final_preds = best_model.predict(x_test)
    print(f"\nBest model: {best_name} ({best_accuracy:.4f})")
    print("Classification report:")
    print(classification_report(y_test, final_preds))

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)
    print(f"Saved model to: {model_out}")


if __name__ == "__main__":
    main()
