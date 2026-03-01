from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List

import cv2 as cv
import mediapipe as mp
import numpy as np

LANDMARK_COUNT = 21
LANDMARK_FEATURES = LANDMARK_COUNT * 3


@dataclass
class PredictionSmoother:
    """Majority-vote smoother for noisy frame-level predictions."""

    window_size: int = 5

    def __post_init__(self) -> None:
        self._history: Deque[str] = deque(maxlen=self.window_size)

    def add(self, label: str) -> str:
        self._history.append(label)
        return Counter(self._history).most_common(1)[0][0]


class HandLandmarkExtractor:
    """Extracts flattened 63-feature hand landmark arrays using MediaPipe."""

    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.7) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.6,
        )

    def close(self) -> None:
        self._hands.close()

    def extract(self, frame_bgr: np.ndarray) -> tuple[List[float] | None, object]:
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, results

        landmarks = results.multi_hand_landmarks[0].landmark
        features: List[float] = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) != LANDMARK_FEATURES:
            return None, results

        return features, results


FIR_TEMPLATES: Dict[str, str] = {
    "THEFT": "Incident Type: Theft\nSummary: Suspected theft reported via sign gesture.\nAction: Verify location and register complaint.",
    "HELP": "Incident Type: Emergency Assistance\nSummary: User signaled urgent help.\nAction: Dispatch nearest responder and contact control room.",
    "ACCIDENT": "Incident Type: Accident\nSummary: Potential accident indicated by user gesture.\nAction: Send medical/police unit and secure scene.",
    "FIRE": "Incident Type: Fire\nSummary: Fire emergency indicated through gesture.\nAction: Alert fire services and trigger evacuation protocol.",
}


def draw_landmarks(frame_bgr: np.ndarray, results: object) -> None:
    if not getattr(results, "multi_hand_landmarks", None):
        return

    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS
    for hand_landmarks in results.multi_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            hand_connections,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )


def labels_upper(labels: Iterable[str]) -> List[str]:
    return [label.strip().upper() for label in labels]
