import cv2
import mediapipe as mp
import joblib
import math
from collections import deque

# --- SHARED FEATURE EXTRACTION ---
# (This must be identical to the one in the capture script)
def extract_features(hand_landmarks):
    features = []
    wrist = hand_landmarks[0]
    
    max_dist = 1e-6 
    for lm in hand_landmarks:
        dx, dy, dz = lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        max_dist = max(max_dist, dist)
        
    for lm in hand_landmarks:
        rel_x = (lm.x - wrist.x) / max_dist
        rel_y = (lm.y - wrist.y) / max_dist
        rel_z = (lm.z - wrist.z) / max_dist
        dist = math.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        angle = math.atan2(rel_y, rel_x)
        features.extend([rel_x, rel_y, rel_z, dist, angle])
        
    return features
# ---------------------------------

try:
    model = joblib.load("gesture_model.pkl")
except FileNotFoundError:
    print("Error: gesture_model.pkl not found. Please train your model first.")
    exit()

CONFIDENCE_THRESHOLD = 0.35
prediction_buffer = deque(maxlen=12)
stable_prediction = ""

mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        feature_vector = []

        if result.hand_landmarks:
            hands = sorted(result.hand_landmarks, key=lambda h: h[0].x)

            for hand_idx in range(2):
                if hand_idx < len(hands):
                    hand_landmarks = hands[hand_idx]
                    features = extract_features(hand_landmarks)
                    feature_vector.extend(features)

                    for lm in hand_landmarks:
                        h, w, _ = frame.shape   
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
                else:
                    feature_vector.extend([0.0] * 105)

            # Predict only if we successfully built the 210-feature vector
            if len(feature_vector) == 210:
                probabilities = model.predict_proba([feature_vector])[0]
                max_confidence = max(probabilities)

                if max_confidence > CONFIDENCE_THRESHOLD:
                    prediction = model.predict([feature_vector])[0]
                    prediction_buffer.append(prediction)

                    if prediction_buffer.count(prediction) > 8:
                        stable_prediction = prediction
                else:
                    prediction_buffer.clear()
                    stable_prediction = ""
        else:
            prediction_buffer.clear()
            stable_prediction = ""

        cv2.putText(
            frame,
            stable_prediction,
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()