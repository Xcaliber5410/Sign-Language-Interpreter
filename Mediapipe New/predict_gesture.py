import cv2
import mediapipe as mp
import joblib
import time
from collections import deque

# Load trained model
model = joblib.load("gesture_model.pkl")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.35

# Prediction smoothing buffer
prediction_buffer = deque(maxlen=12)
stable_prediction = ""

# Initialize MediaPipe
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        landmark_list = []

        if result.hand_landmarks:

            # Sort hands by x coordinate (left to right)
            hands = sorted(result.hand_landmarks, key=lambda h: h[0].x)

            # Force iteration over exactly 2 hand slots to match training data
            for hand_idx in range(2):

                if hand_idx < len(hands):
                    hand_landmarks = hands[hand_idx]
                    
                    # Get the wrist coordinates to act as the origin (0,0,0)
                    wrist = hand_landmarks[0]
                    wrist_x = wrist.x
                    wrist_y = wrist.y
                    wrist_z = wrist.z

                    for lm in hand_landmarks:
                        h, w, _ = frame.shape   
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

                        # TRULY Relative coordinates (Current Landmark - Wrist Landmark)
                        rel_x = lm.x - wrist_x
                        rel_y = lm.y - wrist_y
                        rel_z = lm.z - wrist_z

                        landmark_list.append((rel_x * 1000) // 10)
                        landmark_list.append((rel_y * 1000) // 10)
                        landmark_list.append((rel_z * 1000) // 10)
                else:
                    # Pad missing hands with zeros (matches the dataset builder)
                    landmark_list.extend([0] * 63)

            # Because of the padding above, the list will always be 126 
            # if at least one hand is detected.
            if len(landmark_list) == 126:

                probabilities = model.predict_proba([landmark_list])[0]
                max_confidence = max(probabilities)

                if max_confidence > CONFIDENCE_THRESHOLD:

                    prediction = model.predict([landmark_list])[0]
                    prediction_buffer.append(prediction)

                    # Check if prediction is stable
                    if prediction_buffer.count(prediction) > 8:
                        stable_prediction = prediction

                else:
                    prediction_buffer.clear()
                    stable_prediction = ""
                    
        else:
            # Clear text if no hands are visible at all
            prediction_buffer.clear()
            stable_prediction = ""

        display_text = stable_prediction

        cv2.putText(
            frame,
            display_text,
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