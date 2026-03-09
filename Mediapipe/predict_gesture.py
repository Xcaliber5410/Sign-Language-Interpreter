import cv2
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load("gesture_model.pkl")

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

            for hand_landmarks in result.hand_landmarks:

                for lm in hand_landmarks:

                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    cv2.circle(frame,(cx,cy),5,(0,255,0),-1)

                    landmark_list.append((lm.x*1000)//10)
                    landmark_list.append((lm.y*1000)//10)
                    landmark_list.append((lm.z*1000)//10)

                # Predict gesture
                if len(landmark_list) == 126:

                    prediction = model.predict([landmark_list])
                    
                    cv2.putText(
                        frame,
                        prediction[0],
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