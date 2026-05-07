import cv2
import mediapipe as mp
import csv
import math

# --- SHARED FEATURE EXTRACTION ---
def extract_features(hand_landmarks):
    features = []
    wrist = hand_landmarks[0]
    
    # Find the maximum distance from the wrist to normalize scale
    max_dist = 1e-6 # Avoid division by zero
    for lm in hand_landmarks:
        dx, dy, dz = lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        max_dist = max(max_dist, dist)
        
    for lm in hand_landmarks:
        # 1. Relative Coordinates (Normalized by hand size)
        rel_x = (lm.x - wrist.x) / max_dist
        rel_y = (lm.y - wrist.y) / max_dist
        rel_z = (lm.z - wrist.z) / max_dist
        
        # 2. Polar Coordinates (Distance and Angle in XY plane)
        dist = math.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
        angle = math.atan2(rel_y, rel_x)
        
        features.extend([rel_x, rel_y, rel_z, dist, angle])
        
    return features # Returns 105 features (21 landmarks * 5)
# ---------------------------------

mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

label = input("Enter gesture label: ")
cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if result.hand_landmarks:
            # Sort hands by x coordinate (left to right) to ensure consistent order
            hands = sorted(result.hand_landmarks, key=lambda h: h[0].x)
            row_data = []

            # Force iteration over exactly 2 hand slots
            for hand_idx in range(2):
                if hand_idx < len(hands):
                    hand_landmarks = hands[hand_idx]
                    
                    # Extract 105 features for this hand
                    features = extract_features(hand_landmarks)
                    row_data.extend(features)

                    # Draw landmarks for visual feedback
                    for lm in hand_landmarks:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
                else:
                    # Pad missing hands with 105 zeros
                    row_data.extend([0.0] * 105)

            # Write the unified 210-feature row + label
            if len(row_data) == 210:
                with open("attack_fixed.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data + [label])

        cv2.imshow("Data Capture - Press ESC to exit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()