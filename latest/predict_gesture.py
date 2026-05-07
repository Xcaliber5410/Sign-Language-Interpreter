import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
from google import genai
import textwrap

# Configure Gemini API client - REPLACE WITH A NEW API KEY
client = genai.Client(api_key="") 

# Load model
model = joblib.load("gesture_model.pkl")

# MediaPipe Initialization
mp_hands = mp.tasks.vision.HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

CONFIDENCE_THRESHOLD = 0.35
prediction_buffer = deque(maxlen=15)
stable_prediction = ""
sentence_words = []
last_added_word = ""
generated_sentence = ""
no_hand_counter = 0  

# --- FIR Sequence Wizard & Form Memory ---
fir_sequence = [
    "Step 1: Your Name? (Spell it)",
    "Step 2: What Happened? (Theft, Attack, etc.)",
    "Step 3: When did it happen? (in HH:MM)",
    "Step 4: Where did it happen?",
    "Step 5: Stolen items or injuries?",
    "Step 6: Describe the accused.",
    "Step 7: Any evidence/witnesses?",
    "Step 8: Any other information?"
]
current_step = 0
fir_data = {i: [] for i in range(len(fir_sequence))}
# ----------------------------------------

def calculate_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def extract_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0]
    shifted_coords = coords - wrist
    max_dist = np.max(np.linalg.norm(shifted_coords, axis=1))
    
    normalized_coords = shifted_coords / max_dist if max_dist > 0 else shifted_coords
    flattened_coords = normalized_coords.flatten().tolist()

    angles = []
    joint_groups = [
        (1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8), 
        (9, 10, 11), (10, 11, 12), (13, 14, 15), 
        (14, 15, 16), (17, 18, 19), (18, 19, 20)
    ]
    for a, b, c in joint_groups:
        v1 = coords[a] - coords[b]
        v2 = coords[c] - coords[b]
        angles.append(calculate_angle(v1, v2))

    return flattened_coords + angles

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2
)

cap = cv2.VideoCapture(0)

with mp_hands.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)
        right_features = [0.0] * 73
        left_features = [0.0] * 73

        if results.hand_landmarks:
            no_hand_counter = 0  
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                handedness = results.handedness[idx][0].category_name
                features = extract_features(hand_landmarks)
                
                if handedness == 'Right': right_features = features
                elif handedness == 'Left': left_features = features

            combined_features = right_features + left_features
            probabilities = model.predict_proba([combined_features])[0]
            max_confidence = max(probabilities)

            if max_confidence > CONFIDENCE_THRESHOLD:
                prediction = model.predict([combined_features])[0]
                prediction_buffer.append(prediction)

                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                if prediction_buffer.count(most_common) >= 12:
                    stable_prediction = most_common
                else:
                    stable_prediction = ""
            else:
                prediction_buffer.append("")
                stable_prediction = ""
        else:
            prediction_buffer.clear()
            stable_prediction = ""
            no_hand_counter += 1
            if no_hand_counter > 10:
                last_added_word = ""

        # --- AUTO APPEND TO SENTENCE ---
        if stable_prediction and stable_prediction != last_added_word:
            if len(stable_prediction) == 1:
                if sentence_words and len(sentence_words[-1]) <= 10 and len(last_added_word) == 1:
                    sentence_words[-1] += stable_prediction
                else:
                    sentence_words.append(stable_prediction)
            else:
                sentence_words.append(stable_prediction)
            
            last_added_word = stable_prediction  

        # --- DISPLAY UI ---
        cv2.putText(frame, fir_sequence[current_step], (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Pred: {stable_prediction}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        display_words = " ".join(sentence_words[-6:]) 
        cv2.putText(frame, f"Input: {display_words}", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if generated_sentence:
            wrapped_text = textwrap.wrap(f"FINAL FIR: {generated_sentence}", width=45)
            y_offset = 140
            for line in wrapped_text:
                cv2.putText(frame, line, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25 

        # Display Help Controls at the bottom 
        h, w, _ = frame.shape
        cv2.putText(frame, "[ENTER] Gen FIR | [n] Next | [b] Back | [d]/[BACKSPACE] Delete", (5, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("FIR Sign Language System", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # --- KEYBOARD CONTROLS (FORM NAVIGATION) ---
        if key == ord('n'):  # Next Step
            fir_data[current_step] = list(sentence_words)
            if current_step < len(fir_sequence) - 1:
                current_step += 1
            sentence_words = list(fir_data[current_step])
            generated_sentence = ""

        elif key == ord('b'):  # Back Step
            fir_data[current_step] = list(sentence_words)
            if current_step > 0:
                current_step -= 1
            sentence_words = list(fir_data[current_step])
            generated_sentence = ""

        elif key == ord('c'):  # Clear Current Step entirely
            sentence_words = []
            last_added_word = ""
            generated_sentence = ""
            prediction_buffer.clear()
            
        # --- BACKSPACE LOGIC ---
        elif key == 8 or key == 127 or key == ord('d'):  # Backspace or 'd'
            if sentence_words:
                if len(last_added_word) == 1 and len(sentence_words[-1]) > 1:
                    sentence_words[-1] = sentence_words[-1][:-1]  
                else:
                    sentence_words.pop()  
                
                last_added_word = "DELETED_LOCK" 
                prediction_buffer.clear() 
                stable_prediction = ""

        elif key == 13:  # Enter to COMPLILE AND GENERATE FINAL FIR
            fir_data[current_step] = list(sentence_words)
            
            compiled_inputs = ""
            for i, question in enumerate(fir_sequence):
                if not fir_data[i]:
                    signs = "N/A"
                elif i == 2:  # Step 3: Time Parsing Logic
                    # Join list into one string, e.g. ["1", "4", "3", "0"] -> "1430"
                    raw_time = "".join(fir_data[i]).replace(" ", "").replace(":", "")
                    try:
                        # Extract hours and minutes safely
                        if len(raw_time) >= 2:
                            hh = int(raw_time[:2])
                            mm = raw_time[2:] if len(raw_time) > 2 else "00"
                        else:
                            hh = int(raw_time)
                            mm = "00"
                        
                        # Apply AM/PM logic
                        period = "AM" if hh < 12 else "PM"
                        
                        # Convert to 12-hour format for the FIR text
                        display_hh = hh if hh <= 12 else hh - 12
                        display_hh = 12 if display_hh == 0 else display_hh
                        
                        signs = f"{display_hh}:{mm.ljust(2, '0')} {period}"
                    except ValueError:
                        # Fallback if they signed words like "MORNING" instead of numbers
                        signs = " ".join(fir_data[i])
                else:
                    signs = " ".join(fir_data[i])
                
                compiled_inputs += f"{question}: {signs}\n"

            print("\n--- Sending Data to Gemini ---")
            print(compiled_inputs)
            
            try:
                prompt = f"""
                You are an expert Sign Language Translator drafting a First Information Report (FIR) narrative for an Indian Police Station.
                A deaf complainant has provided the following signs across multiple questions. 
                "N/A" means the question was skipped.

                {compiled_inputs}
                
                Rules:
                1. Single letters (e.g., K E N I L) must be combined into names.
                2. Write ONE cohesive, formal, and legally appropriate paragraph summarizing the entire incident based on the inputs provided.
                3. Do not invent details that were not signed. Ignore "N/A" entries.
                4. Output ONLY the final formal paragraph. No greetings or bullet points.
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                generated_sentence = response.text.strip()
                print(f"\nGenerated FIR Paragraph:\n{generated_sentence}\n")
                
            except Exception as e:
                print(f"Error generating sentence: {e}")
                generated_sentence = "Error: Check terminal."

        elif key == 27:  # Esc to exit
            break

cap.release()
cv2.destroyAllWindows()