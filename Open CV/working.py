import cv2

# 1. Access the webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # 2. Capture frames frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # 3. Draw text on the screen
    # Parameters: image, text, bottom-left corner coordinates, font, font scale, color (BGR), thickness
    cv2.putText(frame, 'Live Feed Active!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # 4. Show real-time video
    cv2.imshow('My Webcam Feed', frame)

    # Wait for 1 millisecond, and check if the 'q' key is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
