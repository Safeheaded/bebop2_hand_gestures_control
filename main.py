import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

# Define landmark indices for thumb and index finger
finger_indices = {
    'thumb': [2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

# Open CSV file for writing
with open('hand_landmarks.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['klasa', 'finger', 'landmark_index', 'x', 'y', 'z'])

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for each finger
                for finger, indices in finger_indices.items():
                    for i in indices:
                        landmark = hand_landmarks.landmark[i]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the frame with hand landmarks
        cv2.imshow('Hand Recognition', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save landmarks to CSV
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for finger, indices in finger_indices.items():
                        for i in indices:
                            landmark = hand_landmarks.landmark[i]
                            writer.writerow(['rock_on', finger, i, landmark.x, landmark.y, landmark.z])

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()