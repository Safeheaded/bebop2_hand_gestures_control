import cv2
import mediapipe as mp
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class HandLandmarksModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HandLandmarksModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def transform_and_normalize_landmarks(data):
    # Convert data to numpy array for easier manipulation
    data_array = np.array(data)

    # Extract the coordinates
    coordinates = data_array[:, 1:].astype(float)

    # Get the first coordinate to use as the new origin
    origin = coordinates[0]

    # Transform the coordinates
    transformed_coordinates = coordinates - origin

    # Normalize the coordinates
    min_vals = transformed_coordinates.min(axis=0)
    max_vals = transformed_coordinates.max(axis=0)
    normalized_coordinates = (transformed_coordinates - min_vals) / (max_vals - min_vals)

    # Update the data array with normalized coordinates
    data_array[:, 1:] = normalized_coordinates

    return data_array.tolist()


finger_mapping = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}

input_dim = 57  # Adjust this based on your input dimensions
num_classes = 2  # Adjust this based on the number of classes
model = HandLandmarksModel(input_dim, num_classes)

# Load the state dictionary
model.load_state_dict(torch.load('hand_landmarks_model_2024-10-26.pth'))

# Set the model to evaluation mode
model.eval()

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
    'ring': [13, 14, 15, 16],
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
            data = []
            for hand_landmarks in results.multi_hand_landmarks:
                for finger, indices in finger_indices.items():
                    for i in indices:
                        landmark = hand_landmarks.landmark[i]
                        data.append([finger, landmark.x, landmark.y])
                        # print('adding data')

            # print(data)
            data = [[finger_mapping[finger], x, y] for finger, x, y in data]

            normalized = transform_and_normalize_landmarks(data)
            if len(normalized) == 19:
                x = torch.tensor(normalized, dtype=torch.float32).view(1, -1)
                y = model(x)
                print(y)
                y = torch.argmax(y, dim=1)
                print(y)
                y = y.item()
                print(y)
                cv2.putText(frame, str(y), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)




        # Display the frame with hand landmarks
        cv2.imshow('Hand Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()