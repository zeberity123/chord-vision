import cv2
import mediapipe as mp
from pathlib import Path

# Function to mark hand landmarks on a frame
def mark_hand_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the video file
video_path = Path("test_vid/custom_3_720p.mp4")
vid = cv2.VideoCapture(str(video_path))

# Prepare the folder to save marked frames
output_folder = Path("test_vid/custom_3")
output_folder.mkdir(parents=True, exist_ok=True)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    frame_count = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Determine when to capture a frame (every 30 frames)
        if frame_count % 2 == 0:
            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand detection
            results = hands.process(frame_rgb)

            # Draw hand annotations on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mark_hand_landmarks(frame, hand_landmarks)

            # Save the marked frame as an image
            marked_frame_path = output_folder / f"frame_{frame_count}.png"
            cv2.imwrite(str(marked_frame_path), frame)

        frame_count += 1

# Release the video capture
vid.release()
cv2.destroyAllWindows()
