import cv2
import mediapipe as mp
from torchvision import transforms
import torch
from collections import deque
from chord_utils import predict_image  # Import the function

# Class labels
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G','_BG']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing transformation
input_size = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
model = torch.load('chord_custom_mix250.pth')
model = model.to(device)
model.eval()

# Convert OpenCV image to PIL image
to_pil = transforms.ToPILImage()

# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('test_vid/custom_3_720p.mp4')
# vid = cv2.VideoCapture('chord_vids/g2_major.mp4')
if not vid.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Get video frame width and height
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
textposition = (10, height - 20)

# Smoothing parameters
prediction_buffer = deque(maxlen=1)  # Store the last 5 predictions
stable_prediction = None

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for i in range(10000):
        ret, frame = vid.read()
        
        if not ret:
            print("Error: Failed to capture a frame.")
            continue
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = hands.process(frame_rgb)
        
        # Draw hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Convert frame to PIL Image
        image = to_pil(frame)
        
        # Get the predicted index
        index = predict_image(input_size, device, model, image)
        
        # Print the prediction for diagnostics
        print(f"Prediction: {classes[index]}")
        
        # Add prediction to buffer
        prediction_buffer.append(index)
        
        # Use the most common prediction in the buffer as the stable prediction
        if len(prediction_buffer) == prediction_buffer.maxlen:
            stable_prediction = max(prediction_buffer, key=prediction_buffer.count)
        
        if stable_prediction is not None:
            print(classes[stable_prediction])  # Sanity check on the stable prediction
            
            # Load the corresponding chord legend image
            chord_legend = cv2.imread(f"chords/{classes[index]}.jpg", cv2.IMREAD_COLOR)
            if chord_legend is not None:
                legend_size = chord_legend.shape[0]
                W = frame.shape[1]
                frame[0:legend_size, W - legend_size:] = chord_legend

            # Put the prediction text on the frame
            cv2.putText(frame, f"Predicted chord: {classes[index]}", textposition, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame with hand landmarks
        cv2.imshow('Chord prediction', frame)

        # Break the loop if the user presses 'q'
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'):
            break

# Release the video capture and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
