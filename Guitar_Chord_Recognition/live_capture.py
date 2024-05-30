import cv2
import numpy as np
from torchvision import transforms
import torch
from collections import deque
from PIL import Image

# Function to predict the chord from the image
def predict_image(input_size, device, model, image):
    image_t = input_size(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_t)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

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
model = torch.load('chord_vision.pth')
model = model.to(device)

# Convert OpenCV image to PIL image
to_pil = transforms.ToPILImage()

# Initialize video capture device
vid = cv2.VideoCapture(-1)
if not vid.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Get video frame width and height
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
textposition = (10, height - 20)

# Smoothing parameters
prediction_buffer = deque(maxlen=30)  # Store last 30 predictions
stable_prediction = None

for i in range(10000):
    ret, frame = vid.read()
    
    if not ret:
        print("Error: Failed to capture a frame.")
        continue
    
    # Convert frame to PIL Image
    image = to_pil(frame)
    
    # Get the predicted index
    index = predict_image(input_size, device, model, image)
    
    # Add prediction to buffer
    prediction_buffer.append(index)
    
    # Use the most common prediction in the buffer as the stable prediction
    if len(prediction_buffer) == prediction_buffer.maxlen:
        stable_prediction = max(prediction_buffer, key=prediction_buffer.count)
    
    if stable_prediction is not None:
        print(classes[stable_prediction])  # Sanity check on the prediction
        
        # Load the corresponding chord legend image
        chord_legend = cv2.imread(f"chords/{classes[stable_prediction]}.jpg", cv2.IMREAD_COLOR)
        if chord_legend is not None:
            legend_size = chord_legend.shape[0]
            W = frame.shape[1]
            frame[0:legend_size, W - legend_size:] = chord_legend

        # Put the prediction text on the frame
        cv2.putText(frame, f"Predicted chord: {classes[stable_prediction]}", textposition, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow('Chord prediction', frame)

    # Break the loop if the user presses 'q'
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break

# Release the video capture and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
