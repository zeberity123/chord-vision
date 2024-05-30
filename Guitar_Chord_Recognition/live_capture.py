import cv2
from torchvision import transforms
import torch
from collections import deque
from chord_utils import predict_image, get_random_images

# def predict_image(input_size, device, model, image):
#     image_t = input_size(image).unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(image_t)
#     _, predicted = torch.max(outputs, 1)
#     return predicted.item()

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G','_BG']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = torch.load('chord_vision.pth')
model = model.to(device)

to_pil = transforms.ToPILImage()

vid = cv2.VideoCapture(-1)
if not vid.isOpened():
    print("Error: Could not open video capture device.")
    exit()

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
textposition = (10, height - 20)

prediction_buffer = deque(maxlen=30)  # Store last 30 predictions
stable_prediction = None

# for i in range(10000):
#     ret, frame = vid.read()
    
#     if not ret:
#         print("Error: Failed to capture a frame.")
#         continue
    
#     image = to_pil(frame)
    
#     index = predict_image(input_size, device, model, image)
    
#     prediction_buffer.append(index)
    
#     if len(prediction_buffer) == prediction_buffer.maxlen:
#         stable_prediction = max(prediction_buffer, key=prediction_buffer.count)
    
#     if stable_prediction is not None:
#         print(classes[stable_prediction])
        
#         chord_legend = cv2.imread(f"chords/{classes[stable_prediction]}.jpg", cv2.IMREAD_COLOR)
#         if chord_legend is not None:
#             legend_size = chord_legend.shape[0]
#             W = frame.shape[1]
#             frame[0:legend_size, W - legend_size:] = chord_legend

#         cv2.putText(frame, f"Predicted chord: {classes[stable_prediction]}", textposition, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
#     cv2.imshow('Chord prediction', frame)

#     keypress = cv2.waitKey(1) & 0xFF
#     if keypress == ord('q'):
#         break


for i in range(10000):
    ret, frame = vid.read()
    
    if not ret:
        print("Error: Failed to capture a frame.")
        continue
    
    data_dir = 'output'

    images, labels = get_random_images(data_dir, input_size, 1)
    image = to_pil(images)
    
    index = predict_image(input_size, device, model, image)
    
    prediction_buffer.append(index)
    
    if len(prediction_buffer) == prediction_buffer.maxlen:
        stable_prediction = max(prediction_buffer, key=prediction_buffer.count)
    
    if stable_prediction is not None:
        print(classes[stable_prediction])
        
        chord_legend = cv2.imread(f"chords/{classes[stable_prediction]}.jpg", cv2.IMREAD_COLOR)
        if chord_legend is not None:
            legend_size = chord_legend.shape[0]
            W = image.shape[1]
            image[0:legend_size, W - legend_size:] = chord_legend

        cv2.putText(image, f"Predicted chord: {classes[stable_prediction]}", textposition, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Chord prediction', image)

    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
