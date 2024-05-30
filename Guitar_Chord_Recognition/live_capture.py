import cv2
import numpy as np
from chord_utils import predict_image
from PIL import Image
import torch
from torchvision import transforms


# Ensure the to_pil function and predict_image function are defined
# def to_pil(image):
#     return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G','_BG']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
model=torch.load('chord_vision.pth')
to_pil = transforms.ToPILImage()

vid = cv2.VideoCapture(-1)

if not vid.isOpened():
    print("Error: Could not open video capture device.")
    exit()

width  = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
textposition = (10, int(height) - 20)

predicted_image = np.zeros((1, 168, 224, 3))

for i in range(10000):
    ret, frame = vid.read()
    
    if not ret:
        print("Error: Failed to capture a frame.")
        continue
    
    # dim = (224, 168)
    # resized = cv2.resize(frame, dim)
    
    # predicted_image[0,:,:,:] = resized
    image = to_pil(frame)
    
    # y_pred = predict_image(image)
    index = predict_image(input_size, device, model, image)
    # print(classes[y_pred]) # sanity check on the prediction
    
    chord_legend = cv2.imread("chords/%s.jpg" % classes[index], cv2.IMREAD_COLOR)
    legend_size = chord_legend.shape[0]
    W = frame.shape[1]
    center_column = 0  # Adjust based on your layout
    frame[0:legend_size, W - legend_size - center_column:W - center_column] = chord_legend
    
    cv2.putText(frame, "Predicted chord: %s " % classes[index], textposition, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Chord prediction', frame)

    keypress = cv2.waitKey(1) & 0xFF
    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
