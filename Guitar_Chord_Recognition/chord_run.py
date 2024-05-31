from matplotlib import pyplot as plt
import torch
from torchvision import transforms

from chord_utils import predict_image, get_random_images


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
model=torch.load('chord_vision_pure_vid.pth')
model.eval()


chords = ['A', 'B', 'C', 'D', 'E', 'F', 'G','_BG']
data_dir = 'output'

to_pil = transforms.ToPILImage()
images, labels = get_random_images(data_dir, input_size, 15)

fig=plt.figure(figsize=(15,15))

for i in range(len(images)):
    image = to_pil(images[i])
    index = predict_image(input_size, device, model, image)
    sub = fig.add_subplot(1, len(images), i+1)
    res = int(labels[i]) == index
    sub.set_title(str(chords[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)

plt.show()