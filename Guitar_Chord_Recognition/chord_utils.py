from torchvision import datasets, transforms, models
import torch
import numpy as np


def predict_image(test_transforms, device, model, image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.detach().clone() # this just gets a copy of the tensor
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(data_dir, test_transforms, num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    
    images, labels = next(iter(loader))
    return images, labels