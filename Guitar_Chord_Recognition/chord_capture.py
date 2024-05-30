import cv2
from matplotlib import pyplot as plt
import time
import os, re, os.path
from ipywidgets import interact, widgets
from IPython.display import clear_output
import numpy as np
import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def chord_capture(chord):
    output_path = "output/" + chord + "/"

    i = 0 

    vid = cv2.VideoCapture(-1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else: 
        i = len(os.listdir(output_path))

    chord_legend = cv2.imread("chords/%s.jpg" % chord, cv2.IMREAD_COLOR)

    camera_widget = widgets.Image(
        format='jpg',
        width=300,
        height=400,
    )

    legend_size = 80
    W = 640
    center_column = W // 2 - legend_size // 2

    run_data_capture = False 

    photo_count = 250
    if run_data_capture: 
        for image in range(photo_count):
            clear_output(wait=True)
            i += 1
            file_name = output_path + "{:03d}.jpg".format(i)

            ret, frame = vid.read()
   
            H,W,C = frame.shape

            cv2.imwrite(file_name, frame)

            frame[0:legend_size, W - legend_size - center_column:W - legend_size - center_column + legend_size] = chord_legend

            plt.title('Camera feed')
            plt.imshow(frame)
            plt.show()

            time.sleep(0.12)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows() 

