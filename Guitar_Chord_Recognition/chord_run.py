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

def which_chord():
    # Display input controls 
    print("Which chord will you capture data for?")
    chord_option = widgets.Dropdown(
    options=["C","D", "E","F","G","A","B", "_BG"],
    value = "A",
    description="Chord"
    )

    clean_current_chord_data = widgets.Checkbox(
        value=False,
        description="Clean existing training data? ",
        disabled=False,
        indent=False
    )

    display(chord_option)
    display(clean_current_chord_data)


which_chord()