# Dependencies
from PIL import Image
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

#Pytorch
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

class VGG():
    def __init__(self):
        # get the "features" portion of VGG19 (we will not need the "classifier" portion)
        self.vgg = models.vgg19(pretrained=True).features

        # freeze all VGG parameters since we're only optimizing the target image
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # move the model to GPU, if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg.to(self.device)

        # Style weights
        self.style_weights = {'conv1_1': 1.,
                            'conv2_1': 0.8,
                            'conv3_1': 0.5,
                            'conv4_1': 0.3,
                            'conv5_1': 0.1}

        # Weights
        self.content_weight = 1  # alpha
        self.style_weight = 1e6  # beta
