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

    def load_image(self, img_path, max_size=400, shape=None):
        ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''

        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        # large images will slow down processing
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        
        if shape is not None:
            size = shape
            
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225))])
        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)
        
        return image

    # helper function for un-normalizing an image 
    # and converting it from a Tensor image to a NumPy image for display
    def im_convert(self, tensor):
        """ Display a tensor as an image. """
        
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image
