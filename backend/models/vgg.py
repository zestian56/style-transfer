# Dependencies
import json
import os
from io import BytesIO
import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Pytorch
import torch
import torch.optim as optim
from torchvision import transforms, models

class VGG():
    def __init__(self):
        # get the "features" portion of VGG19 - remove classifier portion
        self.vgg = models.vgg19(pretrained=True).features

        # freeze all VGG parameters since we will only optimize target image
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
        # Convert to uint8
        image = np.array(image)
        image = (image.astype(np.float64) / image.max()) * 255
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        original_width, original_height = image.size

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
        
        return image, original_width, original_height

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

    def get_features(self, image, model, layers=None):
        """ Run an image forward through a model and get the features for 
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """
        
        ## Complete mapping layer names of PyTorch's VGGNet to names from the paper
        ## Need the layers for the content and style representations of an image
        if layers is None:
            layers = {'0': 'conv1_1', 
                    '5': 'conv2_1', 
                    '10': 'conv3_1', 
                    '19': 'conv4_1', 
                    '21': 'conv4_2',  ## content representation
                    '28': 'conv5_1'}
            
            
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        # This is the forward pass of the network, pass the image through each
        # layer of the CNN
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features

    def gram_matrix(self, tensor):
        """ Calculate the Gram Matrix of a given tensor 
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """
        
        ## get the batch_size, depth, height, and width of the Tensor
        batch_size, d, h, w = tensor.size()
        ## reshape it, so we're multiplying the features for each channel
        tensor = tensor.view(d, h * w)
        ## calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        
        return gram 