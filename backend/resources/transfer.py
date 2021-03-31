# Dependencies
from PIL import Image
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from flask_socketio import emit

#Pytorch
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
from flask import request


from flask_restful import Resource

from models import VGG

model = VGG()

path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static'))

class Transfer():
    def startProcess(self, payload):
        
        emit('updateProcess', { 'progress': 1, 'state': "Loading content image" });
        # Obtain information
        content_file = payload['content']
        style_file = payload['style']
        show_every = payload['show_every']
        steps = payload['steps']
        # load in content and style image
        content, content_w, content_h = model.load_image(content_file)
        content = content.to(model.device)

        # Update content image
        plt.imshow(model.im_convert(content))
        os.remove(path + '/content.jpg')
        plt.savefig(path +  '/content.jpg')
        
        emit('updateProcess', { 'progress': 5, 'state': "Loading style image"  });
        # Resize style to match content, makes code easier
        style, _ , _ = model.load_image(style_file, shape=content.shape[-2:])
        style = style.to(model.device)

        # Update style image
        plt.imshow(model.im_convert(style))
        os.remove(path + '/style.jpg')
        plt.savefig(path + '/style.jpg')
        emit('updateProcess', { 'progress': 10, 'state': "Featuring initiliazation"});

        # get content and style features only once before forming the target image
        content_features = model.get_features(content, model.vgg)
        style_features = model.get_features(style, model.vgg)

        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: model.gram_matrix(style_features[layer]) for layer in style_features}

        # create a third "target" image and prep it for change
        # it is a good idea to start off with the target as a copy of our *content* image
        # then iteratively change its style
        target = content.clone().requires_grad_(True).to(model.device)

        # for displaying the target image, intermittently
        #show_every = 100 #TODO THIS HAS TO BE A VARIABLE

        # iteration hyperparameters
        # Here we specify that the only thing that will be trained is the target image
        optimizer = optim.Adam([target], lr=0.003)
        #steps = 1000  # decide how many iterations to update your image (5000) TODO VARIABLE

        for ii in range(1, steps+1):
            progress = (ii*90 / (steps+1) ) + 10
            iterations = "Iterating {0}/{1}".format(ii, steps)
            emit('updateProcess', { 'progress': progress, 'state': iterations})
            ## get the features from your target image    
            ## Then calculate the content loss
            target_features = model.get_features(target, model.vgg)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
            
            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            # iterate through each style layer and add to the style loss
            for layer in model.style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                _, d, h, w = target_feature.shape
                
                ## Calculate the target gram matrix
                target_gram = model.gram_matrix(target_features[layer])
                ## get the "style" style representation
                style_gram = style_grams[layer]
                ## Calculate the style loss for one layer, weighted appropriately
                layer_style_loss = model.style_weights[layer] * torch.mean((target_gram - style_gram)**2) 
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            ## calculate the *total* loss
            total_loss = model.content_weight * content_loss + model.style_weight * style_loss
            
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # display intermediate images and print the loss
            if  ii % show_every == 0:
                converted_image = model.im_convert(target)
                PIL_image = Image.fromarray(np.uint8(converted_image*255)).convert('RGB')
                img = PIL_image.resize((content_w, content_h), Image.NEAREST)
                os.remove(path + '/target.jpg')
                img.save(path + '/target.jpg')
                with open(path + '/target.jpg', 'rb') as f:
                    image_data = f.read()
                    emit('updateProcess', { 'progress': progress, 'state': iterations, 'img': image_data })

        
        emit('updateProcess', { 'progress': 100, 'state': "Finish :D"})
        emit('endProcess')
        return total_loss.item()

    def get():
	    return "Hello :)"
