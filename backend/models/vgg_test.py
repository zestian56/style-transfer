import unittest
from collections import OrderedDict

import mock
from mock import patch
import torch
import numpy as np

from .vgg import VGG

class VGGTest(unittest.TestCase):

    @patch("torchvision.models.vgg19")
    def test_vgg_init_attributes(self, model_vgg19_mock):
        # TODO model type
        # Act
        style_transfer = VGG()

        # Assert
        self.assertIsInstance(style_transfer.device, torch.device)
        # como evaluar esta basura de modelo
        # Assert model type
        #self.assertItemsEqual(style_transfer.vgg, torch.nn.modules.container.Sequential)
        # Assert last layer conv of model
        #self.assertIsInstance(style_transfer.vgg[36], torch.nn.modules.pooling.MaxPool2d)
        self.assertEqual(style_transfer.style_weights, {'conv1_1': 1., 
                                                        'conv2_1': 0.8,
                                                        'conv3_1': 0.5,
                                                        'conv4_1': 0.3,
                                                        'conv5_1': 0.1})
        self.assertEqual(style_transfer.content_weight, 1)
        self.assertEqual(style_transfer.style_weight, 1e6)
        
    @patch("torchvision.models.vgg19")
    def test_load_image(self, model_vgg19_mock):
        # TODO MOCK image
        # Arrange
        style_transfer = VGG()
        # Como mockear esta fcking imagen
        img_path = "https://i.imgur.com/qeeo195.jpg"
        
        # Act
        img = style_transfer.load_image(img_path)
        
        # Assert
        self.assertIsInstance(img, torch.Tensor)
        expected_shape = torch.Size([1, 3, 655, 400])
        self.assertEqual(img.shape, expected_shape)
        expected_channels = 3
        self.assertEqual(img.shape[1], expected_channels)
        expected_dim = 400
        self.assertTrue(expected_dim in img.shape[2:])
        batch_size = 1
        self.assertEqual(img.shape[0], batch_size)
        
        ### Evaluating a particular max_size
        
        # Act
        expected_dim = 600
        img = style_transfer.load_image(img_path, max_size = expected_dim)
        
        # Assert
        self.assertTrue(expected_dim in img.shape[2:])
        expected_shape = torch.Size([1, 3, 983, 600])
        self.assertEqual(img.shape, expected_shape)
        
        ### Evaluating shape != None
        
        # Act
        expected_shape = [400, 500]
        img = style_transfer.load_image(img_path, shape = expected_shape)
        
        # Assert
        self.assertTrue(img.shape[2:] == torch.Size(expected_shape))
        expected_shape = torch.Size([1, 3, expected_shape[0], expected_shape[1]])
        self.assertEqual(img.shape, expected_shape)
        
        
        
    @patch("torchvision.models.vgg19")
    def test_im_convert(self, model_vgg19_mock):
        ### DONE
        # Arrange
        style_transfer = VGG()
        test_tensor = torch.rand(1, 3, 640, 480)
        
        # Act
        im_converted = style_transfer.im_convert(test_tensor)
        
        # Assert
        self.assertIsInstance(im_converted, np.ndarray)
        expected_max = 1
        self.assertLessEqual(im_converted.max(), expected_max)
        expected_min = 0
        self.assertGreaterEqual(im_converted.min(), expected_min)
        expected_shape = (test_tensor.shape[2], test_tensor.shape[3], test_tensor.shape[1])
        self.assertEqual(im_converted.shape, expected_shape)
        
    @patch("torchvision.models.vgg19")
    def test_get_features(self, model_vgg19_mock):
        ### TODO ASSERT MORE THINGS, I do not know how...
        # Arrange
        style_transfer = VGG()
        img_path = "https://i.imgur.com/qeeo195.jpg"
        
        # Act
        img = style_transfer.load_image(img_path)
        features = style_transfer.get_features(img, model_vgg19_mock)
        
        # Assert
        self.assertIsInstance(features, dict)
        
    @patch("torchvision.models.vgg19")
    def test_gram_matrix(self, model_vgg19_mock):
        ### DONE
        # Arrange
        style_transfer = VGG()
        test_tensor = torch.rand(1, 16, 320, 240)
        
        # Act
        gram_matrix = style_transfer.gram_matrix(test_tensor)
        
         # Assert
        self.assertIsInstance(gram_matrix, torch.Tensor)
        expected_shape = (test_tensor.shape[1], test_tensor.shape[1])
        self.assertEqual(gram_matrix.shape, expected_shape)
