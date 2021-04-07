import unittest
from collections import OrderedDict
import io

import mock
from mock import patch
import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image

from .vgg import VGG

# Stub of response.content
class Response():
    def __init__(self):
        # Create random array
        np_img = np.random.random((640, 480, 3))
        # Transform array to PIL image
        pil_image = Image.fromarray(np.uint8(np_img * 255)).convert('RGB')
        # Transform PIL image to byte array
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        self.content = byte_arr

# Stub of get
def get(path):
    res = Response()
    return res

class VGGTest(unittest.TestCase):

    @patch("torchvision.models.vgg19")
    def test_vgg_init_attributes(self, model_vgg19_mock):
        ### ASSERT COMPLETE ###
        # Act
        style_transfer = VGG()

        # Assert
        # Assert torch device
        self.assertIsInstance(style_transfer.device, torch.device)
        # Assert style weights
        self.assertEqual(style_transfer.style_weights, {'conv1_1': 1., 
                                                        'conv2_1': 0.8,
                                                        'conv3_1': 0.5,
                                                        'conv4_1': 0.3,
                                                        'conv5_1': 0.1})
        # Assert content and style weights
        self.assertEqual(style_transfer.content_weight, 1)
        self.assertEqual(style_transfer.style_weight, 1e6)
        
    @patch("torchvision.models.vgg19")
    @patch("requests.get", get)
    def test_load_image(self, model_vgg19_mock):
        ### ASSERT COMPLETE ###
        # Arrange
        style_transfer = VGG()
        img_path = None
        
        # Act
        img, img_h, img_w = style_transfer.load_image(img_path)
        
        # Assert
        # Assert data type
        self.assertIsInstance(img, torch.Tensor)
        # Assert shape of new img
        expected_shape = torch.Size([1, 3, 533, 400])
        self.assertEqual(img.shape, expected_shape)
        # Assert number of channels of image
        expected_channels = 3
        self.assertEqual(img.shape[1], expected_channels)
        # Assert expected dim of final image size
        expected_dim = 400
        self.assertTrue(expected_dim in img.shape[2:])
        # Assert batch size
        batch_size = 1
        self.assertEqual(img.shape[0], batch_size)
        # Assert original image size vs new image size
        self.assertLessEqual(img.shape[2], img_w)
        self.assertLessEqual(img.shape[3], img_h)
        
        ### Evaluating a particular max_size
        
        # Act
        expected_dim = 600
        img, img_h, img_w = style_transfer.load_image(img_path, max_size = expected_dim)
        
        # Assert
        # Assert new expected dim in width or height
        self.assertTrue(expected_dim in img.shape[2:])
        # Assert shape of new img
        expected_shape = torch.Size([1, 3, 800, 600])
        self.assertEqual(img.shape, expected_shape)
        # Assert original image size vs new image size
        self.assertNotEqual(img.shape[2], img_w)
        self.assertNotEqual(img.shape[3], img_h)
        
        ### Evaluating shape != None
        
        # Act
        expected_shape = [400, 500]
        img, img_w, img_h = style_transfer.load_image(img_path, shape = expected_shape)
        
        # Assert
        # Assert expected width and height in new image
        self.assertTrue(img.shape[2:] == torch.Size(expected_shape))
        # Assert shape of new img
        expected_shape = torch.Size([1, 3, expected_shape[0], expected_shape[1]])
        self.assertEqual(img.shape, expected_shape)
        # Assert original image size vs new image size
        self.assertNotEqual(img.shape[3], img_h)
        self.assertNotEqual(img.shape[2], img_w)
        
        
        
    @patch("torchvision.models.vgg19")
    def test_im_convert(self, model_vgg19_mock):
        ### ASSERT COMPLETE ###
        # Arrange
        style_transfer = VGG()
        # Random tensor
        test_tensor = torch.rand(1, 3, 640, 480)
        
        # Act
        im_converted = style_transfer.im_convert(test_tensor)
        
        # Assert
        # Assert data type
        self.assertIsInstance(im_converted, np.ndarray)
        # Assert max value
        expected_max = 1
        self.assertLessEqual(im_converted.max(), expected_max)
        # Assert min value
        expected_min = 0
        self.assertGreaterEqual(im_converted.min(), expected_min)
        # Assert final expected shape
        expected_shape = (test_tensor.shape[2], test_tensor.shape[3], test_tensor.shape[1])
        self.assertEqual(im_converted.shape, expected_shape)
        
    @patch("torchvision.models.vgg19")
    @patch("requests.get", get)
    def test_get_features(self, model_vgg19_mock):
        ### TODO ASSERT MORE THINGS, I do not know how...
        # Arrange
        style_transfer = VGG()
        img_path = None
        
        # Act
        img = style_transfer.load_image(img_path)
        features = style_transfer.get_features(img, model_vgg19_mock)
        
        # Assert
        self.assertIsInstance(features, dict)
        
    @patch("torchvision.models.vgg19")
    def test_gram_matrix(self, model_vgg19_mock):
        ### ASSERT COMPLETE ###
        # Arrange
        style_transfer = VGG()
        test_tensor = torch.rand(1, 16, 320, 240)
        
        # Act
        gram_matrix = style_transfer.gram_matrix(test_tensor)
        
        # Assert
        # Assert data type
        self.assertIsInstance(gram_matrix, torch.Tensor)
        # Assert gram matrix size
        expected_shape = (test_tensor.shape[1], test_tensor.shape[1])
        self.assertEqual(gram_matrix.shape, expected_shape)
