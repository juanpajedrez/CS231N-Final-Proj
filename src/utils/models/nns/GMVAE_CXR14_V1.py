"""
Date: 2023-11-14
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar
"""

import numpy as np
import torch
import os
import torch.nn.functional as F
from utils import tools as t
from torch import autograd, nn, optim
from torch.nn import functional as F
import torchvision.models as models

class Encoder(torch.nn.Module):
    def __init__(self,  z_dim, y_dim=0, pretrained=True,):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim

        #Get the path to the model
        model_path = os.path.join(os.path.dirname(__file__), \
                "vgg16_finetune", "vgg16_finetune_params.pth")

        # Load pre-trained VGG16 model
        vgg16_model = models.vgg16()
        num_features = vgg16_model.classifier[-1].in_features
        num_classes = 20

        # Replace the classifier the same way it was finetuned
        vgg16_model.classifier[-1] = nn.Linear(num_features, num_classes)
        vgg16_model.classifier.add_module("sigmoid", nn.Sigmoid())

        #Load the model with specific state
        state = torch.load(model_path)
        vgg16_model.load_state_dict(state)

        # Use only the features part and remove the classifier
        self.features = vgg16_model.features

        # Set to evaluation mode if not fine-tuning
        if not pretrained:
            self.features.eval()
        
        # Convolutional layer with kernel size 1x1
        self.conv1x1 = nn.Conv2d(512, 300, kernel_size=1)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(300)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling with kernel size equal to the feature map size
        self.max_pool = nn.MaxPool2d(kernel_size=7)

        #Obtain the net
        self.net = nn.Sequential(
            nn.Linear(300, 2 * z_dim),
        )

    def forward(self, x):
        # Create feature map from vgget16
        feat_map = self.features(x)

        # Apply operations to obtain transition layer from paper
        h = self.conv1x1(feat_map)
        h = self.batch_norm(h)
        h = self.relu(h)
        h = self.max_pool(h)

        # Convert output from 3, 300, 1, 1 to 3, 300
        h = h.view(h.shape[0], h.shape[1])

        #Now pass it through the net to obtain gaussian space
        g = self.net(h)

        #Pass the feature space and get gaussian parameters
        m, v = t.gaussian_parameters(g, dim=1)
        return m, v
    

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 784),
            nn.ELU(),
            nn.Linear(784, 1568),
            nn.ELU(),
            nn.Linear(1568, 3 * 224 * 224), 
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)