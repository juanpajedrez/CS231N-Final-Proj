"""
Date: 2024-05-29
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

Add the transition layer and also with the convolutional tranpose 2d
"""

import torch
from utils import tools as t
from torch import nn
import torchvision.models as models

class Encoder(torch.nn.Module):
    def __init__(self,  z_dim, y_dim=0, pretrained=True,):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim

        # Load pre-trained VGG16 model
        vgg16_model = models.vgg16(weights="IMAGENET1K_V1")

        # Use only the features part and remove the classifier
        self.features = vgg16_model.features

        # Set to evaluation mode if not fine-tuning
        if not pretrained:
            self.features.eval()

        # Use only the features part and remove the classifier
        self.features = vgg16_model.features

        # Set to evaluation mode if not fine-tuning
        if not pretrained:
            self.features.eval()

        #Obtain the number of features from vvg16
        num_features = vgg16_model.classifier[0].in_features

        # Convolutional layer with kernel size 1x1
        self.conv1x1 = nn.Conv2d(512, 300, kernel_size=1)

        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(300)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling with kernel size equal to the feature map size
        self.max_pool = nn.MaxPool2d(kernel_size=7)

        #Obtain the net
        # z space to sample from
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

        #We are gonna reconstruct an image from a 1 x 14 x 14x 14 tensor
        #To do this, we nede to pass a linear layer that outs that, and 
        #then pass it through a relu

        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 14* 14 *14),
            nn.ELU(inplace=True),
            nn.Unflatten(1, (14, 14, 14)),
            # 14 * 14 -> 28 * 28
            nn.ConvTranspose2d(14, 6, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
            # 28 * 28 -> 56 * 56
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(inplace=True),
            # 56 * 56 -> 112 * 112
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(inplace=True),
            # 112 * 112 -> 224 * 224
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(inplace=True),
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)