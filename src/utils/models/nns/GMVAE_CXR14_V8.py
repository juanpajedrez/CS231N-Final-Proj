"""
Date: 2024-05-31
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

Here, we will use a VGG16; BUT we would backpropagate
as well here; to improve the optimization of the GMVAE.
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
        # if not pretrained:
        #     self.features.eval()

        #Obtain the net
        # z space to sample from
        self.transition_net = nn.Sequential(
            # Batch normalization
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Max pooling with kernel size equal to the feature map size
            nn.MaxPool2d(kernel_size=7),
            ## Flatten from (Batch_num, 25088, 1, 1) -> (Batch_num, 25088)
            nn.Flatten(),
            nn.Linear(512, 2 * z_dim),
        )

    def forward(self, x):
        # Create feature map from vgget16
        feat_map = self.features(x)

        #Now pass it through the net to obtain gaussian space
        g = self.transition_net(feat_map)

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
            nn.Linear(z_dim + y_dim, 128 * 14 * 14),
            nn.ELU(inplace=True),
            nn.Unflatten(1, (128, 14, 14)),
            # 14 * 14 -> 28 * 28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 28 * 28 -> 56 * 56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 56 * 56 -> 112 * 112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 112 * 112 -> 224 * 224
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)