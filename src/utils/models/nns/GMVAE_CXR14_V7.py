"""
Date: 2024-05-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

1.) Vgg16 without eval mode, all training
2.) Higher number of activatins
3.) Usage of ELU instead of Relu.

IMPORTANT: The CXR14 images transformation are not using anynore the STD
normalization values (to keep gray structures).
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
        
        #Obtain the number of features from vvg16
        num_features = vgg16_model.classifier[0].in_features

        #Obtain the net
        # z space to sample from
        self.net = nn.Sequential(
            nn.Linear(num_features, 2 * z_dim),
        )

    def forward(self, x):
        # Create feature map from vgget16
        feat_map = self.features(x)
        feat_map = feat_map.view(feat_map.shape[0], -1)

        #Now pass it through the net to obtain gaussian space
        g = self.net(feat_map)

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
            nn.ELU(inplace=True),
            # 28 * 28 -> 56 * 56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            # 56 * 56 -> 112 * 112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            # 112 * 112 -> 224 * 224
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(inplace=True),
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)