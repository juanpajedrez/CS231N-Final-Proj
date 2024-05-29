"""
Date: 2024-05-29
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

Here, we got rid of the transition layer. We also got the Z space directly from the
VGG16 features, then passed it through a linear layer which controlled with a Z dimension;
for gaussian mixture mu and covariance k component.

We kept also initial decoder for now; next version would include first iteration
with convtranspose2d
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