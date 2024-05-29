import os
import sys

import torch
import torch.optim as optim
from torch import nn
from torchvision import models

project_path = os.environ.get("PROJECT_PATH")
data_path = os.environ.get("DATA_PATH")

# Feels hacky. Figure out a better way.
sys.path.append(os.path.join(project_path, "src"))

from train.constants import Architectures
from train.helpers import get_data_loaders, get_dataframes


def get_encoder(architecture):
    if architecture == Architectures.VGG:
        model = models.vgg16(weights="DEFAULT")
        features_module = model.features

    for param in features_module.parameters():
        param.requires_grad = False

    # ToDo: Also add transition layer here and standardise the
    # Z space
    return features_module


def get_decoder(architecture):
    if architecture == Architectures.VGG:
        # incoming image is of 14 * 14 * 14
        decoder = nn.Sequential(
            # 14 * 14 -> 28 * 28
            nn.ConvTranspose2d(14, 6, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # 28 * 28 -> 56 * 56
            nn.ConvTranspose2d(6, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # 56 * 56 -> 112 * 112
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # 112 * 112 -> 224 * 224
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
    return decoder


class VAE(nn.Module):

    def __init__(self, architecture=Architectures.VGG, dim_z=2744, dim_y=0, beta=1) -> None:
        super(VAE, self).__init__()

        self.encoder_module = get_encoder(architecture)

        ## Encoder
        # VGG16 gives 512x7x7 (25088) feature map in the end. Lets treat it as a feature
        # and learn mu and logvar from it
        # f_theta_z
        features_out_dim = 512 * 7 * 7
        self.mu = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.ReLU())
        self.logvar = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.ReLU())

        self.decoder_module = get_decoder(architecture)

        self.beta = beta

    def encode(self, x):
        # x should be of shape (batch_size, 512, 7, 7)
        x = self.encoder_module(x)

        # Flatten it
        x = x.view(x.size(0), -1)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z, y=None):
        # reshape the incoming z into 14 * 14 * 14
        z = z.view(-1, 14, 14, 14)
        return self.decoder_module(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_hat, mu, logvar):
        # Reconstruction loss as MSE
        reconstruction_loss = nn.functional.mse_loss(x, x_hat, reduction="mean")

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl_divergence


def train(learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device", device)

    data_size = "small"
    data_augmentation = False
    batch_size = 32

    dfs_holder, dfs_names = get_dataframes(
        os.path.join(project_path, "meta"), diseases="all", data=data_size
    )

    train_loader, *_ = get_data_loaders(
        dfs_holder,
        dfs_names,
        data_path,
        batch_size=batch_size,
        num_workers=2,
        data_augmentation=data_augmentation,
    )

    model = VAE(Architectures.VGG).to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(images)
            loss = model.loss(images, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss {loss.item()}")

        # Save model
        torch.save(model.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    train()
