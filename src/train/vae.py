import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

project_path = os.environ.get("PROJECT_PATH")
data_path = os.environ.get("DATA_PATH")

# Feels hacky. Figure out a better way.
sys.path.append(os.path.join(project_path, "src"))

from train.constants import Architectures
from train.helpers import get_data_loaders, get_dataframes, p_print


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

    def __init__(
        self, architecture=Architectures.VGG, dim_z=2744, dim_y=0, beta=0.002
    ) -> None:
        super(VAE, self).__init__()

        self.encoder_module = get_encoder(architecture)
        self.dim_z = dim_z

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

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.dim_z)
        s = self.decode(z)
        return s.detach()

    def load_from_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


def evaluate(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches_to_eval = 20

    cycle_loader = iter(data_loader)

    loss = 0
    model.eval()
    with torch.no_grad():
        for i in range(batches_to_eval):
            images, _ = next(cycle_loader)
            images = images.to(device)
            x_hat, *_ = model(images)
            loss += nn.functional.mse_loss(images, x_hat, reduction="mean").item()

    return loss / batches_to_eval

def get_model(model):
  if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
    return model.module
  return model

def train(rank, world_size, learning_rate=1e-4, data_size="all", data_augmentation=False, batch_size=64):
    # summary_writer = SummaryWriter(f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-toy-vae-10-epochs")

    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(rank)
        use_multi_gpu = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False

    print("Using device", device)

    dfs_holder, dfs_names = get_dataframes(
        os.path.join(project_path, "meta"), diseases="all", data=data_size
    )

    train_loader, test_loader, val_loader = get_data_loaders(
        dfs_holder,
        dfs_names,
        data_path,
        batch_size=batch_size,
        num_workers=2,
        data_augmentation=data_augmentation,
        rank=rank,
        world_size=world_size,
        use_multi_gpu=use_multi_gpu,
    )

    model = VAE(Architectures.VGG).to(device)

    # model = get_model(model)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_name = f"models/vae/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-toy-vae-50-epochs"

    # cycle_loader = iter(train_loader)

    # Training loop
    num_epochs = 10
    best_val_loss = 1e5
    # batch_per_epoch = 100

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
                p_print("Epoch: ", epoch, "Iteration: ", i, "Loss: ", loss.item())
                # summary_writer.add_scalar("loss", loss.item(), epoch * len(train_loader) + i)

        val_loss = evaluate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)
            print("saved model")

        p_print(f"Epoch: {epoch}, Val_Loss: {val_loss}")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl",  # NCCL backend optimized for NVIDIA GPUs
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    # If CUDA is available, use it.
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(
            train,
            args=(world_size,),  # 10 epochs, for example
            nprocs=world_size,
            join=True,
        )
    else:
        train(0)

