import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

project_path = os.environ.get("PROJECT_PATH")
data_path = os.environ.get("DATA_PATH")

# Feels hacky. Figure out a better way.
sys.path.append(os.path.join(project_path, "src"))

from train.helpers import get_data_loaders, get_dataframes, p_print


def get_encoder():
    # num_params: 13,93,120
    encoder = nn.Sequential(
        # 1 x 128 x 128 -> 32 x 64 x 64
        nn.Conv2d(1, 32, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2, 0),

        # 32 x 64 x 64 -> 64 x 32 x 32
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2, 0),

        # 64 x 32 x 32 -> 128 x 16 x 16
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2, 0),

        # 128 * 16 * 16 -> 256 x 8 x 8
        nn.Conv2d(128, 256, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2, 0),

        # 256 x 8 x 8 -> 512 x 4 x 4
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 512, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2, 0),

        # 512 x 4 x 4 -> 8192
        nn.Flatten(),
    )
    return encoder

def get_decoder(dim_z):
    # num_params: 10,987,457
    # incoming image is of 14 * 14 * 14
    decoder = nn.Sequential(
        nn.Linear(dim_z, 512 * 4 * 4),
        nn.Unflatten(1, (512, 4, 4)),
        
        # 512 * 4 * 4 -> 256 * 8 * 8
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        
        # 256 * 8 * 8 -> 128 * 16 * 16
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        # 128 * 16 * 16 -> 64 * 32 * 32
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        # 64 * 32 * 32 -> 32 * 64 * 64
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),

        # 32 * 64 * 64 -> 1 * 128 * 128
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    )
    return decoder


class VAE(nn.Module):

    def __init__(self, dim_z=1000, dim_y=0, beta=0.1) -> None:
        super(VAE, self).__init__()

        self.encoder_module = get_encoder()
        self.dim_z = dim_z
        
        features_out_dim = 8192

        self.mu = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.LeakyReLU(0.2, inplace=True))
        self.logvar = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.LeakyReLU(0.2, inplace=True))

        self.decoder_module = get_decoder(self.dim_z)

        self.beta = beta

    def encode(self, x):
        # x should be of shape (batch_size, 1, 128, 128)
        x = self.encoder_module(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z, y=None):
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

    batches_to_eval = min(20, len(data_loader))
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
    if isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    ):
        return model.module
    return model


def train(
    rank,
    world_size,
    learning_rate=1e-4,
    data_size="all",
    data_augmentation=False,
    batch_size=64,
):
    summary_writer = SummaryWriter(f"runs/final-vae-10-epochs")

    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(rank)
        use_multi_gpu = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False

    print("Using device", device)

    dfs_holder, dfs_names = get_dataframes(
        os.path.join(project_path, "meta"), diseases=["No Finding"], data=data_size
    )

    train_loader, test_loader, val_loader = get_data_loaders(
        dfs_holder,
        dfs_names,
        data_path,
        batch_size=batch_size,
        num_workers=4,
        data_augmentation=data_augmentation,
        rank=rank,
        world_size=world_size,
        use_multi_gpu=use_multi_gpu,
        image_size=128,
    )

    model = VAE().to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_name = f"models/vae/final-vae"

    # Training loop
    num_epochs = 10
    best_val_loss = 1e5

    for epoch in range(num_epochs):
        model.train()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(images)
            loss = model.loss(images, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                p_print("Epoch: ", epoch, "Iteration: ", i, "Loss: ", loss.item())
                summary_writer.add_scalar(
                    "loss", loss.item(), epoch * len(train_loader) + i
                )

        if rank == 0:
            val_loss = evaluate(model, val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_name)
                print("saved model")

            p_print(f"Epoch: {epoch}, Val_Loss: {val_loss}")
            summary_writer.add_scalar("loss", loss.item(), epoch)


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
        train(0, 1)
