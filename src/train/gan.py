import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

load_dotenv()

project_path = os.environ.get("PROJECT_PATH")
data_path = os.environ.get("DATA_PATH")

# Feels hacky. Figure out a better way.
sys.path.append(os.path.join(project_path, "src"))

from train.helpers import get_data_loaders, get_dataframes, p_print


def get_optimizer(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    loss = (torch.mean((scores_real - 1) ** 2) + torch.mean(scores_fake**2)) / 2
    return loss


def ls_generator_loss(scores_fake):
    loss = torch.mean((scores_fake - 1) ** 2) / 2
    return loss


def sample_noise(batch_size, dim, seed=None):
    return 2 * torch.rand(batch_size, dim) - 1  # noise between -1 and 1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # classifier layer
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)


class Generator(nn.Module):
    def __init__(self, dim_z=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # dim_z x 1 x 1 -> 512 x 4 x 4
            nn.ConvTranspose2d(dim_z, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # state size. 512 x 4 x 4 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # state size. 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # state size. 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # state size. 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # state size. 32 x 64 x 64 -> 1 x 128 x 128
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. 1 x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

    def load_from_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


def train(
    rank,
    world_size,
    data_size="all",
    data_augmentation=False,
    batch_size=64,
    num_epochs=10,
    dim_z=1024,
):
    summary_writer = SummaryWriter(
        f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-gan_2-10-epochs"
    )

    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(rank)
        use_multi_gpu = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("Using device", device)

    dfs_holder, dfs_names = get_dataframes(
        os.path.join(project_path, "meta"), diseases=["No Finding"], data=data_size
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
        image_size=128
    )

    discriminator = Discriminator()
    if world_size > 1:
        discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    discriminator = discriminator.to(device)

    generator = Generator(dim_z=dim_z)
    if world_size > 1:
        generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    generator = generator.to(device)

    if world_size > 1:
        discriminator = DDP(discriminator, device_ids=[rank])
        generator = DDP(generator, device_ids=[rank])

    D_solver = get_optimizer(discriminator, lr=2e-4)
    G_solver = get_optimizer(generator, lr=2e-4)

    for epoch in range(num_epochs):

        generator = generator.train()
        discriminator = discriminator.train()

        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            images = images.type(dtype)

            D_solver.zero_grad()
            logits_real = discriminator(images)

            g_fake_seed = sample_noise(batch_size, dim_z).to(device)
            g_fake_seed = g_fake_seed.view(batch_size, dim_z, 1, 1)

            # detach it since we don't want it in the computation graph during backprop
            fake_images = generator(g_fake_seed).detach()
            logits_fake = discriminator(fake_images).squeeze()

            d_total_error = ls_discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, dim_z).to(device)
            g_fake_seed = g_fake_seed.view(batch_size, dim_z, 1, 1)
            gen_fake_images = generator(g_fake_seed)
            gen_logits_fake = discriminator(gen_fake_images)
            g_error = ls_generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if rank == 0 and i % 10 == 0:
                p_print(
                    "Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}".format(
                        epoch, i, d_total_error.item(), g_error.item()
                    )
                )
                summary_writer.add_scalar("discriminator_loss", d_total_error.item(), epoch * len(train_loader) + i)
                summary_writer.add_scalar("generator_loss", g_error.item(), epoch * len(train_loader) + i)

            if rank == 0 and i % 150 == 0:
                # put generator in evaluation mode
                generator = generator.eval()
                with torch.no_grad():
                    g_fake_seed = sample_noise(1, dim_z).to(device)
                    g_fake_seed = g_fake_seed.view(1, dim_z, 1, 1)
                    output = generator(g_fake_seed)
                    vutils.save_image(
                        output.data,
                        f"output/gan_2/fake_samples_epoch_{epoch}_{i}.png",
                        normalize=True,
                    )

                # put generator back in training mode
                generator = generator.train()

        # save the model
        if rank == 0:
            torch.save(generator.state_dict(), f"model/gan/bn_2_generator-{epoch}.pt")
            torch.save(
                discriminator.state_dict(), f"model/gan/bn_2_discriminator-{epoch}.pt"
            )

    if world_size > 1:
        cleanup()


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
