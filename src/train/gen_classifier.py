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
from train.helpers import (compute_auc, get_data_loaders, get_dataframes,
                           p_print)
from train.vae import get_encoder


def get_classifier(architecture, dim_z, dim_y):
    if architecture == Architectures.VGG:
        classifier = nn.Sequential(
            nn.Linear(dim_z, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, dim_y),
            nn.Sigmoid(),
        )
    return classifier


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, input, target):
        # input: logits from the model (before sigmoid)
        # target: ground truth labels (0 or 1)
        logits = torch.clamp(input, min=1e-6, max=1 - 1e-6)  # To prevent log(0) error
        loss = -1 * (
            self.pos_weight * target * torch.log(logits)
            + self.neg_weight * (1 - target) * torch.log(1 - logits)
        )
        return loss.mean()  # Mean over the batch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GenClassifier(nn.Module):

    def __init__(self, architecture=Architectures.VGG, dim_z=1000, dim_y=14) -> None:
        super(GenClassifier, self).__init__()

        self.encoder_module = get_encoder(architecture)

        # lets downsample it to dim_z
        self.downsample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),  # 512x7x7 -> 256x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),  # 256x7x7 -> 128x7x7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            Flatten(),
            nn.Linear(128 * 7 * 7, dim_z),
        )

        # self.mu = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.ReLU())
        # self.logvar = nn.Sequential(nn.Linear(features_out_dim, dim_z), nn.ReLU())

        self.classifier_module = get_classifier(architecture, dim_z, dim_y)

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(logvar / 2)
    #     eps = torch.randn_like(std)
    #     z = mu + std * eps
    #     return z

    # def decode(self, z, y=None):
    #     # reshape the incoming z into 14 * 14 * 14
    #     z = z.view(-1, self.dim_z)
    #     return self.decoder_module(z)

    def forward(self, x):
        image_features = self.encoder_module(x)
        z = self.downsample(image_features)
        return self.classifier_module(z)

    def loss(self, y_hat, y):
        # dim y_hat: N, C dim y: N, C
        # calculate the weights for positive and negative samples
        total = y.shape[0] * y.shape[1]
        P = torch.sum(y)
        N = total - P

        weight_positive = total / P
        weight_negative = total / N

        total_loss = 0
        criterion = WeightedBinaryCrossEntropy(weight_positive, weight_negative)
        for c in range(y.shape[1]):
            total_loss += criterion(y_hat[:, c], y[:, c])

        # divide by num classes
        return total_loss/y.shape[1]

    def load_from_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


def evaluate(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches_to_eval = min(50, len(data_loader))

    iter_loader = iter(data_loader)

    model.eval()

    ground_truth = torch.FloatTensor()
    predictions = torch.FloatTensor()
    with torch.no_grad():
        for i in range(batches_to_eval):
            images, targets = next(iter_loader)
            images = images.to(device)

            y_hat = model(images)

            predictions = torch.cat((predictions, y_hat.cpu()), 0)
            ground_truth = torch.cat((ground_truth, targets.cpu()), 0)

        roc_score = compute_auc(ground_truth, predictions)
    return roc_score


def get_model(model):
    if isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    ):
        return model.module
    return model


def train(
    rank,
    world_size,
    learning_rate=5e-4,
    data_size="all",
    data_augmentation=False,
    batch_size=64,
):
    summary_writer = SummaryWriter(
        f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-toy-gen-class-epochs"
    )

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

    model = GenClassifier(Architectures.VGG).to(device)

    # model = get_model(model)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model_name = f"models/vae/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-gen-class-20-epochs"

    # cycle_loader = iter(train_loader)

    # Training loop
    num_epochs = 20
    best_val_accuracy = 0
    # batch_per_epoch = 100

    for epoch in range(num_epochs):
        # Initlialize the ground truth and predictions at every epoch

        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            y_hat = model(images)

            loss = model.loss(y_hat, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and i % 10 == 0:
                p_print("Epoch: ", epoch, "Iteration: ", i, "Loss: ", loss.item())
                summary_writer.add_scalar(
                    "loss", loss.item(), epoch * len(train_loader) + i
                )

        if rank == 0:
            try:
                val_accuracy = evaluate(model, val_loader)
                summary_writer.add_scalar("val_accuracy", val_accuracy, epoch)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), model_name)
                    print("saved model")
            except ValueError:
                val_accuracy = 'n/a'

            p_print(f"Epoch: {epoch}, Val_Accuracy: {val_accuracy}")
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
