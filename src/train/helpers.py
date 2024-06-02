import datetime
import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import models, transforms

from train.constants import Architectures, Optimisers
from utils.df_reader import DfReader
from utils.fig_reader import CXReader


# ToDo: Currently this file is specific to training for Infiltration, which is the
# highest class after No Finding. The hope is to get state of art accuracy on this
def get_dataframes(df_path, diseases="all", data="all"):
    # Create a dataframe compiler
    df_compiler = DfReader(diseases, data)
    # set the path and retrieve the dataframes
    df_compiler.set_folder_path(df_path)
    # Get the dataframe holder and names
    dfs_holder, dfs_names = df_compiler.get_dfs()
    return dfs_holder, dfs_names


def get_data_loaders(
    dfs_holder,
    dfs_names,
    data_path,
    batch_size,
    num_workers,
    data_augmentation,
    rank=0,
    world_size=1,
    use_multi_gpu=False,
    image_size=224,
):

    datasets = [
        ("train", "train.csv", True),
        ("test", "test.csv", False),
        ("val", "val.csv", False),
    ]

    # Create datasets and dataloaders
    train_dataset = CXReader(
        data_path=data_path,
        dataframe=dfs_holder[dfs_names.index("train.csv")],
        transform=get_transforms(data_augmentation, image_size=image_size),
    )
    test_dataset = CXReader(
        data_path=data_path,
        dataframe=dfs_holder[dfs_names.index("test.csv")],
        transform=get_transforms(False, image_size=image_size),
    )
    val_dataset = CXReader(
        data_path=data_path,
        dataframe=dfs_holder[dfs_names.index("val.csv")],
        transform=get_transforms(False, image_size=image_size),
    )

    # ToDo: In case of all classes, lets try to use weighted random sampler
    train_loader = create_data_loader(
        train_dataset,
        batch_size,
        use_multi_gpu,
        num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )

    test_loader = create_data_loader(
        test_dataset,
        batch_size,
        use_multi_gpu,
        num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=False,
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size,
        use_multi_gpu,
        num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=False,
    )

    return train_loader, test_loader, val_loader


def create_data_loader(
    data,
    batch_size,
    use_multi_gpu,
    num_workers,
    world_size=None,
    rank=None,
    shuffle=False,
):
    # send a distributed data sampler if using GPU otherwise, just return a data loader with shuffle
    if use_multi_gpu:
        sampler = DistributedSampler(
            data, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        return DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        sampler = RandomSampler(data, replacement=False)
        return DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )


class HistogramEqualization:
    def __call__(self, img):
        if img.mode == "RGB":
            r, g, b = img.split()
            h = ImageOps.equalize(r)
            e = ImageOps.equalize(g)
            i = ImageOps.equalize(b)
            img = Image.merge("RGB", (h, e, i))
        else:
            img = ImageOps.equalize(img)
        return img


def get_transforms(augmentaiton=False, image_size=224):
    normalize = transforms.Normalize((0.5,), (0.5,))  # normalise between [-1, 1]
    if augmentaiton:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.TenCrop(image_size),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.ToTensor()(crop) for crop in crops]
                    )
                ),
                transforms.Lambda(
                    lambda crops: torch.stack([normalize(crop) for crop in crops])
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=1),
                HistogramEqualization(),
                transforms.ToTensor(),
            ]
        )
    return transform


def custom_classifier(in_features, num_classes):
    # experiment knobs
    # 1. size of hidden layer
    # 2. number of hidden layers
    # 3. activation functions
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, 4096, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, num_classes, bias=True),
    )


def get_optimiser(model, architecture, optimiser="adam"):
    # experiment knobs
    # 1. optimiser
    # 2. learning rate
    if architecture == Architectures.DENSENET:
        parameters = model.classifier.parameters()
    elif architecture == Architectures.RESNET:
        parameters = model.fc.parameters()
    elif architecture == Architectures.VGG:
        parameters = model.classifier.parameters()

    lr = 5e-5
    if optimiser == Optimisers.ADAM:
        return torch.optim.Adam(parameters, lr=lr)
    elif optimiser == Optimisers.SGD:
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)


def get_model(architecture, num_classes):
    if architecture == Architectures.DENSENET:
        return densenet_121(num_classes)
    elif architecture == Architectures.RESNET:
        return resnet_18(num_classes)
    elif architecture == Architectures.VGG:
        return vgg_16(num_classes)


def densenet_121(num_classes):
    densenet121 = models.densenet121(weights="DEFAULT")

    # freeze parameters
    for param in densenet121.parameters():
        param.requires_grad = False

    in_features = densenet121.classifier.in_features

    densenet121.classifier = custom_classifier(in_features, num_classes)

    return densenet121


def resnet_18(num_classes):
    # get pretrained model
    resnet18 = models.resnet18(weights="DEFAULT")

    # freeze parameters
    for param in resnet18.parameters():
        param.requires_grad = False

    in_features = resnet18.fc.in_features

    resnet18.fc = custom_classifier(in_features, num_classes)

    return resnet18


def vgg_16(num_classes):
    # get pretrained model
    vgg16 = models.vgg16(weights="DEFAULT")

    # freeze parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    in_features = vgg16.classifier[0].in_features

    vgg16.classifier = custom_classifier(in_features, num_classes)

    return vgg16


def compute_auc(labels, predictions):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Returns:
        AUROC score for the class
    """
    gt_np = labels.numpy()
    pred_np = predictions.numpy()
    return roc_auc_score(gt_np, pred_np, average="macro")


def pprint(*args):
    print(" ".join(map(str, args)))
    with open("logs/log.txt", "a") as f:
        print(" ".join(map(str, args)), file=f, flush=True)


# Fix the random seed.
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def p_print(*args):
    print(" ".join(map(str, args)))

    # if log directory does not exist, create it
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # open the file in append mode
    with open("logs/log.txt", "a") as f:
        print(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + " "
            + " ".join(map(str, args)),
            file=f,
            flush=True,
        )


def save_model(model, optimizer, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    # if the parent directory does not exist, create it
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    
    torch.save(save_info, filepath)
    p_print(f"save the model to {filepath}")

def load_model(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optim"])
    random.setstate(checkpoint["system_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.random.set_rng_state(checkpoint["torch_rng"])
    p_print(f"loaded the model from {filepath}")
    return model, optimizer
    
