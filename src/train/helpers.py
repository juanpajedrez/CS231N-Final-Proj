import os

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models, transforms

from utils.df_reader import DfReader
from utils.fig_reader import CXReader

class Architectures:
    VGG = 'vgg'
    DENSENET = 'densenet'
    RESNET = 'resnet'

class Optimisers:
    ADAM = 'adam'
    SGD = 'sgd'

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
    dfs_holder, dfs_names, data_path, batch_size, num_workers, data_augmentation
):
    # Create datasets and dataloaders
    train_dataset = CXReader(
        data_path=data_path,
        dataframe=dfs_holder[dfs_names.index("train.csv")],
        transform=get_transforms(data_augmentation),
    )
    test_dataset = CXReader(
        data_path=data_path, 
        dataframe=dfs_holder[dfs_names.index("test.csv")],
        transform=get_transforms(False)
    )
    val_dataset = CXReader(
        data_path=data_path, 
        dataframe=dfs_holder[dfs_names.index("val.csv")],
        transform=get_transforms(False)
    )

    # ToDo: In case of all classes, lets try to use weighted random sampler
    sampler = RandomSampler(train_dataset, replacement=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
    )

    transform_test_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 256x256
        # transforms.CenterCrop((224, 224)),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, val_loader


def get_transforms(augmentaiton=False):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augmentaiton:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.TenCrop(224),
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
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize
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
        parameters = model.classifier[6].parameters()

    if optimiser == Optimisers.ADAM:
        return torch.optim.Adam(parameters, lr=0.001)
    elif optimiser == Optimisers.SGD:
        return torch.optim.SGD(parameters, lr=0.001, momentum=0.9)

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
    resnet18 = models.resnet18(weights='DEFAULT')

    # freeze parameters
    for param in resnet18.parameters():
        param.requires_grad = False

    in_features = resnet18.fc.in_features

    resnet18.fc = custom_classifier(in_features, num_classes)

    return resnet18


def vgg_16(num_classes):
    # get pretrained model
    vgg16 = models.vgg16(weights='DEFAULT')

    # freeze parameters
    for param in vgg16.parameters():
        param.requires_grad = False

    in_features = vgg16.classifier[6].in_features

    vgg16.classifier[6] = custom_classifier(in_features, num_classes)

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
    with open('logs/log.txt', 'a') as f:
        print(" ".join(map(str, args)), file=f, flush=True)