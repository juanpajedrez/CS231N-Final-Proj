"""
Date: 2023-11-30
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30 -> 2024-05-22
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

# We did use some come for reference of CS236 HW2 to do the primary setup from 2021 Rui Shu
"""

#Import necessary modules to set up the fundamentals
import os
import argparse
import tqdm
import torch
import yaml

# == ADDED CODE that is necessary == #
from utils.models.gmvae import GMVAE
from utils import tools as t

#Import the necessary modules
from utils import train
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    #Get the parser so we can run something
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--val", action = "store_true")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
       config = yaml.safe_load(f)

    #Assign the main path to be here
    os.chdir(os.path.dirname(__file__))

    #Create the data path
    data_path = os.path.join(os.getcwd(), os.pardir, "data", "images", "images")
    project_path = os.path.join(os.getcwd(), os.pardir, "meta")

    #Check if cuda device is in
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Get the dataframes that are necessary
    dfs_holder, dfs_names = t.get_dataframes(
        project_path, diseases="all", data=config["training"]["data_size"]
    )

    #Obtain the transfor to use
    transform = t.get_transforms(config["training"]["data_augmentation"])

    #Get the the loaders to perform training, validation, and testing
    train_loader, test_loader, val_loader = t.get_data_loaders(
        dfs_holder, dfs_names, data_path, batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"], data_augmentation=config["training"]["data_augmentation"]
    )

    #Obtain the total number of classes
    num_classes = config["model"]["num_classes"]

    #Obtain the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = [
        ('model={:s}',  'gmvaetest'+config["model"]["loss"]),
        ('z={:02d}',  config["model"]["z"]),
        ('k={:03d}',  config["model"]["k"]),
        ('run={:04d}', config["model"]["run"])
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])

    #Assign type of loss
    gmvae = GMVAE(nn=config["model"]["nn_type"], z_dim=config["model"]["z"],\
        k=config["model"]["k"], name=model_name, loss_type=config["model"]["loss"]).to(device)

    #Lets create a summary writer
    #writer = SummaryWriter(os.path.join(os.getcwd(), os.pardir,\
    #    'runs', f"{model_name}_lr_{train_lr}_loss_{tra}_{config["training"]["num_epochs"]}epochs"))
    writer = None

    #Set the args when train, val, and test.
    if args.train:
        train(model=gmvae,
            train_loader=train_loader,
            fs=False, 
            device=device,
            tqdm=tqdm.tqdm,
            writer=writer,
            config=config,
            iter_max=config["model"]["iter_max"],
            iter_save=config["model"]["iter_save"])

    elif args.val:
        t.load_model_by_name(gmvae, global_step=config["model"]["iter_max"], device=device)
        accuracies, precisions, recalls, f1_scores = t.evaluate_model(gmvae, val_loader, device)
    elif args.test:
        t.load_model_by_name(gmvae, global_step=config["model"]["iter_max"], device=device)
        accuracies, precisions, recalls, f1_scores = t.evaluate_model(gmvae, test_loader, device)
