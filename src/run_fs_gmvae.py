"""
Date: 2023-11-30
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

# We did use some come for reference of CS236 HW2 to do the primary setup from 2021 Rui Shu
"""

#Import necessary modules to set up the fundamentals
import os
import argparse
import tqdm

#Import necessary modules from torch to perform transformations.
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from utils.train import train
from pprint import pprint

# *** ADDED CODE: import fs_gmvae instead of gmvae ***
from utils.models.fs_gmvae import FS_GMVAE
from utils import tools as t

# Import the necessary modules
from utils import CXReader, DfReader

if __name__ == "__main__":

    #Assign the main path to be here
    os.chdir(os.path.dirname(__file__))

    #Create the data path
    data_path = os.path.join(os.getcwd(), os.pardir, "data")
    project_path = os.path.join(os.getcwd(), os.pardir, "meta")

    #Check if cuda device is in
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Get the dataframes that are necessary
    dfs_holder, dfs_names = tools.get_dataframes(
        os.path.join(project_path, "meta"), diseases="all", data=data_size
    )

    #Obtain the transfor to use
    transform = tools.get_transforms(data_augmentation)

    train_loader, test_loader, val_loader = helpers.get_data_loaders(
        dfs_holder, dfs_names, data_path, batch_size=batch_size, num_workers=2, data_augmentation=data_augmentation
    )

    _, sample_label = next(iter(train_loader))
    num_classes = sample_label.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






    #Define the neural network
    nn_type = 'GMVAE_CXR14_V1'


    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--val", action = "store_true")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
       config = yaml.safe_load(f)
    if args.train:
       train(config)
    elif args.test or args.val:


    #train_loader, labeled_subset, _ = t.get_mnist_data(device, use_test_subset=True)

    #Create a dataframe compiler
    df_compiler = DfReader()

    #set the path and retrieve the dataframes
    df_compiler.set_folder_path(data_path)

    #Get the dataframe holder and names
    dfs_holder, dfs_names = df_compiler.get_dfs()

    # Define mean and std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define a transformation for converting and normalizing images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.CenterCrop((224, 224)),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # Apply mean and std normalization
    ])
    #Create datasets and dataloaders, with batch size of 16, and shuffle true, and num workers = 4
    test_dataset = CXReader(data_path=data_path, dataframe=dfs_holder[0], transform=transform, device=device)
    train_dataset = CXReader(data_path=data_path, dataframe=dfs_holder[1], transform=transform, device=device)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    #Create a labeled_subset tuple by iterating through 100 values of test dataset
    test_images = []
    test_labels = []
    print("Loading labeled subset...")
    for i in range(100):
        #Sampled images from train to see single shape
        test_image, test_label = test_dataset[i]
        test_images.append(test_image)
        test_labels.append(test_label)

    # Convert the list of tensors to a tensor of tensors
    test_images= torch.stack(test_images, dim=0).to(device)
    test_labels= torch.stack(test_labels, dim=0).to(device)
    labeled_subset = (test_images, test_labels)
    print("Getting labeled subset")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # *** ADDED CODE: set z to 1280 (28*28 -> 224*224 is 64 times by size) ***
    parser.add_argument('--z',         type=int, default=1280,    help="Number of latent dimensions")
    parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
    parser.add_argument('--iter_max',  type=int, default=200, help="Number of training iterations")
    parser.add_argument('--iter_save', type=int, default=25, help="Save model every n iterations")
    parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
    parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
    parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
    parser.add_argument('--loss', type=str, default='bce_and_mse',  help='Flag for selecting loss')
    args = parser.parse_args()
    layout = [
        ('model={:s}',  'fs_gmvae' + args.loss),
        ('z={:02d}',  args.z),
        ('k={:03d}',  args.k),
        ('run={:04d}', args.run)
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    pprint(vars(args))
    print('Model name:', model_name)

    #Define the neural network
    nn_type = 'FSVAE_CXR14_V1_Kuni'

    fs_gmvae = FS_GMVAE(nn = nn_type, z_dim=args.z, k=args.k, name=model_name, loss_type=args.loss).to(device)

    # *** ADDED CODE: set the argument "fs" to True ***
    if args.train:
        writer = t.prepare_writer(model_name, overwrite_existing=args.overwrite)
        train(model=fs_gmvae,
            train_loader=train_loader,
            fs=True,
            device=device,
            tqdm=tqdm.tqdm,
            writer=writer,
            iter_max=args.iter_max,
            iter_save=args.iter_save)
        t.evaluate_lower_bound(fs_gmvae, labeled_subset, fs=True, run_iwae=args.train == 2)

    else:
        t.load_model_by_name(fs_gmvae, global_step=args.iter_max, device=device)
        t.evaluate_lower_bound(fs_gmvae, labeled_subset, fs=True ,run_iwae=True)