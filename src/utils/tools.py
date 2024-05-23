"""
Date: 2023-11-30
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

# We did use some come for reference of HW2 CS236 to do the primary setup from 2021 Rui Shu
# Also, we added our own functions to get the pre processed data from.
"""

import numpy as np
import os
import shutil
import torch

# import tensorflow as tf
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

#Import the ncessary modules
from .df_reader import DfReader
from .fig_reader import CXReader

bce = torch.nn.BCEWithLogitsLoss(reduction='none')
mse = torch.nn.MSELoss(reduction="none")

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
                transforms.CenterCrop((224, 224)),  # Center crop to 224x224
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

def get_optimiser(config, model, lr_set:float = 5e-3):
    lr = lr_set
    if config["training"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config["training"]["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

import torch.nn.functional as F

def evaluate_model(model, data_loader, labels, device:str):
    """
    Instance method that would evaluate with a given
    data loader, the accuracies obtained by the model passed
    """
    model.eval()
    threshold = 0.5
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    #Use no grad to not perform backpropagation for inference time
    with torch.no_grad():
        #Iterate through each of the images and labels
        
        # Calculate the total numbers for metrics
        TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
        for idx, batch in enumerate(data_loader):
    
            #See if it works
            images_inputs, images_labels = batch
            images_inputs, images_labels = images_inputs.to(device), images_labels.to(device)

            #Print the shape of each one of them
            print(f"Inputs shape: {images_inputs.shape}, Labels shape: {labels.shape}")

            #Send the outputs to model in device
            outputs = model(images_inputs)

            #Binarize the output with threshold
            pred_labels = (outputs > threshold).float()

            # Calculate batch-wise TP, FP, TN, FN
            b_TP = torch.sum((pred_labels == 1) & (images_labels == 1)).item()
            b_FP = torch.sum((pred_labels == 1) & (images_labels == 0)).item()
            b_TN = torch.sum((pred_labels == 0) & (images_labels == 0)).item()
            b_FN = torch.sum((pred_labels == 0) & (images_labels == 1)).item()
            TP += b_TP
            FP += b_FP
            TN += b_TN
            FN += b_FN

        #_, predicted = torch.max(outputs, 1)  # Get the index of the maximum log-probability
        accuracy = ((TP + TN) / (TP + FP + TN + FN)) * 100.0
        precision = (TP / (TP + FP)) * 100.0 if (TP + FP) > 0 else 0.0
        recall = (TP / (TP + FN)) * 100.0 if (TP + FN) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print("Accuracy: {:.2f}%".format(accuracy))
        print("Precision: {:.2f}%".format(precision))
        print("Recall: {:.2f}%".format(recall))
        print("F1 Score: {:.2f}%".format(f1_score))

    return accuracies, precisions, recalls, f1_scores



### ALL Helper functions that are necessary to get the 

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def z_score_normalize(tensor, dim=(1, 2)):
    """
    Function to z score a tesnor of batch size x dim
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    z_scored = (tensor - mean) / std
    return z_scored

def scale_to_01(tensor):
    """
    Function to scale score a tesnor of batch size x dim
    """
    scaled_tensor = (tensor - tensor.min().item()) / (tensor.max().item() - tensor.min().item())
    return scaled_tensor

def log_pixel_with_logits(x, logits):
    """
    Additional function!: Would compute the log
    probability from pixel values using logits.
    THIS FUNCTION Uses a specific designed loss
    """
    # scale the input betwen 0 to 1
    x = scale_to_01(x)

    #Obtain the sigmoid of the logits
    logits = torch.sigmoid(logits)

    # Calculate min and max values for all batches
    #min_values = x.min().item()
    #max_values = x.max().item()

    # Calculate scale and shift factors for each batch
    #scale_factors = (max_values - min_values) / 2.0
    #shift_factors = (max_values + min_values) / 2.0

    # Apply scaled and shifted tanh function to logits
    #logits = scale_factors * torch.tanh(logits) + shift_factors
    
    #Find the mse of it
    log_prob = -mse(logits, x)
    return log_prob

def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = -bce(input=logits, target=x)
    return log_prob

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, dim): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z = mu + sigma * e where e is from a gaussian distribution. normalized between 0 and Identity matrix
    ################################################################################
    # Determine epsilon distribution
    epsilon = torch.randn_like(v)
    
    #Calculate Z
    z = m + torch.sqrt(v) * epsilon 
    ################################################################################
    # End of code modification
    ################################################################################
    return z

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    # ASSUME that ln is log for approx to decompose the normal distribution
    log_first_term = -(torch.pow((x-m), 2)/(2*v))
    log_sec_term = -np.log(np.sqrt(2*np.pi))
    log_third_term = -torch.log(torch.sqrt(v))

    #Add them all
    log_probs = log_first_term + log_sec_term + log_third_term
    log_prob = log_probs.sum(-1)

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob

def log_normal_mixture(z, m, v):
    """
    Computes log probability of Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################

    #Determine the zs used for multi variate prior distribution
    multi_zs_prior = z.unsqueeze(1).expand_as(m)

    # ASSUME that ln is log for approx to decompose the normal distribution
    log_first_term = -(torch.pow((multi_zs_prior-m), 2)/(2*v))
    log_sec_term = -np.log(np.sqrt(2*np.pi))
    log_third_term = -torch.log(torch.sqrt(v))

    #Add them all
    log_probs = log_first_term + log_sec_term + log_third_term
    prob_sums = log_probs.sum(-1)
    log_prob = log_mean_exp(prob_sums, -1)   

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))

def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()

def load_model_by_name(model, global_step, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    #Had to modify to add a path
    file_path = os.path.join(os.getcwd(), 'checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def evaluate_lower_bound(model, labeled_test_subset, fs=False, run_iwae=True):
    "We will come back to change to our model class GMVAE"
    #check_model = isinstance(model, GMVAE)
    #assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    xl, yl = labeled_test_subset

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for i in range(repeat):
            print(f"estimate of metrics:...{i + 1}")
            if fs:
                niwae, kl, rec = detach_torch_tuple(fn(xl, yl))
            else:
                niwae, kl, rec = detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 2)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

    if run_iwae:
        for iw in [20]:
            repeat = max(100 // iw, 1) # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))

def save_loss_kl_rec_across_training(model_name, global_step, loss_array, kl_array, rec_array, overwrite_existing=False):
    """
    Additional function :) to save in .npy format inside the checkpoints folder for later to use
    """
    #Set paths for checkpoints saving
    save_dir = os.path.join('checkpoints', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Create file paths for loss, kl, and rec
    loss_path = os.path.join(save_dir, 'loss-{:05d}.pt'.format(global_step)+ ".npy")
    kl_path = os.path.join(save_dir, 'kl-{:05d}.pt'.format(global_step) + ".npy")
    rec_path = os.path.join(save_dir, 'rec-{:05d}.pt'.format(global_step)+ ".npy")

    #np.save
    np.save(loss_path, loss_array)
    np.save(kl_path, kl_array)
    np.save(rec_path, rec_array)


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    maybe_delete_existing(log_dir, overwrite_existing)
    maybe_delete_existing(save_dir, overwrite_existing)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer

def maybe_delete_existing(path, overwrite_existing):
    if not os.path.exists(path):
        return

    if overwrite_existing:
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)
    else:
        raise FileExistsError(
            """
    Unpermitted attempt to delete {}.
    1. To overwrite checkpoints and logs when re-running a model, remember to pass --overwrite 1 as argument.
    2. To run a replicate model, pass --run NEW_ID where NEW_ID is incremented from 0.""".format(path))

def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass