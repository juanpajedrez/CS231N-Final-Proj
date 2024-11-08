"""
Date: 2023-11-30
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar

# We did use some come for reference of HW2 CS236 to do the primary setup from 2021 Rui Shu
"""

# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from utils import tools as t
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

#Get the optimizer
from .tools import get_optimiser

# *** ADDED CODE: Changed the name of an argument from "y_status" to "fs" (keeping same meaning) ***
def train(model, train_loader, device, tqdm, writer:SummaryWriter, config,
          iter_max=np.inf, iter_save=np.inf,
          fs=False, reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(t.reset_weights)

    #Lets get the optimizer
    optimizer = get_optimiser(config, model)
    i = 0

    #Create lists to append the loss, gen/elbo, gen/kl_z, and gen/rec
    loss_array = []
    kl_z_array = []
    rec_array =[]

    with tqdm(total=iter_max) as pbar:
        while True:
            for batch in train_loader:
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()
                xu, yu = batch

                #If condition if xu is 4D tensor, make it three
                if len(xu.shape) > 4:
                    A, B, C, D, E = xu.shape
                    xu = xu.view(A * B, C, D, E)

                    # Repeat the labels tensor to match the collapsed xu structure
                    yu = yu.unsqueeze(1).repeat(1, B, 1).view(A * B, -1)

                xu = xu.to(device)
                yu = yu.to(device)                
                if fs is False:
                    loss, kl, rec = model.loss(xu)
                else:
                    loss, kl, rec = model.loss(xu, yu)

                    #Append the loss, kl and rec
                    loss_array.append(loss.item())
                    kl_z_array.append(kl.item())
                    rec_array.append(rec.item())

                loss.backward()
                optimizer.step()

                # Feel free to modify the progress bar
                # if y_status == 'none':
                pbar.set_postfix(
                    loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    t.save_model_by_name(model, i)
                    t.save_loss_kl_rec_across_training(model.name, i, loss_array, kl_z_array, rec_array)

                if i == iter_max:
                    return
