"""
Date: 2023-11-14
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CXReader(Dataset):
    """
    Class that would perform the following:
    1.) Read all images from the data folder.
    2.) Read the dataframes with the metadata of the files.
    """

    def __init__(self, dataframe:pd.DataFrame, data_path: str,
                device:str, transform:transforms.Compose = None):
        """
        Parameters:
            dataframe(pd.Dataframe): Contains all of the main information
            of the data in question, ids, labels.
            data_path(str): The string data path to access all
            of the required files.
            device(str): Device to send data for processing
            transform(transforms.Compose): Any required torch transforms
            necessary to convert the iamges (for VGGnet16 should be 224, 224).
        """
        # Define the folder_path
        self.data_path = data_path 
        self.dataframe = dataframe
        self.transform = transform 

    def __len__(self):
        """
        Instance method of the abstract class Dataset from torch,
        should return the length of all data entries in question.
        """
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        """
        Instance method of the abstract class Dataset from torch,
        should retrun the desired image to select from.
        """
        
        # Based on the idx, select the desired row
        row = self.dataframe.iloc[idx]

        img_name = row['id']

        #Obtain the values for the binarized labels
        #REMEMBER: Daframe first column are images names
        #and last column is the associated patient id, so drop these.
        label = row[1:-1].values.astype(np.float16)
        label = torch.from_numpy(label)

        #Create an image path wth the datapath
        img_path = os.path.join(self.data_path, img_name)

        #Load the image
        image = Image.open(img_path).convert("RGB")

        #Perform any necessary preprocessing
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
    