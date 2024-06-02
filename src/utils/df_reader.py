"""
Date: 2023-11-14
Original Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez, 
Based Project: CS236 Final Project, GMVAE for X-rays images.

Date: 2024-04-30
Current Authors: Juan Pablo Triana Martinez, Abhishek Kumar
"""

#Import the necessry modules

import pandas as pd
import os
from tqdm import tqdm

class DfReader:
    """
    Class that would read the following:
    1.) Dataframe labels train
    2.) Datarframe labels test
    3.) Dataframe labels val
    """
    def __init__(self, diseases='all', data='all'):
        # code for backward compatibility
        if diseases == 'all':
            self.diseases = [
                    'Infiltration', 'Effusion', 'Atelectasis', 'Nodule',
                    'Mass', 'Pneumothorax', 'Consolidation', 'Pleural Thickening',
                    'Cardiomegaly', 'Emphysema', 'Edema', 'Subcutaneous Emphysema', 
                    'Fibrosis', 'Pneumonia'
                ]
        else:
            self.diseases = diseases
        
        # we can use small dataset to overfit various architectures to understand
        # the representational capabilities of the networks
        assert(data in ('all', 'small'))
        self.data = data

    def set_folder_path(self, folder_path:str):
        """
        Instance method that would set the folder path where
        the desired dataframes are going to be read
        Parameters:
            folder_path(str): String path that would access the data
        """
        # Define the folder_path
        self.folder_path = folder_path

    def get_columns(self):
        return ['id'] + self.diseases + ['subj_id']

    def get_dfs(self):
        """
        Instance method that would retrieve the dataframes from 
        the .csv files for test, train, and val
        """

        #Assign a dataframe holder
        dfs_holder:list = []
        dfs_names:list = []

        #Iterate through each of the directories of users
        for filename in tqdm(os.listdir(self.folder_path)):
            #Check if .csv is inside the file
            if ".csv" in filename:
                #Create a pseudo path and print the name of the file
                pseudo_path = os.path.join(self.folder_path, filename)
                df_read:pd.DataFrame = pd.read_csv(pseudo_path)
                print(f"The file: {filename} has been retrieved")

                # get the relevant columns
                df_read = df_read[self.get_columns()]

                # temporary change to get only the no finding cases for training GAN
                df_read = df_read[df_read["No Finding"] == 1]

                # filter out by how much data is needed
                df_read = self.filter_data(df_read)

                # append the dataframe to the list
                dfs_holder.append(df_read)
                dfs_names.append(filename)
        
        #Return the list with the dataframes
        return dfs_holder, dfs_names
    
    def filter_data(self, df):
        if self.data == 'all':
            return df
        
        small_data_size = 1400
        
        # in case of small data, lets try to return some 1400 images. these
        # images can be consumed in ~ 40 iterations without augmentation and 
        # around 350 iterations with augmentation

        # the incoming frame already has the relevant diseases. lets try to
        # get an even distribution of diseases in the small dataset
        df_small = pd.DataFrame()
        num_sample = small_data_size // len(self.diseases)
        for disease in self.diseases:
            df_disease = df[df[disease] == 1]
            df_disease = df_disease.sample(n=num_sample, random_state=1)
            df_small = pd.concat([df_small, df_disease])

        return df_small
