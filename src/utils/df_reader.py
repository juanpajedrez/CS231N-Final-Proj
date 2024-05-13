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
        """
        In MICCAI 2023 data, we have 20 diseases. We will only be using 14 diseases for our project.
        The top 14 entries are:
        No Finding                    59406
        Infiltration                  19894
        Effusion                      13317
        Atelectasis                   11559
        Nodule                         6331
        Mass                           5782
        Pneumothorax                   5302
        Consolidation                  4667
        Pleural Thickening             3385
        Cardiomegaly                   2776
        Emphysema                      2516
        Edema                          2303
        Subcutaneous Emphysema         1991
        Fibrosis                       1686
        Pneumonia                      1431
        Tortuous Aorta                  742
        Calcification of the Aorta      455
        Pneumoperitoneum                316
        Pneumomediastinum               253
        Hernia                          227
        """
        # Top 14 statistically significant diseases
        diseases = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule',
                    'Mass', 'Pneumothorax', 'Consolidation', 'Pleural Thickening',
                    'Cardiomegaly', 'Emphysema', 'Edema', 'Subcutaneous Emphysema', 
                    'Fibrosis', 'Pneumonia', 'No Finding']
        
        return ['id'] + diseases + ['subj_id']

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

                # append the dataframe to the list
                dfs_holder.append(df_read)
                dfs_names.append(filename)
        
        #Return the list with the dataframes
        return dfs_holder, dfs_names
