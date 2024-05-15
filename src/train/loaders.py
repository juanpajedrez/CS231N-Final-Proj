import os

from torchvision import transforms
from utils.df_reader import DfReader
from utils.fig_reader import CXReader


# ToDo: Currently this file is specific to training for Infiltration, which is the 
# highest class after No Finding. The hope is to get state of art accuracy on this
def get_loaders(data_path, df_path, batch_size, num_workers):
    #Create a dataframe compiler
    df_compiler = DfReader(diseases='Infiltration')
    #set the path and retrieve the dataframes
    df_compiler.set_folder_path(df_path)
    #Get the dataframe holder and names
    dfs_holder, dfs_names = df_compiler.get_dfs()

    train_df = dfs_holder[dfs_names.index('train.csv')]

    #Define transformations
    transform = transforms.Compose([
        transforms.Resize(256), # resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Create datasets and dataloaders
    train_dataset = CXReader(data_path=data_path, dataframe=dfs_holder[dfs_names.index('train.csv')], transform=transform)
    test_dataset = CXReader(data_path=data_path, dataframe=dfs_holder[dfs_names.index('test.csv')])
    val_dataset = CXReader(data_path=data_path, dataframe=dfs_holder[dfs_names.index('val.csv')])

    # Create a sampler to handle class imbalance
    class_counts = train_df.Infiltration.value_counts()
    class_weights = 1. / class_counts
    samples_weights = torch.Tensor(class_weights[train_df.Infiltration].to_list())
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)







