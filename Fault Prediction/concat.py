import os
import pandas as pd

# set the path to the folder containing the CSV files
folder_path = 'promise_data'

# create an empty list to store dataframes
dfs = []

# loop through each folder in the directory
for folder in os.listdir(folder_path):
    # check if it's a directory
    if os.path.isdir(os.path.join(folder_path, folder)):
        # loop through each CSV file in the folder
        for filename in os.listdir(os.path.join(folder_path, folder)):
            if filename.endswith('.csv'):
                # read the CSV file as a dataframe
                df = pd.read_csv(os.path.join(folder_path, folder, filename))
                # append the dataframe to the list
                dfs.append(df)

# concatenate all dataframes in the list into a single dataframe
concatenated_df = pd.concat(dfs)

# save the concatenated dataframe to a CSV file
concatenated_df.to_csv('concatenated_dataset.csv', index=False)
