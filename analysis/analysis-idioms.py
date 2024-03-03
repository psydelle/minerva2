import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set up pandas options for data exploration
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

# Set up matplotlib for plotting
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Set up seaborn for plotting
sns.set_theme(style='whitegrid', palette='flare')

# Read the CSV file into a pandas DataFrame
stimuli = pd.read_csv('data\stimuli_idioms_clean.csv')
print("Stimuli data: ")
print(stimuli.head())

minerva = pd.read_csv('results\\results-stimuli_idioms_clean-300p-last_1-kwics-concat-all-fp_0.0-0.8-m2k_0.95-0.995-m2mi_300-full-sweep.csv')
print("Minerva data: ")
print(minerva.head())

# describe the data
print("Stimuli data description: ")
print(stimuli.describe())
print("Minerva data description: ")
print(minerva.describe())

# Check for missing values
print("Missing values in stimuli:", stimuli.isnull().sum())
print("Missing values in minerva:", minerva.isnull().sum())

# Check for duplicates
print("Number of duplicates in stimuli: ", stimuli.duplicated().sum())
print("Number of duplicates in minerva: ", minerva.duplicated().sum())


#make a list of unique items in minerva['embedding_model']
models = minerva['embedding_model'].unique()
print(models)

# make a list of unique minerva_k in minerva['minerva_k']
minerva_k = minerva['minerva_k'].unique()
print(minerva_k)

# find items that are not in both dataframes
print(minerva[~minerva['item'].isin(stimuli['item'])])

# for each item not in both dataframes, add an s to the end of the character string in minerva['item']
# minerva['item'] = minerva['item'].apply(lambda x: x + 's' if x not in stimuli['item'].values else x)

# find items that are not in both dataframes
print(minerva[~minerva['item'].isin(stimuli['item'])])

# merge the two dataframes on 'item'
minerva = minerva.merge(stimuli, on='item', how='left')
print(minerva.head())


# split the dataset by embedding model and write to a new csv
for model in models:
    minerva_model = minerva[minerva['embedding_model'] == model]
    print("Model: ", model)
    # write the updated df to a new csv
    minerva_model.to_csv(f'results\minerva_results_{model}.csv', index=False)
    print("Done with ", model, "! Check the results folder for the CSV file.")

# write the updated df to a new csv
print("Writing the updated dataframe to a new CSV file...")
minerva.to_csv('results\minerva_results_full.csv', index=False)
print("Done! Check the results folder for the updated CSV file.")