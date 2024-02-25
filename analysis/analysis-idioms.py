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
stimuli = pd.read_csv('data\stimuli_idioms.csv')
print(stimuli.head())

minerva = pd.read_csv('results\combo_results-stimuli_idioms_clean-99p-last_1-kwics-concat-m2k_0.95-m2mi_300-sweep.csv')
print(minerva.head())

# describe the data
print(stimuli.describe())
print(minerva.describe())

# Check for missing values
print(stimuli.isnull().sum())
print(minerva.isnull().sum())

# Check for duplicates
print(stimuli.duplicated().sum())
print(minerva.duplicated().sum())

#make a list of unique items in minerva['embedding_model']
models = minerva['embedding_model'].unique()
print(models)

# make a list of unique minerva_k in minerva['minerva_k']
minerva_k = minerva['minerva_k'].unique()
print(minerva_k)

# find items that are not in both dataframes
print(minerva[~minerva['item'].isin(stimuli['item'])])

# for each item not in both dataframes, add an s to the end of the character string in minerva['item']
minerva['item'] = minerva['item'].apply(lambda x: x + 's' if x not in stimuli['item'].values else x)

# find items that are not in both dataframes
print(minerva[~minerva['item'].isin(stimuli['item'])])

# merge the two dataframes on 'item'

minerva = minerva.merge(stimuli, on='item', how='left')

print(minerva.head())

# split df by minerva_k

# for i in minerva_k:
#     minerva_k_subset = minerva[minerva['minerva_k'] == i]
#     print(minerva_k_subset.head())
#     # write the updated df to a new csv
#     minerva_k_subset.to_csv(f'results\idioms_minerva_k_{i}.csv', index=False)

# for item in minerva['item']:
#     print(item)
#     if item in stimuli['item'].values:
#         minerva.loc[minerva['item'] == item, 'type'] = stimuli.loc[stimuli['item'] == item, 'type'].values[0]
#         minerva.loc[minerva['item'] == item, 'fitem'] = stimuli.loc[stimuli['item'] == item, 'fitem'].values[0]
#         minerva.loc[minerva['item'] == item, 'score'] = stimuli.loc[stimuli['item'] == item, 'score'].values[0]
#         print(minerva.loc[minerva['item'] == item, 'type'])

# write the updated df to a new csv
minerva.to_csv('results\minerva_full_results.csv', index=False)