## DOCUMENT DETAILS ----------------------------------------------------------
#
# Project: Project Literal: Experiment #1
# Working Title: Productivity vs. Idiosyncrasy in Collocations
# Authors: Sydelle de Souza
# Created: 2023-10-25
# Python Version: 3.9.16
# License: MIT

#-----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------
#
# this script cleans the raw json files from the experiment and merges them into
# a pandas dataframe. it then writes the dataframe to a csv file. it will write
# one csv file called experiment_data.csv in the same folder as this script if
# no folder is specified. 
# it is hard-coded to work with the data from this experiment only.
# it is not a module, it is a script (that should be run from the command line)
# if you want to reuse it for another experiment, you will need to change the
# vars in the clean_json_file function to match the vars in your json files.
# you will also need to change the path in the merge_json_files function to
# match the path to your json files.

#-----------------------------------------------------------------------------#

# Import necessary libraries
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# Function to clean json file and remove unnecessary columns
def clean_json_file(path):
    """Function to clean json file by removing unnecessary columns
    Args:
        path (str): path to the json file 
    Returns: 
        df (pandas dataframe): dataframe containing the cleaned data
    """
    with open(path) as json_file:
        json_data = json.load(json_file)

    data = [] # create an empty list to store the data that we extract from the json file
    sociodemo_data = {} # create an empty dictionary to store the sociodemographic data which is doubly nested in filedata
    for trial_vars in json_data['filedata']['trials']: # loop through the trials nested in filedata
        if trial_vars['dataType'] in ["ajtTrial"]: # only keep the trials that are relevant
            data.append(trial_vars) # append the relevant trials to the data list
        elif trial_vars['dataType'] == 'info': # if the trial is sociodemographic data
            for key, item in trial_vars['response'].items(): # loop through the keys and items in the response dictionary
                sociodemo_data[key] = item # add the key and item to the sociodemographic data dictionary
    
    df = pd.DataFrame(data) # create a pandas dataframe from the data list
    for key, item in sociodemo_data.items(): # loop through the keys and items in the sociodemographic data dictionary
        df[key] = item # add the key as the column name and the item as the column value to the dataframe
    
    df['id'] = json_data['prolific_id'] # add the prolific id to each corresponding row
    df['folda'] = json_data['folda'] # add the fold number to each corresponding row
    df['foldb'] = json_data['foldb'] # add the fold number to each corresponding row


    df = df.drop(columns=['trial_type', 'trial_index', 'time_elapsed', 'internal_node_id']) # we don't need these columns
    #remove html tags from stimulus column
    df['stimulus'] = df['stimulus'].str.replace('</p>', '')
    df['stimulus'] = df['stimulus'].str.replace('<p id="stimuli">', '')
    #remove trailing whitespace
    df['stimulus'] = df['stimulus'].str.strip()
    #capitalize first letter of each word in condition
    df['condition'] = df['condition'].str.title()
    #rename columns
    df = df.rename(columns={ 'goldResponse': 'Correct', 'rt': 'RT', 'response': 'Response', 'stimulus': 'Item', 'condition': 'Condition', 'id': 'ID', 'folda': 'Fold'})

    return df


# Function to merge json files

def merge_json_files(path):
    """Function to merge json files and write to csv
    Args:
        path (str): path to the folder containing the json files
    Returns:
        df (pandas dataframe): dataframe containing all the data from the json files
    """
    # create a list of all the json files in the data folder
    # json_paths = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    json_paths = path.glob('*.json') # glob is a unix style pathname pattern expansion module
    # create an empty list to store the data
    data = []
    # loop through the json files' paths
    for p in json_paths:
        #print the path to the file that is being processed
        print(f"* Processing {p} *")
        # open the json file
        data.append(clean_json_file(p))

    return pd.concat(data)

# clean_json_file(Path(__file__, '../../data/pilot_data/5f03689af7a20a55f0edc079.json'))

# run def merge_json_files(path) to merge json files
df = merge_json_files(Path(__file__, '../../data/experiment-data/')) # change the path to the folder containing the json files

# merge prolific csv files

# Function to clean prolific csv file and remove unnecessary columns
def clean_prolific_csv_file(path):
    """Function to clean prolific csv file by removing unnecessary columns
    Args:
        path (str): path to the csv file
    Returns:
        df (pandas dataframe): dataframe containing the cleaned data
    """
    with open(path) as csv_file:
        csv_data = pd.read_csv(csv_file)

    # create a pandas dataframe from the data list
    df = pd.DataFrame(csv_data) 

    # drop unnecessary columns
    df = df.drop(columns=['Submission id', 'Started at', 'Completed at', 'Reviewed at', 'Archived at', 'Completion code', 'Total approvals', 'Student status', 'Employment status'])
    
    # keep rows that have status as APPROVED
    df = df[df['Status'] == 'APPROVED']
    
    # replace column names with spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # rename Participant Id column to id
    df = df.rename(columns={'Participant_id': 'ID', 'Age': 'Prolific_Age'})
    
    return df

def merge_prolific_csv_files(path):
    """Function to merge prolific csv files and write to csv
    Args:
        path (str): path to the folder containing the csv files
    Returns:
        df (pandas dataframe): dataframe containing all the data from the csv files
    """
    # create a list of all the csv files in the data folder that start with prolific
    csv_paths = path.glob('prolific_export_*.csv') # glob is a unix style pathname pattern expansion module
    # create an empty list to store the data
    data = []
    # loop through the csv files' paths
    for p in csv_paths:
        #print the path to the file that is being processed
        print(f"* Processing {p} *")
        # open the csv file
        data.append(clean_prolific_csv_file(p))

    return pd.concat(data)


# run def merge_prolific_csv_files(path) to merge prolific csv files

df_prolific = merge_prolific_csv_files(Path(__file__, '../../data/experiment-data/')) # change the path to the folder containing the csv files

#rename participant id column to ID

df_prolific = df_prolific.rename(columns={'Participant id': 'ID'})


# merge prolific csv files with json files by ID
df = pd.merge(df, df_prolific, on='ID', how='left')

# write dataframe to csv file
df.to_csv("results/experiment-data.csv", index=False) # write the dataframe to a csv file in the data folder

df_prolific.to_csv("data/prolific-data.csv", index=False) # write the dataframe to a csv file in the data folder


message = "DONE! A new csv has been created. Check the data folder!"

print(f"{'*' * 60}\n\n* {message} *\n\n{'*' * 60}")
