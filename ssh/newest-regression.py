## DOCUMENT DETAILS ----------------------------------------------------------

# Project: CDT in NLP Individual Project
# Working Title: Investigating Collocational Processing with Minerva2
# Author: Sydelle de Souza
# Institution: University of Edinburgh
# Supervisors: Dr Frank Mollica and Dr Alex Doumas
# Date: 2022/12/21
# Python version: 3.9.12

#-----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------

# this file contains the code for transforming the stimuli into BERT embeddings. 
# we use a neural network to learn the weights of the normalized en 
# BERT embeddings that predict the normalized PTBERT embeddings. PT ~ weights(EN)
# These weights are then used to the transform the EN embeddings which are then
# used as input along with the PT embeddings to the MINERVA2 model in order to 
# simulate the L2 experiment.

#-----------------------------------------------------------------------------#


## ACKNOWLEDGEMENTS  ----------------------------------------------------------


#-----------------------------------------------------------------------------#


## Set-Up ---------------------------------------------------------------------

## importing packages and stuff
import numpy as np # for arrays and stuff
import pandas as pd # for dataframe manipulation
import random
import os

import tokenizers # for tokenization
import torch as torch # for tensors

import torch.nn as nn # for neural networks
import torch.optim as optim # for optimizers

import torchvision.models as models # for pretrained models

from torch.utils.data import Dataset, DataLoader # for datasets and dataloaders
from transformers import pipeline # for BERT
from handle_embeddings import *  # for transforming BERT embeddings
from exract_embeddings import *   # for extracting BERT embeddings

#-----------------------------------------------------------------------------#

## set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

## set the random seed for reproducibility
random.seed(420)
torch.manual_seed(420)
torch.cuda.manual_seed(420)

## read in the dataset
df = pd.read_csv("stimuli.csv") 

## define the models and tokenizers
layers = [-4, -3, -2, -1] # we're using the last 4 layers of BERT

en_model = AutoModel.from_pretrained('distilbert-base-cased', output_hidden_states=True)  
pt_model = AutoModel.from_pretrained('distilbert-portuguese-cased', output_hidden_states=True)  

en_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
pt_tokenizer = AutoTokenizer.from_pretrained('distilbert-portuguese-cased')  

en_space = torch.zeros((768, len(df)))
pt_space = torch.zeros((768, len(df)))

## get the BERT embeddings for the stimuli
for inx in range(en_space.shape[1]):
    en_space[:, inx] = get_word_vector(df.item[inx], en_tokenizer, en_model, layers)
    pt_space[:, inx] = get_word_vector(df.item_pt[inx], pt_tokenizer, pt_model, layers)

en_space = normalize(en_space, ['unit', 'center', 'unit'])
pt_space = normalize(en_space, ['unit', 'center', 'unit'])


## here we're creating a new class object called CollocDataset
class CollocDataset(Dataset):

    ## now we define the __init__ fxn specifically for this class
    ## which means that when this class obj is instantiated, it will have some special attributes of its own that Dataset will not

    def __init__(self, pairs):
        self.pairs = pairs.reset_index() 

    ## define len to get the length of the obj
    def __len__(self): 
        return len(self.pairs.index)  

    ## defining getitem allows you to index the elements in the obj
    def __getitem__(self, inx):
        return en_space[:, inx], pt_space[:, inx]


## assign the device as cuda if GPU is available, if not, then cpu

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


## finally, we define the neural net model (henceforth nn), not to be confused with the BERT model!
#
class CollocNet(nn.Module):
    def __init__(self):
        super(CollocNet, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(768, 768))

    def forward(self, x): 
        translated_en = self.stack(x)
        return translated_en  # returns output in the form of a tensor

## now let's set up a function to train our model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # define size as the length of the dataset
    model.train()  # this puts the model in train mode which means it will update params, backpropagate loss, apply dropout, etc.

    # iterate through the dataset in batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # convert training and test tensors to GPU compatible versions

        # Compute prediction error
        target = model(X)  # get the probability predictions of the batch (?)
        var = torch.ones(768, 768, requires_grad=True)  # TODO: makes ure the dimensions are correct
        loss = loss_fn(target, y, var)

        # Backpropagation
        optimizer.zero_grad()  # refresh the optimizer's loss function derivatives to zero for each batch, so that it doesn't backpropagate the previous batch's loss
        loss.backward()
        optimizer.step()  # here the optimizer backpropagates the loss, updating the model's params (?)

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # print some stats


## now at long last, we are at the test function
#
def test(dataloader, model, loss_fn):
    model.eval()  # here we put the model in evaluation mode, which means that any dropout is disabled
    test_loss, correct = 0, 0  # set the counters to zero

    output_df = dict()

    with torch.no_grad():  # no_grad doesn't store loss gradients, it doesn't need them as it's not going to update parameters (efficiency ?)

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)  # here it predicts logits
            var = torch.ones(768, 768, requires_grad=True)  # TODO: makes ure the dimensions are correct
            test_loss += loss_fn(pred, y, var).item()  # loss fxn gives loglikelihood of y given model, adds it to a counter

            output_df[X] = pred

    return test_loss, output_df  # return a tuple


# define the criterion---the loss function, here binary cross entropy
#
criterion = nn.GaussianNLLLoss(reduction='mean')


# hyperparameters
#
learning_rate = 1e-3  # velocity with which the params are adjusted
folds = 5  # the dataset is evenly split into 5 folds and we test using 'leave k out'
epochs = 20  # set epochs to 20

# for each fold, train on all all other folds and test on this one
#
for k in range(folds):
    print(f"\nFold {k + 1}\n")

    ## here we assign the training and test to separate CollocDataset objects and then load them into the PyTorch Dataloader class
    ## this makes it easier to iterate through the dataset in batches
    training_data = CollocDataset((df[(df['k'] != k + 1)])) 
    test_data = CollocDataset((df[(df['k'] == k + 1)]))

    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)  
                                  

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    # we need to start with a fresh model for each fold
    #
    model = CollocNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # the algorithm used to adjust the params

    ## run the functions on the actual data! it will do it over the number of epochs
    #
    test_total_loss = 0  # capture the total loss from the test data
    train_total_loss = 0  # capture the total loss from the training data

    for t in range(epochs):
        print(f"Fold {k + 1} Epoch {t + 1}\n-------------------------------")

        train(train_dataloader, model, criterion, optimizer)  # first train...

        train_total_loss, _ = test(train_dataloader, model, criterion)
        test_total_loss, translation_dictionary = test(test_dataloader, model, criterion)  # then test.

        # report status of this epoch to the console
        #
        print(f"\nTraining Report:\n\tTotal loss: {train_total_loss:>8f}")
        print(f"\nTesting Report:\n\tTotal loss: {test_total_loss:>8f}\n")

        # set up a dataframe to write the current epoch's results to a uniquely-named CSV file
        #
        train_df = pd.DataFrame(data={"mode": 'train',
                                      "epoch": [t + 1],
                                      "total_loss": [train_total_loss]})  # accuracy calculation for training data

        test_df = pd.DataFrame(data={"mode": 'test',
                                     "epoch": [t + 1],
                                     "total_loss": [test_total_loss]})  # accuracy calculation for test data

        if t == 0:
            # delete the file if it exists and write the dataframe with column names to the top of the new file
            #
            train_df.to_csv(f"results_fold_{k + 1}.csv", mode='w', header=True)
            test_df.to_csv(f"results_fold_{k + 1}.csv", mode='a', header=False)

        else:
            # append the dataframe to the existing file without column names
            #
            train_df.to_csv(f"results_fold_{k + 1}.csv", mode='a', header=False)
            test_df.to_csv(f"results_fold_{k + 1}.csv", mode='a', header=False)

    # save the model state for current fold
    #
    torch.save(model.state_dict(), f"l2_simulation_{k + 1}_epoch{t + 1}.pth")

    print(f"Done with fold {k + 1}\n")

print("********************************\n\nAll done!\n\n********************************")
