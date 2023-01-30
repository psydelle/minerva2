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

# this file contains the code for the MINERVA2 model, a simulation model
# of human memory, which we are using to investigate collocational processing. 
# The model accounts for data from both episodic and semantic memory from a single 
# system. Theoretically speaking, the model comprises a long-term memory system
# as well as a short-term memory system that can communicate with the 
# each other. The long-term memory system is a matrix of M x N, where M is the 
# The short-term memory system can send a "probe" to the long-term memory system
# and the long-term memory system can reply with an "echo". 

#-----------------------------------------------------------------------------#

## ACKNOWLEDGEMENTS  ----------------------------------------------------------

# Ivan Vegner
# Sean Memery
# Giulio Zhou

#-----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------
import torch # for tensors
import random # for random number generation
import pandas as pd # for dataframe manipulation
import os  # for file management
import pickle # for saving and loading objects
from transformers import AutoTokenizer, AutoModel
from exract_embeddings import get_word_vector # for BERT embeddings
import matplotlib.pyplot as plt # for plotting
import numpy as np

import csv as csv # for reading in the dataset, etc.
import os # for file management
from joblib import Parallel, delayed # for parallel processing
from filelock import FileLock
#-----------------------------------------------------------------------------#

# set the random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# set current working directory to this folder 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


## read in the dataset
df = pd.read_csv("stimuli.csv") # same dataset as MSc project
dataset = list(df['item']) # list of items
fcoll = list(df['fcoll'].str.replace(r'\D', '')) # collocation frequencies

## convert the collocational frequencies to a list of floats
fcoll = [float(i) for i in fcoll]

## let's normalize fcoll to be between 0 and 1
normalized_fcoll = [float(i)/sum(fcoll) for i in fcoll]

print('loaded the dataset and normalized the collocational frequencies')

M = 10000 

if not os.path.isfile('colloc2BERT-SC-Stimuli.dat'):

    # set up the model and tokenizer for BERT embeddings
    def get_bert(mod_name="distilbert-base-uncased"): 
        tokenizer = AutoTokenizer.from_pretrained(mod_name)
        model = AutoModel.from_pretrained(mod_name, output_hidden_states=True) 
        return tokenizer, model

    def grab_bert(colloc, model, tokenizer, layers = [-4, -3, -2, -1]):
        return get_word_vector(colloc, tokenizer, model, layers) 

    # grab BERT embeddings for the items in the dataset
    colloc2BERT = dict()
    tokenizer, model = get_bert() 

    for item in dataset:
        print('dealing with this shit: ', item, '')
        colloc2BERT[item] = grab_bert(item, model, tokenizer) 

    # write the embeddings dictionary to a file to be re-used next time we run the code
        #
    colloc2BERTfile = open('colloc2BERT.dat', 'wb')
    pickle.dump(colloc2BERT, colloc2BERTfile)
    colloc2BERTfile.close()
    print("Dictionary written  to file\n")

else:
    # get the previously calculated embeddings from the file in which they were stored
    #
    colloc2BERTfile = open('colloc2BERT.dat', 'rb')
    colloc2BERT = pickle.load(colloc2BERTfile)
    colloc2BERTfile.close()   
    print("Read from file\n") 


colloc_bert_embeddings = torch.stack(list(colloc2BERT.values())) # stack the embeddings into a tensor

# sample from the collocations to make a M x 768 matrix
sampled_collocs = torch.stack(random.choices(colloc_bert_embeddings, k=M-len(colloc_bert_embeddings), weights=normalized_fcoll))
matrix = torch.concat([colloc_bert_embeddings, sampled_collocs], dim=0)
assert matrix.size() == (M, 768), "Huh?"

# Ivan's pedantic memory optimizations: since the concat we do not need the 
# original tensors anymore because they have been copied when matrix was made
del colloc_bert_embeddings, sampled_collocs



#### Now we got to add some noise to the memory matrix (parameter L)
L = 0.6 # 0.6 is what the meta paper says
# noise between 0 and 1


import matplotlib.pyplot as plt

class Minerva2(object):
    '''
    This is a class for the Minerva2 model
    '''
    def __init__(self, F=None, M=None, Mat=None):
        if Mat is not None:
            self.Mat = Mat.to(device)
            self.M = Mat.shape[0]
            self.F = Mat.shape[1]
        else:
            assert F is not None, "You need to specify the number of features"
    
    def activate(self, probe, tau=1.0):
        similarity = torch.cosine_similarity(probe, self.Mat, dim=1) # had the wrong axis
        activation = (similarity**tau) * torch.sign(similarity)  # make sure we preserve the signs
        return activation

    def echo(self, probe, tau=1.0):
        activation = self.activate(probe, tau)
        return torch.tensordot(activation, self.Mat, dims=([0], [0])) 

    def recognize(self, probe, tau=1.0, k=0.955, maxiter=450): # maxiter is set to 450 because Souza and Chalmers (2021) set their timeout to 4500ms
        echo = self.echo(probe, tau)
        similarity = torch.cosine_similarity(echo, self.Mat, dim=1)
        big = torch.max(similarity)
        if big < k and tau < maxiter:
            big, tau = self.recognize(probe, tau+1, k, maxiter)
        return big, tau
        
##-----------------------------------------------------------------------------##


## Let's run our experiment. First we generate random seeds to simulate 
## 99 l1 participants from Souza and Chalmers (2021)
n = 99 # sample size
p = 0
seed = []
for s in range(n):
    seed.append(random.randint(0, 9999999))

## Now we run the experiment
output = [] # initialize an empty list to store the output

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def iter(p, s, out_filename):
    #print(f"\nSeed {s}\n")
    random.seed(s)
    torch.manual_seed(s)
    noise_gaussian = torch.normal(0, 1, (M, 768))
    noise_mask = torch.rand((M, 768)) # noise is a tensor of random numbers between 0 and 1
    noisy_mem = torch.where(noise_mask < L, matrix + noise_gaussian, matrix) # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
    noisy_mem = noisy_mem.to(device)

    minz = Minerva2(Mat=noisy_mem) # initialize the Minerva2 model with the noisy memory matrix

    #print(f"\nBegin simulation: {n} L1 Subjects\n---------------------------------")

    for item, vector in colloc2BERT.items():
        print(f"Participant {p+1} | Seed {s} \n----------------------------------")
        #vector = colloc2BERT['forget dream']
        act, rt = minz.recognize(vector.to(device))
        output.append([item, act.detach().cpu(), rt])
        print(f"{output[-1]} \n----------------------------------") # print the last item in the list (the one we just appended)

# set up a dataframe to write the current results to a uniquely-named CSV file

        results_l1 = pd.DataFrame(data = {"mode": 'l1', 
                                        "id": [s],
                                        "participant": [p+1],
                                        "item": [item],
                                        "act": [act.item()],
                                        "rt": [rt]})

        with FileLock(out_filename+ ".lock"):
            if not os.path.exists(out_filename):
                # delete the file if it exists and write the dataframe with column names to the top of the new file
                results_l1.to_csv(out_filename, mode = 'w', header = True, index=False)
            else:
                # append the dataframe to the existing file without column names
                results_l1.to_csv(out_filename, mode = 'a', header = False, index=False)
            
    print(f" Done with Participant {p+1} | Seed {s}  \n----------------------------------")

NUM_CPUS = -1
out_file = "l1-results.csv"
if os.path.exists(out_file):
    os.remove(out_file)

results = Parallel(n_jobs=NUM_CPUS)(delayed(iter)(p,s,out_file) for p,s in enumerate(seed))    

print("********************************\n\nAll done!\n\n********************************")