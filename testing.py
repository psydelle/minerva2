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
# system. Theoretically speaking, the model is concerned with long-term memory
# while also assuming a short-term memory system that can communicate with the 
# long-term memory system. The two memory systems communicate with each other. 
# The short-term memory system can send a "probe" to the long-term memory system
# and the long-term memory system can reply with an "echo". 

#-----------------------------------------------------------------------------#

## ACKNOWLEDGEMENTS  ----------------------------------------------------------

# Ivan Vegner
# 

#-----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------
import torch # for tensors
import random # for random number generation
import pandas as pd # for dataframe manipulation
import os  # for file management
import pickle # for saving and loading objects
from transformers import AutoTokenizer, AutoModel
from exract_embeddings import get_word_vector # for BERT embeddings

# set random seeds for reproducibility

random.seed(40)
torch.manual_seed(40)

## read in the dataset

df = pd.read_csv("FinalDataset.csv") # same dataset as MSc project
dataset = list(df['item']) # list of items
print('loaded the dataset')

M = 10000 

if not os.path.isfile('colloc2BERT.dat'):

    # set up the model
    #
    def get_bert(mod_name="distilbert-base-uncased"): 
        tokenizer = AutoTokenizer.from_pretrained(mod_name)
        model = AutoModel.from_pretrained(mod_name, output_hidden_states=True) 
        return tokenizer, model

    def grab_bert(colloc, model, tokenizer, layers = [-4, -3, -2, -1]):
        return get_word_vector(colloc, tokenizer, model, layers) 

    # grab BERT embeddings for the items in the dataset
    #
    
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
sampled_collocs = torch.stack(random.choices(colloc_bert_embeddings, k=M-len(colloc_bert_embeddings)))
matrix = torch.concat([colloc_bert_embeddings, sampled_collocs], dim=0)
assert matrix.size() == (M, 768), "Huh?"

# Ivan's pedantic memory optimizations: since the concat we do not need the 
# original tensors anymore because they have been copied when matrix was made
del colloc_bert_embeddings, sampled_collocs



# Obsolete, does the same as above:
# colloc2BERT = dict()
# for collocation in dataset:
#     print('dealing with this shit: ', collocation, '')
#     colloc2BERT[collocation] = grab_bert(collocation)

# ### For our next trick, we will sample the collocations to make a M

# sampled_collocs = random.choices(list(colloc2BERT.values()), k=M-len(colloc2BERT))

# matrix = torch.zeros((M, 768))
# for i, v in enumerate(colloc2BERT.values()):
#     matrix[i, :] = v
# for i, v in enumerate(sampled_collocs):
#     matrix[i+len(colloc2BERT), :] = v


#### Now we got to add some noise to the memory matrix (parameter L)
L = 0.6 # 0.6 is what the meta paper says
# noise between 0 and 1

noise = torch.rand((M, 768)) # noise is a tensor of random numbers between 0 and 1

noisy_mem = torch.where(noise < L, torch.zeros((M, 768)), matrix) # if the noise is less than L, then the memory is zero, otherwise it is the original matrix

print(noisy_mem, noisy_mem.shape)

class Minerva2(object):
    '''
    This is a class for the Minerva2 model
    '''
    def __init__(self, F=None, M=None, Mat=None):
        if Mat is not None:
            self.Mat = Mat
            self.M = Mat.shape[0]
            self.F = Mat.shape[1]
        else:
            assert F is not None, "You need to specify the number of features"
    
    def probe(self, probe, tau=1.0):
        similarity = torch.cosine_similarity(probe, self.Mat, dim=0)
        #similarity is always the same tensor why???? 
        activation = similarity**tau
        return activation

    def recognize(self, probe, tau=1.0, k=0.99, maxiter=4):
        activation = self.probe(probe, tau)
        big = torch.max(activation)
        if big < k and tau < maxiter: 
            activation, tau = self.recognize(probe, tau+1, k, maxiter)
            #self.recognize(probe, tau+1, k, maxiter)
        return torch.max(activation), tau
        


minz = Minerva2(Mat=noisy_mem) 
output = [] 


for item, vector in colloc2BERT.items():
    act, rt = minz.recognize(vector)
    print("appending to output" + item)
    output.append([item, act, rt])
#act, rt = minz.recognize(colloc2BERT['chase dream'])
print(output)

