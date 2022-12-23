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

# this file contains the code for the Minerva2 model, a simulation model
# of human memory, which we are using to investigate collocational processing. 
# The model is based on the Minerva model by Hintzmann (1986)

#-----------------------------------------------------------------------------#



import random
print('imported random')
import pandas as pd
print('imported pandas')
from exract_embeddings import *
print('imported extract_embeddings')

## read in the dataset
#
df = pd.read_csv("FinalDataset.csv")
dataset = list(df['item'])[0:10]
print('loaded the dataset')

def get_bert(mod_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(mod_name)
    model = AutoModel.from_pretrained(mod_name, output_hidden_states=True)
    return tokenizer, model

def grab_bert(colloc, model, tokenizer, layers = [-4, -3, -2, -1]):
    return get_word_vector(colloc, tokenizer, model, layers) 


M = 10000
colloc2BERT = dict()
tokenizer, model = get_bert()
for collocation in dataset:
    print('dealing with this shit: ', collocation, '')
    colloc2BERT[collocation] = grab_bert(collocation, model, tokenizer)

colloc_bert_embeddings = torch.stack(list(colloc2BERT.values()))
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


#### Now we got to add some noise to the memory matrix (paramerer L)
L = 0.6 # cuz this is what the meta paper says
# noise between 0 and 1
noise = torch.rand((M, 768))

noisy_mem = torch.where(noise < L, torch.zeros((M, 768)), matrix)

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
        activation = similarity**tau
        return activation

    def recognize(self, probe, tau=1.0, k=0.99, maxiter=1000):
        activation = self.probe(probe, tau)
        if torch.argmax(activation) < k and tau < maxiter:
            self.recognize(probe, tau+1, k, maxiter)
        else:
            return torch.max(activation), tau
        


minz = Minerva2(Mat=noisy_mem)
act, rt = minz.recognize(colloc2BERT['recall vow'])
print(act, rt)








