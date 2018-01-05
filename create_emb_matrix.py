import torch
import re, pickle, collections, numpy as np, keras, math, operator, pdb

#from gensim.models import word2vec, KeyedVectors
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision.models import inception_v3
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import re
from os import path
from glob import glob
from torch.utils.data import DataLoader
from PIL import Image
import pickle
PAD = 0; SOS = 1; EOS = 2; UNK=3



def load_glove_me(loc):
    w2v = {}
    for line in open(loc):
        l = line.split()
        w2v[l[0]] = np.array(l[1:], dtype=np.float32)
    return w2v

def create_emb_mat(targ_vocab, embed_path, glove_dict_path, dim_em=300):
    if not path.exists(glove_dict_path):
        w2v = load_glove_me(embed_path)
        pickle.dump(w2v, open(glove_dict_path, 'wb'))
    else:
        w2v = pickle.load(open(glove_dict_path, 'rb'))
    
    n_en_vec = len(w2v.keys()) #number of words encoded    
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_em)) #initialize empty container
    found=0
    for i, word in targ_vocab.idx2word.items():
        #pdb.set_trace()
        try: emb[i] = w2v[word]; found+=1 #if we find it, use the word's vecotrs
        except KeyError: emb[i] = np.random.normal(scale=0.6, size=(dim_em,)) #else randomeness
    return emb, found


    