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
import pdb
from PIL import Image
from utils import *
from encoder import *
from gru_decoder import *
FLOAT_DTYPE = torch.cuda.FloatTensor
LONG_DTYPE = torch.cuda.LongTensor



def create_emb_layer(emb_mat, non_trainable=True):
    emb_mat = torch.FloatTensor(emb_mat)
    output_size, emb_size = emb_mat.size() # get size
    emb = nn.Embedding(output_size, emb_size) #lookup table for embeddings
    emb.load_state_dict({'weight': emb_mat}) #load pretrained embeddings into embedding lookup
    if non_trainable:
        for param in emb.parameters(): 
            param.requires_grad = False
    return emb, emb_size, output_size



class GruDecoder(nn.Module):
    def __init__(self, embeds, num_layers=2, batch_size=32):
        super(GruDecoder, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed, self.emb_size, self.output_size = create_emb_layer(embeds, non_trainable=True)
        self.gru = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, num_layers=num_layers)
        self.out = nn.Linear(self.emb_size, self.output_size)
        
    
    def forward(self, x, hidden, tm1=False):
        if not tm1:
            x = self.embed(x).unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        return F.log_softmax(self.out(x.squeeze(0))), hidden
    
    
class DecoderLSTM(nn.Module):
    def __init__(self, embeds, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        self.embed, self.emb_size, self.output_size = create_emb_layer(embeds, non_trainable=True)
        #self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.embed_size, num_layers, batch_first=True)
        self.linear = nn.Linear(self.embed_size, self.output.size) ##NEED CROSS ENTROPY LOSS BC NO SOFTMAX
    
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0]) ##NEED CROSS ENTROPY LOSS BC NO SOFTMAX
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()
