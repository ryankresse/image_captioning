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
from utils import *

if torch.cuda.is_available(): 
    FLOAT_DTYPE = torch.cuda.FloatTensor
else: 
    FLOAT_DTYPE= torch.FloatTensor

class Encoder(nn.Module):
    def __init__(self, batch_size=32, emb_size=300, pool_size=8, fc_in_size=2048, dropout=0.75):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.poolInput = nn.AvgPool2d(pool_size)
        self.linear = nn.Linear(fc_in_size, emb_size, bias=False)
        self.cnn = inception_v3(pretrained=True)
        self.bn = nn.BatchNorm1d(emb_size, momentum=0.1)
        self.do = nn.Dropout(dropout)
        self.cnn.eval() # IS THIS RIGHT???
        for param in self.cnn.parameters(): param.requires_grad = False
        targ_layer = self.cnn._modules.get('Mixed_7c')
        self.cnn_out = torch.zeros([self.batch_size, fc_in_size, pool_size, pool_size]).type(FLOAT_DTYPE)
        def fun(m, i, o): self.cnn_out.copy_(o.data)
        self.h = targ_layer.register_forward_hook(fun)
        
    def forward(self, x):
        _ = self.cnn(x)
        x = self.poolInput(Variable(self.cnn_out))
        x = x.view(self.batch_size, -1)
        x = self.linear(x)
        x = self.bn(x)
        x = self.do(x)
        return x #BATCH NORM AND DO HERE? IF SO NEED TO CALL EVAL AT INFERENCE
    
         
