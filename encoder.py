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
from encoder import *
from gru_decoder import *


class Encoder(nn.Module):
    def __init__(self, batch_size=32, emb_size=300, pool_size=8, fc_in_size=2048, testing=False):
        super(Encoder, self).__init__()
        self.batch_size=batch_size
        self.emb_size= emb_size
        self.poolInput = nn.AvgPool2d(pool_size)
        self.toEmbed = nn.Linear(fc_in_size, emb_size, bias=False)
        self.cnn = inception_v3(pretrained=True)
        for param in self.cnn.parameters(): 
            param.requires_grad = False
        self.cnn.eval()
        if testing:
            pass # IS THIS RIGHT???
            #self.cnn.eval()

        targ_layer = self.cnn._modules.get('Mixed_7c')
        self.out = torch.zeros([self.batch_size, fc_in_size, pool_size, pool_size]).type(FLOAT_DTYPE)
        def fun(m, i, o): self.out.copy_(o.data)
        self.h = targ_layer.register_forward_hook(fun)
        
    def forward(self, x):
        _ = self.cnn(x)
        x = self.poolInput(Variable(self.out))
        x = x.view(self.batch_size, -1)        
        return F.tanh(self.toEmbed(x))
    
