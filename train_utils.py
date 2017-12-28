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
from utils import *
from glob import glob
from torch.utils.data import DataLoader
from PIL import Image
PAD = 0; SOS = 1; EOS = 2; UNK=3

def trainFor(enc, enc_opt, dec, dec_opt, tr_dl, tr_name_ids, dev_dl, dev_name_ids, val_every=1, epochs=10000, print_every=25):
    num_tr_steps = len(tr_dl.dataset.imgs) // enc.batch_size
    num_dev_steps = len(dev_dl.dataset.imgs) // enc.batch_size
    
    for epoch in range(epochs):
        gen = ImgCapLoader(tr_dl, tr_name_ids)
        loss = trainEpoch(gen, enc, enc_opt, dec, dec_opt, print_every, num_tr_steps)
        print('epoch {} avg. tr loss: {}'.format(epoch, loss))
        if (epoch + 1) % val_every == 0:
            enc.eval()
            print('Validating at epoch {}'.format(epoch))
            gen_tr_val = ImgCapLoader(tr_dl, tr_name_ids)
            tr_val_loss = valEpoch(gen_tr_val, enc, dec, num_tr_steps)
            print('epoch {} avg. tr loss, no teacher forching: {}'.format(epoch, tr_val_loss))
            
            gen_dev_val = ImgCapLoader(dev_dl, dev_name_ids)
            gen_val_loss = valEpoch(gen_dev_val, enc, dec, num_dev_steps)
            print('epoch {} avg. dev loss, no teacher forching: {}'.format(epoch, gen_val_loss))
            enc.train()