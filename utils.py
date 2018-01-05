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
PAD = 0; SOS = 1; EOS = 2; UNK=3

def validate(dl, cap_dict, enc, dec, batch_size=32, crit=nn.NLLLoss()):
    loader = ImgCapLoader(dl, cap_dict, batch_size)
    num_steps = len(dl.dataset.imgs) // BATCH_SIZE
    pdb.set_trace()
    tot_loss = 0.0
    for i, (imgs, ids) in enumerate(loader):
        if i == (num_steps - 1):break
        print('Validating {} of {} steps'.format(i, num_steps))
        loss = 0.0
        targ_len = ids.shape[1]
        ids = Variable(torch.LongTensor(ids)).type(LONG_DTYPE)
        dec_input = long_t([SOS]*imgs.size()[0]).type(LONG_DTYPE)
        hidden = enc(Variable(imgs).type(FLOAT_DTYPE)).unsqueeze(0).repeat(dec.num_layers, 1, 1)
        for di in range(targ_len):
            dec_output, hidden = dec(dec_input, hidden) #get output hidden
            _, dec_input = dec_output.topk(1)            
            dec_input = dec_input.squeeze(1)
            loss += crit(dec_output, dec_input) #compute loss
        tot_loss += loss.data[0]
    return tot_loss / (num_steps - 1)


re_apos = re.compile(r"(\w)'s\b")         # make 's a separate word
re_mw_punc = re.compile(r"(\w[’'])(\w)")  # other ' in a word creates 2 words
re_punc = re.compile("([\"().,;:/_?!—])") # add spaces around punctuation
re_mult_space = re.compile(r"  *")        # replace multiple spaces with just one

def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()

PAD = 0; SOS = 1; EOS = 2; UNK=3

def toks2ids(sents):
    #create new counter from all tokens in all lines
    sents_vals = list(sents.values())
    voc_cnt = collections.Counter(t for sent in sents_vals for t in sent) 
    
    #sort vocab in reverse order
    vocab = sorted(voc_cnt, key=voc_cnt.get, reverse=True)
    
    vocab.insert(PAD, "<PAD>")
    vocab.insert(SOS, "<SOS>")
    vocab.insert(EOS, "<EOS>")
    vocab.insert(UNK, "<UNK>")
    # {word: index of word}
    w2id = {w:i for i,w in enumerate(vocab)}
    #make each sentence into a list of ids
    ids = [[w2id[t] for t in sent] for sent in sents_vals]
    name_ids_dict = {name: id_vals for name, id_vals in zip(sents.keys(), ids)}
    return ids, vocab, w2id, voc_cnt, name_ids_dict


def turnId2W(ids, id2w):
    return ' '.join([id2w.get(id, '<UNK>') for id in ids])

def load_glove_me(loc):
    w2v = {}
    for line in open(loc):
        l = line.split()
        w2v[l[0]] = np.array(l[1:], dtype=np.float32)
    return w2v

def create_emb_mat(w2v, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec)) #initialize empty container
    found=0
    for i, word in enumerate(targ_vocab): 
        try: emb[i] = w2v[word]; found+=1 #if we find it, use the word's vecotrs
        except KeyError: emb[i] = np.random.normal(scale=0.6, size=(dim_vec,)) #else randomeness

    return emb, found

def padAndEOS(ids, maxlen=32):
    ids = list(map(lambda x:  x + [EOS], ids))
    return pad_sequences(ids, maxlen+2, 'int64', "post", "post")
    padded = padAndEOS(ids)

    
mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

scaleCrop = transforms.Compose([transforms.Scale(299), transforms.CenterCrop((299, 299))])

norm = transforms.Normalize(mean=mu,std=std)
mu_rev = torch.FloatTensor(mu).unsqueeze(1).unsqueeze(1)
std_rev = torch.FloatTensor(std).unsqueeze(1).unsqueeze(1)
preproc = transforms.Compose([scaleCrop, transforms.ToTensor(), norm])
tr_trans = transforms.Compose([scaleCrop, transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])

def denormF(x):
    std = torch.FloatTensor(std_rev).expand_as(x)
    mu = torch.FloatTensor(mu_rev).expand_as(x)
    return (x * std ) + mu


denorm = transforms.Lambda(denormF)
revTrans = transforms.Compose([denorm, transforms.ToPILImage()])

def long_t(arr): return Variable(torch.LongTensor(arr)).cuda()


def ImgCapLoader(dl, caps_dict, batch_size=32, shuffle=True, testing=False):
    if shuffle:
        dl.dataset.imgs = [(path, int(clazz)) for path, clazz in np.random.permutation(dl.dataset.imgs)]
    img_names = dl.dataset.imgs
    num_batches = len(img_names) // batch_size
    dl_iter = dl.__iter__()
    for i in range(0, len(img_names), batch_size):
        basenames = [path.basename(name) for name, clz in img_names[i:i+batch_size]]
        if not testing:
            ids = np.vstack(padAndEOS([caps_dict[name] for name in basenames]))
        else:
            ids = np.vstack(padAndEOS([[0.] for name in basenames]))
        imgs, _ = dl_iter.__next__()
        yield imgs, ids
        

def padAndEOS(ids, maxlen=30):
    ids = list(map(lambda x:  x + [EOS], ids))
    return pad_sequences(ids, maxlen+2, 'int64', "post", "post")
    padded = padAndEOS(ids)

    
def getCaps(ids, id2w):
    caps = []
 

    for i in ids:
        caps.append(turnId2W(i, id2w))
        #print()
        #print("\n")
    return caps


def displayCapImg(preds, trues, imgs, id2w, num_img=5):
    for i in range(num_img):
        pred = preds[i, :]; true = trues[i,:]; img = imgs[i, :, :, :]
        
        print('true cap: {}'.format(turnId2W(true, id2w)))
        print('predicted cap: {}'.format(turnId2W(pred, id2w)))
        rev = revTrans(img)
        plt.imshow(rev)
        plt.show()

        
def sanityCheckInputs(imgs,ids, id2w, num_img=5):
    for i in range(num_img):
        img_ids = ids[i,:]; img = imgs[i, :, :, :]
        print('cap: {}'.format(turnId2W(img_ids, id2w)))
        rev = revTrans(img)
        plt.imshow(rev)
        plt.show()

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def make_samp_caps(all_caps, img_dir):
    names = [path.basename(x) for x in glob(img_dir)]
    return [cap for cap in all_caps if cap[0] in names]

def displaySamples(imgs, pred, true, vocab):
    for samp in range(pred.size()[0]):
        plt.imshow(revTrans(imgs[samp, :, :, :].data.cpu()))
        plt.show()
        print('Predicted: {}'.format(vocab.turnId2W(pred.data[samp].cpu().numpy())))
        print('True: {}'.format(vocab.turnId2W(true.data[samp].cpu().numpy())))

def sampleBatch(enc, dec, loader):
    
    for i, (imgs, caps, ls) in enumerate(loader):

        loader.dataset.sanity(imgs, caps, ls)
        imgs = to_var(imgs); caps = to_var(caps) #VOLATILE ON IMGS MAYBE?
        enc.eval(); dec.eval()
        imgs_enc = enc(imgs)
        samples = dec.sample(imgs_enc)
        enc.train(); dec.train()
        break
        
    return imgs, samples, caps