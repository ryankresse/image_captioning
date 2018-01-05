import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import pdb
from PIL import Image
import matplotlib.pyplot as plt
from build_vocab import Vocabulary

from utils import preproc, revTrans

class FlikrDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, name_caps, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.name_caps = name_caps
        self.names = list(name_caps.keys())
        self.caps = list(name_caps.values())
        #self.num_caps_per_img = len(self.caps[0])
        self.cap_ix = np.random.randint(len(self.caps[0]))
        self.vocab = vocab
        self.transform = transform
    
    
    
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        vocab = self.vocab
        #cap_ix = np.random.randint(low=0, high=self.num_caps_per_img)
        #print(cap_ix)
        caption = self.caps[index][self.cap_ix]
        path = self.names[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.names)
    
    def sanity(self, imgs, caps, ls):
        ix = 0 # np.random.randint(0, imgs.size()[0])
        print(self.vocab.turnId2W(caps.numpy()[ix, :]))
        print('caption length: {}'.format(ls[ix]))
        print('caption computed length: {}'.format(np.sum(caps.numpy()[ix, :] != 0)))
        plt.imshow(revTrans(imgs[ix, :, :, :]))
        plt.show()

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_loader(root, name_caps, vocab, transform=preproc, batch_size=4, shuffle=True, num_workers=-1, drop_last=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # flikr caption dataset
    flikr = FlikrDataset(root=root,
                       name_caps=name_caps,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for flikr dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=flikr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=drop_last)
    return data_loader