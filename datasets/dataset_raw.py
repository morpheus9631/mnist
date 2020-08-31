# MnistDatasetRaw

import os, sys
import codecs
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).copy()
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).copy()
        return torch.from_numpy(parsed).view(length).long()


class myMnistDataset(Dataset):
    
    Resources = {
        'train': [ "train-images-idx3-ubyte", "train-labels-idx1-ubyte" ],
        'valid': [ "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"  ]
    }
    
    def __init__(self, path, train=True, transform=None, debug=True):
        self.datapath = path
        self.train = train
        self.transform = transform
        
        datafiles = self.Resources['train'] if self.train else self.Resources['valid']
        self.imgs = read_image_file(os.path.join(self.datapath, datafiles[0]))
        self.lbls = read_label_file(os.path.join(self.datapath, datafiles[1]))
        self.len = len(self.imgs)

        if debug: 
            phase_name = 'train' if self.train else 'valid'
            print('{} set:\n  images: {}\n  labels: {}'.format(
                phase_name, self.imgs.shape, self.lbls.shape)
            )
  
    def __getitem__(self, index):
        img, lbl = self.imgs[index], int(self.lbls[index])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, lbl
                          
    def __len__(self):
        return self.len
    
    @property
    def images(self):
        return self.imgs

    @property
    def labels(self):
        return self.lbls
