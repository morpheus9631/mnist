# MnistDatasetCsv

import os 
import numpy as np
import sys
from PIL import Image

import torch
from torch.utils.data import Dataset

class myMnistDataset(Dataset):
    
    training_file = 'mnist_train.csv'
    test_file = 'mnist_test.csv'
    
    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.csvPath = path
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        data = np.loadtxt(os.path.join(self.csvPath, data_file), delimiter=',', dtype=np.float32)
        
        self.len = data.shape[0]
        imgs = torch.from_numpy(data[:, 1:])
        self.images = imgs.type(torch.uint8).view(self.len, 28, 28)
        
        lbls = torch.from_numpy(data[:, [0]])
        self.labels = lbls.view(self.len)

    def __getitem__(self, index):
        img, lbl = self.images[index], int(self.labels[index])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
            
        return img, lbl

    def __len__(self):
        return self.len
    
    @property
    def train_labels(self):
        return self.labels

    @property
    def test_labels(self):
        return self.labels

    @property
    def train_data(self):
        return self.images

    @property
    def test_data(self):
        return self.images
