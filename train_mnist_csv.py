# train_mnist_csv.py

from __future__ import print_function

import os, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class myMnistDataset(Dataset):
    
    csvPath = 'D:\\GitWork\\mnist\\processed\\'
    
    training_file = 'mnist_train.csv'
    test_file = 'mnist_test.csv'
    
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    
    def __init__(self, train=True, transform=None, target_transform=None):
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


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)

# Transform
BatchSize = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3801,))
])

# Create datasets
trainSet = myMnistDataset(train=True, transform=transform)
testSet = myMnistDataset(train=False, transform=transform)

# Create DataLoader
trainLoader = DataLoader(dataset=trainSet, batch_size=BatchSize, shuffle=True)
testLoader = DataLoader(dataset=testSet, batch_size=BatchSize, shuffle=False)   

print(trainSet.train_data.size())
print(trainSet.train_labels.size())

# print(type(trainSet.train_data[0]))
# print(type(trainSet.train_labels[0]))

# print(trainSet.train_data[0])
# print(trainSet.train_labels[0])

# GPU
use_gpu = torch.cuda.is_available()
device = 'cuda:0' if use_gpu else 'cpu'
print('GPU State:', device)

net = Net().to(device)
print(net)

# Parameters
epochs = 3
lr = 0.002
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

# Train
for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i+1) % BatchSize == 0 or i+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, i+1, len(trainLoader), running_loss/2000))

print('Training Finished.')

# Test
correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('\nAccuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            # print(class_correct)
            # print(class_total)

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))



