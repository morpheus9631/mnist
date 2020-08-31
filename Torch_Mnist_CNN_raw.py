# train_mnist_v3.py

from __future__ import print_function, division  

import os, sys
import argparse
import codecs
# import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from configs.config_train import  get_cfg_defaults
from datasets.dataset_raw import myMnistDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


class Net(nn.Module):
    def __init__(self, insize, size1, size2, outsize):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=insize, out_features=size1),
            nn.ReLU(),
            nn.Linear(in_features=size1, out_features=size2),
            nn.ReLU(),
            nn.Linear(in_features=size2, out_features=outsize),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


def train(net, lr, momentum, epochs, trainLoader, dev, bsize):
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Train
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(trainLoader):
            images, labels = data[0].to(dev), data[1].to(dev)
            images = images.view(images.shape[0], -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Foward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i+1) % bsize == 0 or i+1 == len(trainLoader):
                print('[%d/%d, %d/%d] loss: %.3f' % (
                    epoch+1, epochs, i+1, len(trainLoader), running_loss/2000))
    return


def test(validLoader, device, net):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validLoader:
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
        for data in validLoader:
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
    print()
    for i in range(10):
        print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))
    return    


def main():
    args = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg); print() 

    # Prepare dataset and dataloader
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3801,))
    ])

    # Create datasets
    RawPath = cfg.DATA.RAW_PATH
    trainSet = myMnistDataset(path=RawPath, train=True, transform=transform)
    validSet = myMnistDataset(path=RawPath, train=False, transform=transform)

    # Create dataloaders
    BatchSize = cfg.TRAIN.BATCH_SIZE
    trainLoader = DataLoader(dataset=trainSet, batch_size=BatchSize, shuffle=True)
    validLoader = DataLoader(dataset=validSet, batch_size=BatchSize, shuffle=False)        

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print('\nGPU State:', device)

    input_size   = cfg.TRAIN.INPUT_SIZE
    hidden_size1 = cfg.TRAIN.HIDDEN_SIZE1
    hidden_size2 = cfg.TRAIN.HIDDEN_SIZE2
    num_classes  = cfg.TRAIN.NUM_CLASSES
    
    net = Net(input_size, hidden_size1, hidden_size2, num_classes).to(device)
    print(); print(net)

    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    epochs = cfg.TRAIN.EPOCHES

    # For train 
    train(net, lr, momentum, epochs, trainLoader, device, BatchSize)

    # For test
    test(validLoader, device, net)


if __name__=='__main__':
    main()



