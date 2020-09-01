# train_mnist_csv.py

from __future__ import print_function, division

import os, sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset_csv import myMnistDataset
from configs.config_train import get_cfg_defaults

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


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


def main():
    args = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print('\n', cfg)

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3801,))
    ])

    # Create datasets
    ProcPath = cfg.DATA.PROCESSED_PATH
    trainSet = myMnistDataset(path=ProcPath, train=True,  transform=transform)
    testSet  = myMnistDataset(path=ProcPath, train=False, transform=transform)

    # Create DataLoader
    BatchSize = cfg.TRAIN.BATCH_SIZE
    trainLoader = DataLoader(dataset=trainSet, batch_size=BatchSize, shuffle=True)
    testLoader  = DataLoader(dataset=testSet, batch_size=BatchSize, shuffle=False) 

    print('\nimage size: ', trainSet.train_data.size())
    print('\nlabel size: ', trainSet.train_labels.size())

    # GPU
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'
    print('\nGPU State:', device)

    net = Net().to(device)
    print('\n', net)

    # Parameters
    lr = cfg.TRAIN.LEARNING_RATE
    num_mom = cfg.TRAIN.MOMENTUM
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=num_mom)

    # Train
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    print('\nTraining start...')
    for epoch in range(num_epochs):
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
                print('[%d/%d, %d/%d] loss: %.3f' % (
                    epoch+1, num_epochs, i+1, len(trainLoader), running_loss/2000))

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

if __name__=='__main__':
    main()

