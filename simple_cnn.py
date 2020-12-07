import numpy as np
import torch
from torch import nn

#simple CNN
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernelsize=5):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernelsize, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=16, kernelsize=3)
        self.unit2 = Unit(in_channels=16, out_channels=16, kernelsize=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,2))

        self.unit3 = Unit(in_channels=16, out_channels=32, kernelsize=3)
        self.unit4 = Unit(in_channels=32, out_channels=32, kernelsize=3)

        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,2))

        self.unit5 = Unit(in_channels=32, out_channels=64, kernelsize=3)
        self.unit6 = Unit(in_channels=64, out_channels=64, kernelsize=3)

        self.pool3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2, 2))


        #Conv
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, self.unit4, self.pool2,
                                 self.unit5, self.unit6, self.pool3)

        #FC
        self.fc = nn.Sequential(nn.Linear(28*28*64, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(1024, 128),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(128,4),
                                nn.LogSoftmax(dim=1))

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 28*28*64)
        output = self.fc(output)
        return output