import torch
from torch import nn


"""
Please use this mask to filter out the duplicate cells first
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mask = torch.zeros(9, 9).type(torch.BoolTensor)
for i in range(9):
    mask[i][i:] = True
FEATURE_SIZE = 45
mask = mask.to(DEVICE)


class LinearModel(torch.nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(FEATURE_SIZE, 1)

    def forward(self, x):
        out = x[mask]
        return self.linear(out)


class Model2(torch.nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.linear1 = torch.nn.Linear(FEATURE_SIZE, 256)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 16)
        self.linear4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        out = x[mask]
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out

class Model3(torch.nn.Module):

    def __init__(self):
        super(Model3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,3),
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,3),
            )
        
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x):
        out = x.unsqueeze(0)
        out = out.unsqueeze(0)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.squeeze(3)
        out = out.squeeze(2)
        out = self.linear(out)
        out = out.squeeze(1)
        return out


class FiveLayerNN(torch.nn.Module):

    def __init__(self):
        super(FiveLayerNN, self).__init__()
        self.linear1 = torch.nn.Linear(FEATURE_SIZE, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 16)
        self.linear5 = torch.nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x[mask]
        out = self.linear1(out)
        out = self.sig(out)
        out = self.linear2(out)
        out = self.sig(out)
        out = self.linear3(out)
        out = self.sig(out)
        out = self.linear4(out)
        out = self.sig(out)
        out = self.linear5(out)
        return out


class ThreeLayerNN(torch.nn.Module):

    def __init__(self):
        super(ThreeLayerNN, self).__init__()
        self.linear1 = torch.nn.Linear(FEATURE_SIZE, 128)
        self.linear2 = torch.nn.Linear(128, 16)
        self.linear3 = torch.nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x[mask]
        out = self.linear1(out)
        out = self.sig(out)
        out = self.linear2(out)
        out = self.sig(out)
        out = self.linear3(out)
        return out


class TwoLayerNN(torch.nn.Module):

    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.linear1 = torch.nn.Linear(FEATURE_SIZE, 32)
        self.linear2 = torch.nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x[mask]
        out = self.linear1(out)
        out = self.sig(out)
        out = self.linear2(out)
        return out

