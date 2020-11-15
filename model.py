import torch
from torch import nn


class LinearModel(torch.nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(9*9, 1)

    def forward(self, x):
        out = torch.flatten(x)
        return self.linear(out)


class Model2(torch.nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.linear1 = torch.nn.Linear(9*9, 256)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128,16)
        self.linear4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        out = torch.flatten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out
