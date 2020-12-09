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
