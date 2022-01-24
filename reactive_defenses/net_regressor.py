import torch.nn as nn
import torch.nn.functional as F


class NetRegressor(nn.Module):

    def __init__(self):
        super(NetRegressor, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
