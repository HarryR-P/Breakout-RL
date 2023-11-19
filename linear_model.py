import torch
from torch import nn

class LinearBreakoutQNet(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()

        self.fc1 = nn.Linear(n_features, 1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.act2 = nn.ReLU()
        self.final_fc = nn.Linear(256, n_actions)

    def forward(self, state):
        h = self.act1(self.fc1(state))
        h = self.act2(self.fc2(h))
        return self.final_fc(h)