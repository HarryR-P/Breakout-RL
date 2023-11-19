import torch
from torch import nn

class BreakoutQNet(nn.Module):

    def __init__(self, state_shape, n_actions):
        super().__init__()

        in_c, in_h, in_w = state_shape

        cov_out_size = lambda in_size, kernal, stride: int(((in_size - kernal) / stride) + 1)

        fc_height = cov_out_size(cov_out_size(in_h, 8, 4), 4, 2)
        fc_width = cov_out_size(cov_out_size(in_w, 8, 4), 4, 2)

        self.conv1 = nn.Conv2d(in_c, out_channels=16, kernel_size=8, stride=4)
        self.activ1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.activ2 = nn.ReLU()
        #self.drop1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=fc_height*fc_width*32, out_features=256)
        self.activ3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.activ4 = nn.ReLU()
        #self.drop2 = nn.Dropout(0.2)
        self.final_fc = nn.Linear(in_features=64, out_features=n_actions)

    def forward(self, state):
        h = self.activ1(self.conv1(state))
        h = self.activ2(self.conv2(h))
        h = self.flatten(h)
        h = self.activ3(self.fc1(h))
        h = self.activ4(self.fc2(h))
        return self.final_fc(h)