import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import namedtuple, deque
from itertools import count
from RL_model import BreakoutQNet
from gymnasium.wrappers import FrameStack

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.functional import crop

warnings.simplefilter("ignore")

def main():
    # transforms = v2.Compose([
    #     v2.ToTensor(),
    #     v2.Resize(size=(110,84)),
    # ])
    env = gym.make('ALE/Breakout-v5',
                   obs_type="ram")
    env = FrameStack(env, 4)
    state, info = env.reset()

    print(torch.tensor(state).view(-1).shape)
    return

if __name__ == '__main__':
    main()