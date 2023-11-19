import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
from RL_model import BreakoutQNet
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
from train import transform

import warnings
warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make('ALE/Breakout-v5', obs_type="grayscale", render_mode='rgb_array')
    env = FrameStack(env, 4)
    env = RecordVideo(env, video_folder='data\\final_video')
    n_actions = env.action_space.n
    state, info = env.reset()
    state_shape = transform(state).shape
    policy_net = BreakoutQNet(state_shape, n_actions).to(device)
    policy_net.load_state_dict(torch.load('data\\model.pth'))
    state = transform(state).unsqueeze(0).to(device)
    running_reward = 0.0
    done = False
    with torch.no_grad():
        while not done:
            action = policy_net(state).max(1)[1].view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            running_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = transform(observation).unsqueeze(0).to(device)
            state = next_state
    return

if __name__ == '__main__':
    main()