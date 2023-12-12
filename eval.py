import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
from RL_model import BreakoutQNet
from linear_model import LinearBreakoutQNet
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
from train import transform
import os
import shutil
import time

import warnings
warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make('ALE/Breakout-v5', obs_type="ram", render_mode='rgb_array', frameskip=4)
    env = FrameStack(env, 4)
    env = RecordVideo(env, video_folder='data\\final_video', episode_trigger=lambda t: True, disable_logger=True)
    n_actions = env.action_space.n
    state, info = env.reset()
    state_shape = state.shape[0] * state.shape[1]
    policy_net = LinearBreakoutQNet(state_shape, n_actions).to(device)
    policy_net.load_state_dict(torch.load('data\\model_best.pth'))
    state = torch.tensor(state, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
    total_reward = 0.0
    max_reward = 33.0
    for ep in range(500):
        running_reward = 0.0
        done = False
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
        with torch.no_grad():
            while not done:
                action = policy_net(state).max(1)[1].view(1, 1)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                running_reward += reward
                total_reward += reward
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
                state = next_state
                env.render()
        env.close_video_recorder()
        print(f'Reward: {running_reward}')
        # time.sleep(1)
        if running_reward > max_reward:
            max_reward = running_reward
            if not os.path.exists(f'data\\saved_final_videos\\final_video_{int(running_reward)}r'):
                os.mkdir(f'data\\saved_final_videos\\final_video_{int(running_reward)}r')
            shutil.copy(f'data\\final_video\\rl-video-episode-{ep}.meta.json', f'data\\saved_final_videos\\final_video_{int(running_reward)}r\\rl-video-episode-{ep}.meta.json')
            shutil.copy(f'data\\final_video\\rl-video-episode-{ep}.mp4', f'data\\saved_final_videos\\final_video_{int(running_reward)}r\\rl-video-episode-{ep}.mp4')
    print(f'Average Reward: {total_reward/100}')
    return

if __name__ == '__main__':
    main()