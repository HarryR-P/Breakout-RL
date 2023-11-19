import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
from RL_model import BreakoutQNet
from gymnasium.wrappers import FrameStack, RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2

import warnings
warnings.simplefilter("ignore")

num_episodes = 10000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(110,84), antialias=True),
])

steps_done = 0

def main():
    env = gym.make('ALE/Breakout-v5', obs_type="grayscale", render_mode='rgb_array')
    env = FrameStack(env, 4)
    env = RecordVideo(env, 'data//progress_videos', episode_trigger=lambda t: t % 1000 == 0, disable_logger=True)

    n_actions = env.action_space.n
    state, info = env.reset()
    state_shape = transform(state).shape

    policy_net = BreakoutQNet(state_shape, n_actions).to(device)
    target_net = BreakoutQNet(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    loss_func = nn.SmoothL1Loss()
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    reward_per_episode = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = transform(state).unsqueeze(0).to(device)
        running_reward = 0.0
        done = False
        while not done:
            action = select_action(state, env, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            running_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = transform(observation).unsqueeze(0).to(device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer, loss_func)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        reward_per_episode.append(running_reward)
        print(f'Episode {i_episode+1}/{num_episodes}, running reward: {running_reward}')

    torch.save(policy_net.state_dict(),'data\\model.pth')
    print('Complete')
    plt.plot(reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig('data\\reward_per_ep.png')
    plt.show()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

def select_action(state, env, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

def optimize_model(memory, policy_net, target_net, optimizer, loss_func):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def transform(state):
    frame_list = []
    for frame in state:
        t_frame = transforms(frame)
        frame_list.append(t_frame)
    return torch.cat(frame_list)

    


if __name__ == '__main__':
    main()