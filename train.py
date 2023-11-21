import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
from RL_model import BreakoutQNet
from linear_model import LinearBreakoutQNet
from gymnasium.wrappers import FrameStack, RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.functional import crop

import warnings
warnings.simplefilter("ignore")

num_episodes = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.3
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(110,84)),
])

steps_done = 0

def main():
    env = gym.make('ALE/Breakout-v5', 
                   obs_type="ram", 
                   render_mode='rgb_array',
                   frameskip=4)
    env = FrameStack(env, 4)
    env = RecordVideo(env, 'data//progress_videos', episode_trigger=lambda t: t % 100 == 0, disable_logger=True)

    n_actions = env.action_space.n
    state, info = env.reset()
    # state_shape = transform(state).shape
    state_shape = state.shape[0] * state.shape[1]

    # policy_net = BreakoutQNet(state_shape, n_actions).to(device)
    policy_net = LinearBreakoutQNet(state_shape, n_actions).to(device)
    target_net = LinearBreakoutQNet(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    total_reward = 0.0

    reward_per_episode = []
    mean_reward = []
    score = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        # state = transform(state).unsqueeze(0).to(device)
        state = torch.tensor(state, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
        running_reward = 0.0
        done = False
        while not done:
            action = select_action(state, env, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            running_reward += reward
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).view(-1).unsqueeze(0)

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
        mean_reward.append(total_reward/(i_episode+1))
        if running_reward >= score:
            score = running_reward
            torch.save(policy_net.state_dict(),'data\\model_best.pth')
        print(f'Episode {i_episode+1}/{num_episodes}\n\trunning reward: {running_reward}, average reward: {total_reward/(i_episode+1):.2f}, top score {score}')
    
    torch.save(policy_net.state_dict(),'data\\model_final.pth')
    print('Complete')
    plt.plot(reward_per_episode)
    plt.plot(mean_reward)
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
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def transform(state):
    frame_list = []
    for frame in state:
        t_frame = crop(transforms(frame),20,0,84,84)
        frame_list.append(t_frame)
    return torch.cat(frame_list)

    


if __name__ == '__main__':
    main()