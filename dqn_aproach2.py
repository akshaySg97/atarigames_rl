import gymnasium as gym
import numpy as np
import random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import ale_py
import time
gym.register_envs(ale_py)


class MemoryReplay:
    def __init__(self, capacity):
        self.buff = deque(maxlen=capacity)
    
    def makechoice(self, size):
        return rnd.sample(self.buff, size)
    
    def add(self,experience):
        self.buff.append(experience)

    def __len__(self):
        return len(self.buff)


class DQNetwork(nn.Module):
    def __init__(self, input, num_acts):
        super(DQNetwork, self).__init__()
        self.cnnlayers = nn.Sequential(
            nn.Conv2d(input[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Final spatial dimensions: (128, 7, 7)
        self.fullyConn = nn.Sequential(
            nn.Linear(128 * 22 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_acts)
        )
    
    def forward(self, state):
        state = self.cnnlayers(state)
        print("Shape after convolutional layers:", state.shape)
        state = state.view(state.size(0), -1)
        return self.fullyConn(state)

    



# def convert_img_frames(obs):
#     converted_frame = lambda obs: np.expand_dims(np.mean(obs, axis=2) / 255.0, axis=0)
#     return converted_frame

def convert_img_frames(obs):
    # Convert to grayscale by averaging across the color channels
    obs = np.mean(obs, axis=2)
    # Normalize pixel values to the range [0, 1]
    obs = obs / 255.0
    # Add a channel dimension
    obs = np.expand_dims(obs, axis=0)
    return obs



def train(evn_name, gamma, epsilon_strt, epsilon_end, ep_decay, episodes, learning_rt_alpha,buff_size, device, tgt_updt_freq_C, batch_size):

    env = gym.make(evn_name)
    print(env)
    # 1- Number of channels (grey scale image), 84 - height of 84 pixels
    # 84 width of image 84 pixels
    ip_shape = (1,84,84)
    num_acts = env.action_space.n

    dq_network = DQNetwork(input=ip_shape, num_acts=num_acts).to(device)
    target_network = DQNetwork(input=ip_shape, num_acts=num_acts).to(device)
    # load_state_dict - from pytorch load parameters into a model from state_dict
    # i.e copy dqnetwork into target network
    # state_dict - from pytorch contains all parameters of  a model (weights and biases)
    target_network.load_state_dict(dq_network.state_dict())

    optimize = optim.Adam(dq_network.parameters(), lr=learning_rt_alpha)
    error_loss  = nn.SmoothL1Loss()

    experience_buffer = MemoryReplay(buff_size)

    epsilon = epsilon_strt
    ep_decay_rate = (epsilon_strt - epsilon_end) / ep_decay

    for episode in range(episodes):
        obs = convert_img_frames(env.reset()[0])
        done = False
        total_rwd = 0

        while not done:
            if rnd.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action = torch.argmax(dq_network(obs_tensor)).item()
                
            
            next_obs, rwrd, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = convert_img_frames(next_obs)
            experience_buffer.add((obs, action, rwrd, next_obs, done))
            obs = next_obs
            total_rwd += rwrd

            if len(experience_buffer) >= batch_size:
                past_transitions = experience_buffer.makechoice(batch_size)
                buff_obs, buff_actions, buff_rwds, buff_next_obs, buff_done = zip(*past_transitions)
                buff_obs = torch.tensor(np.array(buff_obs), dtype=torch.float32).to(device)
                buff_actions = torch.tensor(buff_actions, dtype=torch.int64).unsqueeze(1).to(device)
                buff_rwds = torch.tensor(buff_rwds, dtype=torch.float32).unsqueeze(1).to(device)
                buff_next_obs = torch.tensor(np.array(buff_next_obs), dtype=torch.float32).to(device)
                buff_done = torch.tensor(buff_done, dtype=torch.float32).unsqueeze(1).to(device)

                q_vals = dq_network(buff_obs).gather(1, buff_actions)
                with torch.no_grad():
                    next_q_vals = target_network(buff_next_obs).max(1)[0].unsqueeze(1)
                    target_q_vals = buff_rwds + gamma * next_q_vals * (1-buff_done)

                loss = error_loss(q_vals, target_q_vals)
                optimize.zero_grad()
                loss.backward()
                optimize.step()

            if episode % tgt_updt_freq_C == 0:
                target_network.load_state_dict(dq_network.state_dict())

        epsilon = max(epsilon_end, epsilon - ep_decay_rate)
        print(f"Episdoe {episode + 1}: Total reward = {total_rwd}")
    env.close()


    
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
evn_name = "ALE/IceHockey-v5"
episodes = 50
gamma = 0.99
epsilon_strt = 1.0
epsilon_end = 0.1
ep_decay = 10000
learning_rt_alpha = 1e-4
buff_size = 100000
batch_size = 32
tgt_updt_freq_C = 1000

start_time = time.time()


train(evn_name, gamma, epsilon_strt, epsilon_end, ep_decay, episodes, learning_rt_alpha,buff_size, device, tgt_updt_freq_C, batch_size)

end_time = time.time()
print(f"Training time for {episodes} episodes: {end_time - start_time:.2f} seconds")
