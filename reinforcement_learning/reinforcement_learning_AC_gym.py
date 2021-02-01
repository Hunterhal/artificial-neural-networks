import numpy as np
import random
import gym
import time
import matplotlib.pyplot as plt

sim_size = 1000
max_epoch = 100

sim = gym.make('CartPole-v0')
ase_w = np.random.randn(4)
ace_w = np.random.randn(4) * 0.01


gamma = 0.95
beta = 0.5
alpha = 1000
delta = 0.9
lambd = 0.8
noise_sigma = 0.01

for epoch in range(max_epoch):
    x = sim.reset()

    reward = 0
    e = np.zeros(4)
    p = 0
    x_tilda = 0
    reward_buffer = []
    real_reward_buffer = []
    reward_buffer.append(reward)
    real_reward_buffer.append(reward)
    e_buffer = []
    e_buffer.append(e)
    p_buffer = []
    p_buffer.append(p)
    x_tilda_buffer = []
    x_tilda_buffer.append(x_tilda)

    index = 0
    state = np.zeros((1, 4))
    while True:
        sim.render()

        y = np.tanh( np.dot(ase_w, x) + np.random.normal(0, noise_sigma, 1))
        x, real_reward, end, _ = sim.step(int(np.where(y>0, 1, 0)))
        real_reward += -1
        x_tilda = lambd * x_tilda + (1 - lambd) * x
        x_tilda_buffer.append(x_tilda)
        
        p = np.dot(ace_w, x)
        p = -1/(1 + np.exp(-p))
        
        reward = real_reward + gamma * p - p_buffer[-1]

        ase_w = ase_w + alpha * reward * e
        ace_w = ace_w + beta * (reward + gamma * p - p_buffer[-1]) * x_tilda
        
        e_buffer.append(delta * e + (1-delta) * y * x)
        e = e_buffer[-1]

        reward_buffer.append(reward)
        p_buffer.append(p)
        state = np.concatenate((state, x.reshape(1, -1)), axis=0)
        print(real_reward, reward, p_buffer[-2], p_buffer[-1])

        if end is True:
            break
    
plt.show()