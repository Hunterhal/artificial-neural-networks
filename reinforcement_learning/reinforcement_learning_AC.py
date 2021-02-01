import numpy as np
import random
import matplotlib.pyplot as plt

class cart_pole():
    def __init__(self, sim_size = 100, plot_on = False):
        self.sim_size = sim_size
        self.plot_on = plot_on
        self.epoch = -1

        self.g = 9.8
        self.mc = 1
        self.mp = 0.1
        self.l = 0.5
        self.uc = 5e-4
        self.up = 2e-6
        self.force = 10

        self.beta = 0.02

        if(plot_on is True):
            plt.close()
            self.fig, self.axs = plt.subplots(5, sharex=True)
            self.t = np.arange(self.sim_size)

    def reset(self):
        # x: 0, theta: 1, x': 2, theta':3
        #Start randomly 
        self.state = np.zeros((self.sim_size, 4), dtype = "float32")
        self.state[0, 0] = np.random.normal(0, 0.5, 1)
        self.state[0, 1] = np.random.normal(0, 0.1, 1)
        self.epoch += 1
        if(self.plot_on is True):
            plt.close()
            self.fig, self.axs = plt.subplots(5, sharex=True,figsize=(16, 9))
            self.fig
            self.t = np.arange(self.sim_size)
            plt.suptitle("Epoch: %d"%(self.epoch))
            for i in range(4):
                self.axs[i].cla()

        self.iteration_index = 0 

        return self.state[0, :]

    def step(self, action):
        F = np.where(action> 0, self.force, -self.force)
        x = self.state[self.iteration_index, 0]
        theta = self.state[self.iteration_index, 1]
        x_dot = self.state[self.iteration_index, 2]
        theta_dot = self.state[self.iteration_index, 3]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        m_total = self.mp + self.mc
        theta_acc = (self.g * sin_theta + cos_theta * ((-F - self.mp * self.l * theta_dot**2 * sin_theta)) 
                    / self.l * (4/3 - ((self.mp*cos_theta**2)/m_total)))
        
        x_acc = ((F + self.mp * self.l * (theta_dot**2 * sin_theta - theta_acc * cos_theta)) 
                / m_total)

        self.iteration_index += 1

        self.state[self.iteration_index, 0] = x + self.beta * x_dot
        self.state[self.iteration_index, 2] = x_dot + self.beta * x_acc
        self.state[self.iteration_index, 1] = theta + self.beta * theta_dot
        self.state[self.iteration_index, 3] = theta_dot + self.beta * theta_acc

        if self.plot_on is True:
            self.plot_data()

        sim_end = self.check_end()

        if sim_end is False: 
            reward = 0
        else:
            reward = -1
        
        return self.state[self.iteration_index, :], reward, sim_end

    def check_end(self):
        if self.iteration_index == self.sim_size - 1:
            return True
        
        if abs(self.state[self.iteration_index, 0]) > 2.4 or abs(self.state[self.iteration_index, 1] ) > 12 * np.pi / 180:
            return True

        return False

    def plot_data(self):
        self.axs[0].cla()
        self.axs[0].plot(self.t[:self.iteration_index], self.state[:self.iteration_index, 0])
        self.axs[0].set_title('Cart Position')
        self.axs[1].cla()
        self.axs[1].plot(self.t[:self.iteration_index], self.state[:self.iteration_index, 2])
        self.axs[1].set_title('Cart Velocity')
        self.axs[2].cla()
        self.axs[2].plot(self.t[:self.iteration_index], self.state[:self.iteration_index, 1] * 180 / np.pi)
        self.axs[2].set_title('Pole Angle')
        self.axs[3].cla()
        self.axs[3].plot(self.t[:self.iteration_index], self.state[:self.iteration_index, 3] * 180 / np.pi)
        self.axs[3].set_title('Pole Velocity')
        
        plt.pause(0.02)

    def plot_reward(self, real, tilda):
        t = np.arange(len(tilda))
        self.axs[4].cla()
        self.axs[4].plot(t, real, 'r')
        self.axs[4].plot(t, tilda, 'b')
        self.axs[4].set_title('Red Real Reward, Blue Improved Reward')


sim_size = 5000
max_epoch = 100
sim = cart_pole(sim_size, plot_on=True)

ase_w = np.random.randn(4)
ace_w = np.random.randn(4)

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

    while True:

        y = np.tanh( np.dot(ase_w, x) + np.random.normal(0, noise_sigma, 1))

        action = np.random.randn(1)
        x, real_reward, end = sim.step(y)

        x_tilda = lambd * x_tilda + (1 - lambd) * x
        x_tilda_buffer.append(x_tilda)
        
        p = np.dot(ace_w, x)
        p = -1/(1 + np.exp(-p))
        
        ace_w = ace_w + beta * (reward + gamma * p - p_buffer[-1]) * x_tilda
        reward = real_reward + gamma * p - p_buffer[-1]

        ase_w = ase_w + alpha * reward * e
        e_buffer.append(delta * e + (1-delta) * y * x)
        e = e_buffer[-1]


        reward_buffer.append(reward)
        real_reward_buffer.append(real_reward)
        p_buffer.append(p)
        sim.plot_reward(real_reward_buffer, reward_buffer)
        print(real_reward, reward, p_buffer[-2], p_buffer[-1])

        if end is True:
            break
    
