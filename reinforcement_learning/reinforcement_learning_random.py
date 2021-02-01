import numpy as np
import random
import matplotlib.pyplot as plt

class cart_pole():
    def __init__(self, sim_size = 100):
        self.sim_size = sim_size
        self.g = 9.8
        self.mc = 1
        self.mp = 0.1
        self.l = 0.5
        self.uc = 5e-4
        self.up = 2e-6
        self.force = 10

        self.beta = 0.1

    def reset(self, plot_on = False):
        # x: 0, theta: 1, x': 2, theta':3
        #Start randomly 
        self.state = np.zeros((self.sim_size, 4), dtype = "float32")
        self.plot_on = plot_on
    
        if(plot_on is True):
            plt.close()
            self.fig, self.axs = plt.subplots(4, sharex=True)
            self.t = np.arange(self.sim_size)

        self.iteration_index = 0 

        return self.state[0, :]

    def step(self, action):
        F = np.where(action>= 0, self.force, -self.force)
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
            reward = 1
        else:
            reward = 0
        
        return self.state[self.iteration_index, :], reward, sim_end

    def check_end(self):
        if self.iteration_index == self.sim_size:
            return True
        
        if abs(self.state[self.iteration_index, 0]) > 2.4 or abs(self.state[self.iteration_index, 1]) > 12:
            return True

        return False

    def plot_data(self):
        for i in range(4):
            self.axs[i].cla()
            self.axs[i].plot(self.t[:self.iteration_index], self.state[:self.iteration_index, i])
        
        plt.pause(0.1)
    

sim_size = 100
max_epoch = 10
sim = cart_pole(sim_size)

for epoch in range(max_epoch):
    x = sim.reset(plot_on=True)
    while True:
        action = np.random.randn(1)
        new_state, reward, end = sim.step(action)
        print(new_state, reward, end)

        if end is True:
            break
    
