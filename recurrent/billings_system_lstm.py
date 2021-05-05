import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_dataset(sample_size = 100, initial = 70):
    y = np.zeros(sample_size)
    noise = np.random.normal(0, 0.01, sample_size)

    for i in range(2, sample_size):
        y[i] = ((0.8 + 0.5 * np.exp(-1 * np.square(y[i-1])))*y[i-1] - 
                (0.3 + 0.9 * np.exp(-1 * np.square(y[i-1])))*y[i-2] + 
                (0.1 * np.sin(np.pi * y[i-1])) + 
                noise[i])

    x_train = np.array(noise[initial:], dtype=np.float32).reshape(-1, 1)
    y_train = np.array(y[initial:], dtype=np.float32).reshape(-1, 1)

    dataset_train_torch = TensorDataset(torch.tensor(x_train), torch.tensor(y_train)) # create your datset
    
    train_loader = DataLoader(dataset_train_torch, shuffle=False, batch_size=1) # create your dataloader

    return y, train_loader

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class Out_Scale(nn.Module):
    def forward(self, input):
        return input * 2

class Net(nn.Module):
    def __init__(self, context_size):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(1, context_size, bias=False)
        self.context_layer = nn.Linear(context_size, context_size, bias=False)

        self.out_layer = nn.Linear(context_size, 1, bias=False)

    def forward(self, x, c_pre, h_pre):
        x_1 = F.tanh( self.input_layer(x) )
        x_2 = F.tanh( self.context_layer(c_pre) )
        comb = x_2 + x_1

        

        out = self.out_layer(x)

        return out, x

context_size = 30
network = Net(context_size)

print(network)
max_epoch = 100

sample_size = int(300)
initial = int(0)
train_test = 0.9
test_index = int((sample_size - initial) * train_test)

criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=1e-2)
y_pred = np.zeros(sample_size - initial, dtype=np.float32)
t = np.arange(sample_size)
y, train_loader = create_dataset(sample_size = sample_size, initial = initial)
plt.plot(t[initial:], y[initial:])
plt.show()
out = False

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax_train_state = fig.add_subplot(gs[0, 0])
ax_test_state = fig.add_subplot(gs[1, 0])
ax_main = fig.add_subplot(gs[:, 1:])

for test_number in range(100):
    network.apply(init_weights)
    loss_buffer = np.zeros(max_epoch)
    y, train_loader = create_dataset(sample_size = sample_size, initial = initial)
    c = torch.zeros(1, context_size)
    train_loss_buffer = []
    train_loss_buffer.append(0)
    for epoch in range(max_epoch):
        train_loss = 0
        test_loss = 0
        noise = np.random.normal(0, 0.01, sample_size).reshape(-1, 1).astype(np.float32)
        for i_batch, data in enumerate(train_loader):
            features, labels = data[0], data[1]
            predict, state = network(torch.tensor(noise[i_batch]), c)
            y_pred[i_batch] = predict.clone().detach()
            if i_batch < (test_index):
                optimizer.zero_grad()
                loss = criterion(predict, labels)
                train_loss += loss.data
                loss.backward()
                optimizer.step()
            
            c = torch.tensor( state )

        ax_main.clear()
        ax_train_state.clear()
        ax_test_state.clear()

        ax_main.plot(t[initial:], y[initial:])
        ax_main.plot(t[initial:initial+test_index], y_pred[:test_index], c='g')
        ax_main.plot(t[initial+test_index:], y_pred[test_index:], c='r')
        ax_main.grid()
        ax_main.set_title("Test :%d, Epoch: %d, Training loss: %.7f"%(test_number, epoch, (train_loss/test_index)))

        ax_train_state.plot(y[:test_index-1], y[1:test_index], c='b')
        ax_train_state.plot(y_pred[:test_index-1], y_pred[1:test_index], c='g')
        ax_train_state.grid()
        ax_train_state.set_title("Train State")

        ax_test_state.plot(y[test_index:-1], y[test_index+1:], c='b')
        ax_test_state.plot(y_pred[test_index:-1], y_pred[test_index+1:], c='r')
        ax_test_state.grid()
        ax_test_state.set_title("Test State")

        plt.pause(0.1)
        print("Test :%d, Epoch: %d, Training loss: %.7f"%(test_number, epoch, (train_loss/test_index)))
        train_loss_buffer.append(train_loss/test_index)
        if(train_loss_buffer[-1] < 0.04):
            out = True
            break
        elif( (train_loss_buffer[-1] > 0.2) and (len(train_loss_buffer) > 3)):
            break

    if out is True:
        break