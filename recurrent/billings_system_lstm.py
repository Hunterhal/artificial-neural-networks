import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_dataset(sample_size = 100, initial = 50, seq_length=5):
    y = np.zeros(sample_size)
    noise = np.random.normal(0, 0.001, sample_size)

    for i in range(2, sample_size):
        y[i] = ((0.8 + 0.5 * np.exp(-1 * np.square(y[i-1])))*y[i-1] - 
                (0.3 + 0.9 * np.exp(-1 * np.square(y[i-1])))*y[i-2] + 
                (0.1 * np.sin(np.pi * y[i-1])) + 
                noise[i])

    noise = np.linspace(0, 4 * 2 * np.pi, sample_size)
    y = np.cos(noise)

    x_train = np.empty((1, seq_length), dtype=np.float32)
    y_train = np.empty((1, 1), dtype=np.float32)
    for i in range((sample_size - initial)//seq_length - 1):
        x_temp = np.array(noise[initial + (i*seq_length):initial + ((i+1)*seq_length)], dtype=np.float32).reshape(1, -1)
        y_temp = np.array(y[initial + ((i+1)*seq_length-1)], dtype=np.float32).reshape(1, -1)
        x_train = np.concatenate((x_train, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)

    #print(x_train.shape, y_train.shape)
    #print(x_train)
    #print(y_train)
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
    def __init__(self, context_size, seq_length = 5):
        super(Net, self).__init__()
        self.seq_legth = seq_length
        enable_bias = False

        self.wf = nn.Linear(1, context_size, bias=enable_bias)
        self.uf = nn.Linear(context_size, context_size, bias=enable_bias)

        self.wc = nn.Linear(1, context_size, bias=enable_bias)
        self.uc = nn.Linear(context_size, context_size, bias=enable_bias)

        self.wi = nn.Linear(1, context_size, bias=enable_bias)
        self.ui = nn.Linear(context_size, context_size, bias=enable_bias)

        self.wo = nn.Linear(1, context_size, bias=enable_bias)
        self.uo = nn.Linear(context_size, context_size, bias=enable_bias)

        self.out_layer = nn.Linear(context_size, 1, bias=False)

    def forward(self, x, h_pre, c_pre):
        H = h_pre
        C = c_pre
        for j in range(self.seq_legth):
            #f = torch.sigmoid( self.wf(x) + self.uf(H) )
            g = torch.tanh( self.wc(x) + self.uc(H) )
            #i = torch.sigmoid( self.wi(x) + self.ui(H) )
            #o = torch.sigmoid( self.wo(x) + self.uo(H) )

            #C = (f * C) + (i * g)
            #H = o * torch.tanh(C)
            H = g
            
        out = self.out_layer(H)

        return out, H, C

context_size = 30
seq_length = 1
network = Net(context_size, seq_length=seq_length)

print(network)
max_epoch = 100

sample_size = int(300)
initial = int(0)
train_test = 0.9
test_index = int((sample_size - initial) * train_test)

criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=1e-2, momentum=0.9)
y_pred = np.zeros(sample_size - initial, dtype=np.float32)
t = np.arange(sample_size)
# y, train_loader = create_dataset(sample_size = sample_size, initial = initial, seq_length=seq_length)
# plt.plot(t[initial:], y[initial:])
# plt.show()
out = False

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax_train_state = fig.add_subplot(gs[0, 0])
ax_test_state = fig.add_subplot(gs[1, 0])
ax_main = fig.add_subplot(gs[:, 1:])

for test_number in range(100):
    network.apply(init_weights)
    loss_buffer = np.zeros(max_epoch)
    y, train_loader = create_dataset(sample_size = sample_size, initial = initial, seq_length=seq_length)
    h = torch.zeros(1, context_size)
    c = torch.zeros(1, context_size)
    train_loss_buffer = []
    train_loss_buffer.append(0)
    for epoch in range(max_epoch):
        train_loss = 0
        test_loss = 0
        noise = np.random.normal(0, 0.01, sample_size).reshape(-1, 1).astype(np.float32)
        for i_batch, data in enumerate(train_loader):
            features, labels = data[0], data[1]
            predict, H, C = network(torch.tensor(noise[i_batch]), h, c)
            y_pred[i_batch] = predict.clone().detach()
            if i_batch < (test_index):
                optimizer.zero_grad()
                loss = criterion(predict, labels)
                train_loss += loss.data
                loss.backward()
                optimizer.step()
            
            h = H.clone().detach()
            c = C.clone().detach()

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
        elif( (train_loss_buffer[-1] > 0.2) and (len(train_loss_buffer) > 5) or torch.isnan(train_loss)):
            break

    if out is True:
        break
plt.show()