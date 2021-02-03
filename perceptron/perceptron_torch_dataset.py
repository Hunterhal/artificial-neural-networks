import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def create_color_list(data_set):
    return ["b" if data==1 else "r" for data in data_set]

dimension = 2
size = 100
X_train = np.concatenate((np.random.multivariate_normal(np.ones(dimension), 0.3*np.eye(dimension), size), 
                          np.random.multivariate_normal(-1 * np.ones(dimension), 0.3*np.eye(dimension), size)))

y_train = np.concatenate((np.ones(size), -1*np.ones(size)))

print(y_train.shape)

#Create perceptron
perceptron = nn.Sequential(
    nn.Linear(2, 1),
    nn.Tanh()
)
print(perceptron)
#Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(perceptron.parameters(), lr=1e-2)

#Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
plt.title("Training Data")
plt.show()

#Error buffer
mse = []
#Numpy to torch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

dataset_train_torch = TensorDataset(X_train_torch, y_train_torch) # create your datset
train_loader = DataLoader(dataset_train_torch, shuffle=True, batch_size=1) # create your dataloader

#Draw class separating line
discriminator = np.zeros((2,2))
discriminator[0,0] = -10
discriminator[1,0] = 10
fig = plt.figure()
plt.axis([-3, 3, -3, 3])
plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)

a = perceptron[0].weight[0, 0] / perceptron[0].weight[0, 1]
b = perceptron[0].bias / perceptron[0].weight[0, 1]
discriminator[0,1] = -1 * a * discriminator[0,0] + b
discriminator[1,1] = -1 * a * discriminator[1,0] + b
line,  = plt.plot(discriminator[:, 0], discriminator[:, 1], c='k')

#Train loop
for i in range(1000):
    training_loss = 0
    for i_batch, data in enumerate(train_loader):
        features, labels = data[0], data[1]
        #Forward pass data
        yd = perceptron(features)
        #Clear gradients
        optimizer.zero_grad()
        #Calculate loss
        loss = criterion(labels, yd.view(1, 1))
        #Append loss data
        training_loss += loss.clone().detach()
        #Backpropagate error
        loss.backward()
        #Run optimizer
        optimizer.step()

    training_loss /= size
    mse.append(training_loss)

    #mse[-1] is the last error
    print("Iteration: {}, Error is: {}".format(i, mse[-1]))
    a = perceptron[0].weight[0, 0] / perceptron[0].weight[0, 1]
    b = perceptron[0].bias / perceptron[0].weight[0, 1]
    print(a, b)

    line.remove()
    discriminator[0,1] = -1 * a * discriminator[0,0] + b
    discriminator[1,1] = -1 * a * discriminator[1,0] + b
    line, = plt.plot(discriminator[:, 0], discriminator[:, 1], c='k')
    plt.pause(0.01)

    #Stop condition
    if mse[-1] < 0.001:
        print("Training stopped")
        break

mse = np.array(mse)
plt.figure()
plt.plot(np.arange(len(mse)), mse)
plt.grid()
plt.title("Training Loss")
plt.show()
plt.title("Training End")

