import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

dimension = 2
size = 200
variance = 0.3
data_size = size//2
X_train = np.concatenate((np.random.multivariate_normal([0, 0], variance*np.eye(dimension), data_size), 
                          np.random.multivariate_normal([2, 2], variance*np.eye(dimension), data_size), 
                          np.random.multivariate_normal([0, 2], variance*np.eye(dimension), data_size), 
                          np.random.multivariate_normal([2, 0], variance*np.eye(dimension), data_size)))
"""
cls1 = np.random.multivariate_normal([1, 1], variance*np.eye(dimension), size//2)
phi = np.random.uniform(0, 2*np.pi, size//2)
rad = np.random.uniform(2.1, 2.6, size//2)
cls2 = np.array([rad*np.cos(phi) + 1, rad*np.sin(phi) + 1])
cls1 = np.concatenate((cls1, cls2.T))
phi = np.random.uniform(0, 2*np.pi, size)
rad = np.random.uniform(1.5, 2, size)
cls2 = np.array([rad*np.cos(phi) + 1, rad*np.sin(phi) + 1])


X_train = np.concatenate((cls1, cls2.T))
"""
y_train = np.concatenate((np.ones(size), -1*np.ones(size)))
#Numpy to torch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

def create_color_list(data_set):
    return ["b" if data==1 else "r" for data in data_set]

def calculate_map(perceptron, resolution):
    classification_map = torch.zeros(resolution, resolution)
    perceptron.train(False)
    with torch.no_grad():
        for i in range(resolution):
            map_row = torch.cat((-2 + (i * 6 * torch.ones(resolution, 1) / resolution), -2 + (6 * torch.arange(resolution, dtype=torch.float32).view(-1, 1) / resolution)), axis=1)
            classification_map[i, :] = perceptron(map_row).view(-1)
    perceptron.train(True)
    return classification_map

#Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
plt.title("Training Data")
plt.show()

batch_size = 1
print(X_train_torch.size())
dataset_train_torch = TensorDataset(X_train_torch, y_train_torch) 
train_loader = DataLoader(dataset_train_torch, shuffle=True, batch_size=batch_size) 

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

network = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Tanh()
)

print(network)
max_epoch = 500

criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)

#Draw class separating line
discriminator = np.zeros((2,2))
discriminator[0,0] = -10
discriminator[1,0] = 10
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
ax.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)

map_c = np.asarray(calculate_map(network, 100).clone().detach())
im = ax.imshow(map_c, vmin=-1, vmax=1, cmap='RdBu', extent=[-2, 4, -2, 4], alpha=0.6)  

network.apply(init_weights)
loss_buffer = np.zeros((max_epoch, 2))
mse = []
for epoch in range(max_epoch):
    training_loss = 0
    for i_batch, data in enumerate(train_loader):
        features, labels = data[0], data[1]
        predict = network(features)

        optimizer.zero_grad()
        loss = criterion(predict, labels)
        training_loss += loss.clone().detach()
        loss.backward()
        optimizer.step()

    training_loss /= size
    mse.append(training_loss)

    #mse[-1] is the last error
    print("Epoch: {}, Error is: {}".format(epoch, mse[-1]))
    
    map_c = np.asarray(calculate_map(network, 100).clone().detach())
    upscaled_image = Image.fromarray(map_c).resize([300, 300], resample=Image.LANCZOS)
    upscaled_image = np.asarray(upscaled_image.rotate(90))
    im.set_data(upscaled_image)
    plt.title("Epoch: %d, Error is: %.4f"%(epoch, mse[-1]))
    plt.pause(0.01)

    #Stop condition
    if mse[-1] < 0.01:
        print("Training stopped")
        break

mse = np.array(mse)
plt.figure()
plt.plot(np.arange(len(mse)), mse)
plt.grid()
plt.title("Training Loss")
plt.show()
plt.title("Training End")
