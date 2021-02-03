import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def create_color_list(data_set):
    return ["b" if data==1 else "r" for data in data_set]

def calculate_map(perceptron, resolution):
    classification_map = torch.zeros(resolution, resolution)
    with torch.no_grad():
        for i in range(resolution):
            map_row = torch.cat((-2 + (i * 6 * torch.ones(resolution, 1) / resolution), -2 + (6 * torch.arange(resolution, dtype=torch.float32).view(-1, 1) / resolution)), axis=1)
            classification_map[i, :] = perceptron(map_row).view(-1)
    return classification_map

dimension = 2
size = 100
X_train = np.concatenate((np.random.multivariate_normal(2 * np.ones(dimension), 0.3*np.eye(dimension), size), 
                          np.random.multivariate_normal(0 * np.ones(dimension), 0.3*np.eye(dimension), size)))

y_train = np.concatenate((np.ones(size), -1*np.ones(size)))

print(y_train.shape)

#Create perceptron
perceptron = nn.Sequential(
    nn.Linear(2, 1, bias=True),
    nn.Tanh()
)
print(perceptron)

#Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(perceptron.parameters(), lr=1e-3, momentum=0.9)

#Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
plt.title("Training Data")
plt.show()

#Error buffer
mse = []
#Numpy to torch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)

#Draw class separating line
discriminator = np.zeros((2,2))
discriminator[0,0] = -10
discriminator[1,0] = 10
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
ax.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)

map_c = np.asarray(calculate_map(perceptron, 100).clone().detach())
im = ax.imshow(map_c, vmin=-1, vmax=1, cmap='RdBu', extent=[-2, 4, -2, 4], alpha=0.7)  

a = perceptron[0].weight[0, 0] / perceptron[0].weight[0, 1]
b = perceptron[0].bias / perceptron[0].weight[0, 1]
discriminator[0,1] = -1 * a * discriminator[0,0] + b
discriminator[1,1] = -1 * a * discriminator[1,0] + b
line, = ax.plot(discriminator[:, 0], discriminator[:, 1], c='k')

#Train loop
for i in range(1000):
    training_loss = 0
    indices = np.arange(size * 2)
    random.shuffle(indices)
    for j in indices:
        #Forward pass data
        yd = perceptron(X_train_torch[j, :])
        #Clear gradients
        optimizer.zero_grad()
        #Calculate loss
        loss = criterion(y_train_torch[j].view(1, 1), yd.view(1, 1))
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
    discriminator[0,1] = -1 * (a * discriminator[0,0] + b)
    discriminator[1,1] = -1 * (a * discriminator[1,0] + b)
    
    map_c = np.asarray(calculate_map(perceptron, 100).clone().detach())
    upscaled_image = Image.fromarray(map_c).resize([300, 300], resample=Image.LANCZOS)
    upscaled_image = np.asarray(upscaled_image.rotate(90))
    im.set_data(upscaled_image)
    line.remove()

    line, = ax.plot(discriminator[:, 0], discriminator[:, 1], c='k')
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

