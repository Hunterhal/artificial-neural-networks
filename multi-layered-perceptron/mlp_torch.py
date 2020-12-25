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

f = open("iris.data", "r")
data = f.read()
data_list = data.split("\n")

data_seperate = []
dataset = []
for i in range(len(data_list)):
    dataset.append([])
    data_seperate = data_list[i].split(",")
    x1 = float(data_seperate[0])
    x2 = float(data_seperate[1])
    x3 = float(data_seperate[2])
    x4 = float(data_seperate[3])

    if data_seperate[4] == "Iris-virginica":
        y = 2
    elif data_seperate[4] == "Iris-versicolor":
        y = 1
    elif data_seperate[4] == "Iris-setosa":
        y = 0

    dataset[i].append([x1, x2, x3, x4, y])

train = int(40)
dataset = np.array(dataset, dtype=np.float32).reshape(150, 5)
cls1 = np.arange(50, dtype=np.int)
random.shuffle(cls1)
cls1_train = cls1[0: train]
cls1_test = cls1[train:]

cls2 = np.arange(50, 100, dtype=np.int)
random.shuffle(cls2)
cls2_train = cls2[0: train]
cls2_test = cls2[train:]

cls3 = np.arange(100, 150, dtype=np.int)
random.shuffle(cls3)
cls3_train = cls3[0: train]
cls3_test = cls3[train:]

dataset_train = dataset[np.concatenate((cls1_train, cls2_train, cls3_train)), :]
dataset_test = dataset[np.concatenate((cls1_test, cls2_test, cls3_test)), :]

def create_color_list(data_set):
    result=[]
    for i in range(len(data_set)):
        if(data_set[i] == 0):
            result.append('b')
        elif(data_set[i] == 1):
            result.append('g')
        elif(data_set[i] == 2):
            result.append('r')

    return result

def labels_to_vector(labels):
    result = torch.zeros(len(labels), 3, dtype=torch.float32)
    for i in range(len(labels)):
        result[i, int(labels[i])] = 1
    return result

def vector_to_labels(labels):
    return torch.argmax(labels, axis=1).detach().numpy()

colors = create_color_list(dataset[:, 4])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], s=dataset[:, 3]*10, c=colors)
plt.show()
batch_size = 4
dataset_train_torch = TensorDataset(torch.tensor(dataset_train[:, 0:4]), torch.tensor(labels_to_vector(dataset_train[:, 4]))) # create your datset
train_loader = DataLoader(dataset_train_torch, shuffle=True, batch_size=batch_size) # create your dataloader
dataset_test_torch = TensorDataset(torch.tensor(dataset_test[:, 0:4]), torch.tensor(labels_to_vector(dataset_test[:, 4]))) # create your datset
test_loader = DataLoader(dataset_test_torch, shuffle=True, batch_size=batch_size) # create your dataloader

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.01)

network = nn.Sequential(
    nn.Linear(4, 10),
    nn.Sigmoid(),
    nn.Linear(10, 3),
    nn.Sigmoid()
)

print(network)
max_epoch = 500

criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=2e-2, momentum=0.9)

for test_number in range(10):
    network.apply(init_weights)
    loss_buffer = np.zeros((max_epoch, 2))
    for epoch in range(max_epoch):
        train_loss = 0
        test_loss = 0
        for i_batch, data in enumerate(train_loader):
            features, labels = data[0], data[1]
            predict = network(features)

            optimizer.zero_grad()
            loss = criterion(predict, labels)
            train_loss += loss.data
            loss.backward()
            optimizer.step()

        test_gt = np.array([])
        test_pred = np.array([])
        for i_batch, data in enumerate(test_loader):
            features, labels = data[0], data[1]
            predict = network(features)

            test_gt = np.concatenate( (test_gt, vector_to_labels(labels)) )
            test_pred = np.concatenate( (test_pred, vector_to_labels(predict)) )

            loss = criterion(predict, labels)
            test_loss += loss.data

            loss_buffer[epoch, :] = [train_loss/40, test_loss/10]

        if(test_loss/10 < 0.02):
            break

    print("Test :%d, Epoch: %d, Training loss: %.3f, Test loss: %.3f"%(test_number, epoch, train_loss/40, test_loss/10))
    
    cm = confusion_matrix(test_gt, test_pred)
    print("Confusion matrix:")
    print(cm)
    plt.plot(np.arange(epoch), loss_buffer[0:epoch, 1], label="%d. Test"%(test_number))
plt.legend()
plt.grid()
plt.show()