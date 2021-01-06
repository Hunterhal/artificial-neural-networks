import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

size = 100
total_size = size * 3
variance = 0.2
train_size = int(total_size * 0.8)
dimension = 3

def create_color_list(data_set):
    colors =[]
    for i in data_set:
        if i == 0:
            colors.append('b')
        elif i == 1:
            colors.append('g')
        elif i == 2:
            colors.append('r')
        elif i == 3:
            colors.append('k')
    return colors

X_train = np.concatenate((np.random.multivariate_normal([-1, 1, -1], variance*np.eye(dimension), size), 
                        np.random.multivariate_normal(-1 * np.ones(dimension), variance*np.eye(dimension), size), 
                        np.random.multivariate_normal(1*np.ones(dimension), variance*np.eye(dimension), size)))

y_train = np.concatenate((np.zeros(size), np.ones(size), 2*np.ones(size)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=create_color_list(y_train))

grid_x = int(10)
grid_y = int(10)
weight_size = grid_x * grid_y
weights = (np.random.rand(weight_size, dimension) - 0.5) * 5
neuron_class = 3 * np.ones(weight_size)
ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='k', marker='^')
plt.show()

index_map = np.zeros((weight_size, 2))
for i in range(weight_size):
    index_map[i, :] = [int(i/grid_y), int(i%grid_x)]

print(index_map)

dataset = list(zip(X_train, y_train))
random.shuffle(dataset)
train_set = dataset[:train_size]
test_set = dataset[train_size:]

def exp_scale(k, sigma_0 = 50, sigma_1 = 10):
    return sigma_0 * np.exp(-1 * k / sigma_1)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax2 = fig.add_subplot(1, 2, 1)
scat3 = ax2.scatter(index_map[:, 0], index_map[:, 1], c=create_color_list(neuron_class), s = 50)
scat2 = ax2.scatter(index_map[:, 0], index_map[:, 1], s = 5)

def neighbor(k, i):
    global scat2, scat3, ax2
    scaling = 2 * np.square(exp_scale(k))
    h = np.zeros(len(index_map))
    for j in range(len(index_map)):
        neuron_dist = np.linalg.norm(index_map[i, :]-index_map[j, :])
        h[j] = np.exp(-1 * neuron_dist / scaling)

    scat2.remove()
    scat3.remove()
    scat3 = ax2.scatter(index_map[:, 0], index_map[:, 1], s=50, c=create_color_list(neuron_class))
    scat2 = ax2.scatter(index_map[:, 0], index_map[:, 1], s=5, c = h)

    return h
a, b = zip(*train_set)
train_x = np.array(a)
train_y = np.array(b)
ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], s=5, c=create_color_list(train_y))
scat = ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='k', marker='^')

max_iterations = 1000
learning_rate = 0.01
old_weights = np.zeros(weights.shape)

for k in range(max_iterations):
    neuron_class = 3 * np.ones(weight_size)
    print(k)
    random.shuffle(train_set)
    for x, y in train_set:
        d = np.zeros(weight_size)
        for i in range(weight_size):
            d[i] = np.linalg.norm(x-weights[i, :])
        winner_index = np.argmin(d)
        neuron_class[winner_index] = y
        h = neighbor(k, winner_index)
        for j in range(weight_size):
            delta_w = learning_rate * h[j] * (x - weights[j])
            weights[j] += delta_w 

    scat.remove()
    scat = ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], s=10, c=create_color_list(neuron_class), marker='^')
    plt.pause(0.1)
    if(np.linalg.norm(weights - old_weights).mean() < 0.01):
        break

plt.show()