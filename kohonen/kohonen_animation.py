import numpy as np
import random
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickWriter
from PIL import Image

size = 100
total_size = size * 3
variance = 0.5
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
            colors.append('c')
        elif i == 4:
            colors.append('k')
    return colors

X_train = np.concatenate((np.random.multivariate_normal([2, 2, 2], variance*np.eye(dimension), size), 
                        np.random.multivariate_normal([0, -2, -2], variance*np.eye(dimension), size), 
                        np.random.multivariate_normal([-2, 0, 2], variance*np.eye(dimension), size),
                        np.random.multivariate_normal([2, 2, -2], variance*np.eye(dimension), size)))

y_train = np.concatenate((np.zeros(size), np.ones(size), 2*np.ones(size), 3*np.ones(size)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=create_color_list(y_train))

grid_x1 = int(10)
grid_x2 = int(10)
weight_size = grid_x1 * grid_x2
weights = (np.random.rand(weight_size, dimension) - 0.5) * 5
neuron_class = 4 * np.ones(weight_size)
ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='k', marker='^')
plt.show()

index_map = np.zeros((weight_size, 2))
for i in range(grid_x2):
    for j in range(grid_x1):
        index_map[i*grid_x2 + j, :] = [i, j]      

print(index_map)

dataset = list(zip(X_train, y_train))
random.shuffle(dataset)
train_set = dataset[:train_size]
test_set = dataset[train_size:]

def exp_scale(k, sigma_0 = 3, sigma_1 = 70):
    return sigma_0 * np.exp(-1 * k / sigma_1)

fig = plt.figure(constrained_layout=True)
widths = [2, 4, 1]
heights = [1, 4, 1]
gs = fig.add_gridspec(3, 3, width_ratios=widths, height_ratios=heights)
print(gs)
ax = fig.add_subplot(gs[:, 1:], projection='3d')
ax.axis([-3,3,-3,3])
ax.set_zlim([-3,3]) 
ax.title.set_text('Data Space')
plt.pause(0.1)

ax2 = fig.add_subplot(gs[1, 0])
scat3 = ax2.scatter(index_map[:, 0], index_map[:, 1], c=create_color_list(neuron_class), s = 200)
h_grid = np.zeros((grid_x1, grid_x2))
im = ax2.imshow(h_grid, vmin=0, vmax=1, cmap='inferno')  
ax2.title.set_text('Neural Neighborhood')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(2)

def neighbor(k, i):
    global scat3, ax2, h_grid
    scaling = 2 * np.square(exp_scale(k))
    h = np.zeros(len(index_map))
    for j in range(len(index_map)):
        neuron_dist = np.linalg.norm(index_map[i, :]-index_map[j, :])
        h[j] = np.exp(-1 * neuron_dist / scaling)
    return h

a, b = zip(*train_set)
train_x = np.array(a)
train_y = np.array(b)
ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], s=5, c=create_color_list(train_y))
scat = ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='k', marker='^')

max_iterations = 100
learning_rate = 0.005
old_weights = np.zeros(weights.shape)
writer = ImageMagickWriter(fps=4)
file_name = "kohonens.gif"
try:
    with writer.saving(fig, file_name, dpi=100):
        for k in range(max_iterations):
            neuron_class = 4 * np.ones(weight_size)
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

            for j in range(len(index_map)):
                h_grid[int(j%grid_x1), int(j/grid_x1)] = h[j]
            
            scat3.remove()
            scat3 = ax2.scatter(index_map[:, 0], index_map[:, 1], s=50, c=create_color_list(neuron_class))
            upscaled_image = Image.fromarray(h_grid).resize([100, 100], resample=Image.LANCZOS)
            upscaled_image = np.asarray(upscaled_image)
            im.set_data(upscaled_image)

            scat.remove()
            scat = ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], s=10, c=create_color_list(neuron_class), marker='^')

            plt.pause(0.1)        
            writer.grab_frame()
            if(np.linalg.norm(weights - old_weights).mean() < 0.01):
                break
except KeyboardInterrupt:
    pass
writer.finish()