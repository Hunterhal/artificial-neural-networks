import numpy as np
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 1000, endpoint=True)
#seconds
simulation_speed = 0.4

def function(x):
    return x**2 - 0.6 * np.cos(np.pi * 3 * x) + 1

def derivative(x):
    return 2 * x + 0.6 * 5 * np.pi * np.sin(np.pi * 3 * x)

y = function(x)
learning_rate = 0.01
momentum = 0.9
print("Click on plot to set starting x point")
#select inital point
fig, ax = plt.subplots()
line, = ax.plot(x,y)
plt.title("Learning Rate: %.3f, Momentum: %.2f" % (learning_rate, momentum))
plt.grid()
plt.pause(simulation_speed)

#x_point = x[np.random.randint(len(x))]
points = fig.ginput(show_clicks = True)
x_point = points[0][0]
y_point = function(x_point)
print("Start: ", x_point, y_point)

ax.scatter(x_point, y_point, s=20, c='r')
plt.pause(simulation_speed)
older_x_point = x_point
old_x = x_point
y_point_old = 0

for i in range(50):
    #update point
    older_x_point = old_x
    y_point_old = y_point
    old_x = x_point
    x_point = x_point - learning_rate * derivative(x_point)
    x_point = x_point + momentum * (old_x - older_x_point)
    y_point = function(x_point)
    print("Epoch %d:"%(i), x_point, y_point)
    ax.scatter(x_point, y_point, s=20, c='r')
    plt.pause(simulation_speed)
    if(i > 15 and (abs(y_point-y_point_old) < 0.001) or y_point < 0.23):
        break

print("Iteration Stopped")

plt.show()


