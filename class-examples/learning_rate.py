import numpy as np
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 1000, endpoint=True)
#seconds
simulation_speed = 0.4

def function(x):
    return x**2 + np.sin(np.pi * x) + 1

def derivative(x):
    return 2 * x + np.pi * np.cos(np.pi * x)

y = function(x)
learning_rate = 0.01
print("Click on plot to set starting x point")
#select inital point
fig, ax = plt.subplots()
line, = ax.plot(x,y)
plt.title("Learning Rate: %.3f"%(learning_rate))
plt.grid()
plt.pause(simulation_speed)

#x_point = x[np.random.randint(len(x))]
points = fig.ginput(show_clicks = True)
x_point = points[0][0]
y_point = function(x_point)
print("Start: ", x_point, y_point)

ax.scatter(x_point, y_point, s=20, c='r')
plt.pause(simulation_speed)

old_x = 0
y_point_old = 0
for i in range(50):
    #update point
    old_x = x_point
    y_point_old = y_point
    x_point = x_point - learning_rate * derivative(x_point)
    y_point = function(x_point)
    print("Epoch %d:"%(i), x_point, y_point)
    ax.scatter(x_point, y_point, s=20, c='r')
    plt.pause(simulation_speed)
    if(i > 15 and (abs(y_point-y_point_old) < 0.001) or y_point < 0.23):
        break

print("Iteration Stopped")

plt.show()


