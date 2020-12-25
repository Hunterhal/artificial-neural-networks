import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#Create a MLP network input is a list
class MLP():
    def __init__(self, network, learning_rate = 1e-2, bias = True, activation_type = "tanh"):
        #Initialize
        self.network = network
        self.network_size = len(network) - 1
        self.learning_rate = learning_rate
        self.enable_bias = bias
        self.activation_type = activation_type

        self.reset()
        print(self.__str__())

    def reset(self):
        #Reset MLP variables
        self.weights = []

        for i in range(self.network_size):
            if self.enable_bias is True:
                self.weights.append( np.random.rand(self.network[i + 1]+1) - 0.5 )
            else:
                self.weights.append( np.random.rand(self.network[i], self.network[i+1]) - 0.5 )

    def forward(self, data):
        #Check the dimension of the data
        if data.shape[1] != self.network[0]:
            raise ValueError("Error, wrong input dimension")

        #Bias is enabled add 1 at the end of the data
        if self.enable_bias is True:
            self.weights.append( np.random.rand(self.network[i + 1]+1) - 0.5 )
        else:
            self.weights.append( np.random.rand(self.network[i], self.network[i+1]) - 0.5 )

        #Get batch size from the input data
        self.batch_size = data.shape[0]

        #Create forward pass buffers
        self.activations = []
        self.lin_combs = []
        #First activation is the input data
        self.activations.append(np.array(data))
        #Initialize other layers as zeros
        for layer in range(self.network_size):
            self.activations.append(np.zeros((self.batch_size, self.network[layer + 1])))
            self.lin_combs.append(np.zeros((self.batch_size, self.network[layer + 1])))  

        #Forward pass aall the data through the network
        for batch_index in range(self.batch_size):
            for layer in range(self.network_size):    
                #print(self.activations[layer][batch_index].shape, self.weights[layer].shape)        
                self.lin_combs[layer][batch_index] = np.matmul(self.activations[layer][batch_index], 
                                                               self.weights[layer])
                self.activations[layer + 1][batch_index] = self.activation_function(self.lin_combs[layer][batch_index])
        return self.activations[-1]

    def activation_function(self, data):
        #Select activation function and use it
        if self.activation_type is "tanh":
            return np.tanh(data)
        elif self.activation_type is "sigmoid":
            return 1/ (1 + np.power(np.e, -1 * data))
        elif self.activation_type is "relu":
            return np.where(data>0, data, 0)
        else:
            return data

    def derivative_activation(self, layer, batch):
        #Select activation function and use it
        if self.activation_type is "tanh":
            result = 1 - (self.activations[layer+1][batch])**2
        elif self.activation_type is "sigmoid":
            result = self.activations[layer+1][batch] * (1 - self.activations[layer+1][batch])
        elif self.activation_type is "relu":
            result = np.where(self.lin_combs[layer][batch]>0, 1.0, 0)
        else:
            result = np.ones((1, self.network[layer + 1]))
        return result

    def backward(self, ground_truth):
        #Intialize local gradients buffer
        local_gradients = []
        weight_gradients = []
        bias_gradients = []
        for layer in range(self.network_size):
            local_gradients.append(np.zeros((self.batch_size, self.network[layer+1])))
            if self.enable_bias is True:
                bias_gradients.append(np.zeros((self.network[layer+1])))

            weight_gradients.append(np.zeros((self.network[layer], self.network[layer+1])))

        #Calculate error
        self.error = ground_truth.reshape(self.batch_size, 1) - self.activations[-1]
        
        #Calculate the local gradients of output
        for batch_index in range(self.batch_size):
            local_gradients[-1][batch_index] = -1 * self.error[batch_index] * self.derivative_activation(-1, batch_index)

        #Calculate local gradients for all the layers
        for batch_index in range(self.batch_size):
            for layer in reversed(range(self.network_size - 1)):
                temp1 = np.matmul(self.weights[layer+1], local_gradients[layer + 1][batch_index].reshape(-1, 1)).reshape(-1)
                temp2 = self.derivative_activation(layer, batch_index)
                #print(temp1.shape, temp2.shape, (temp1*temp2).shape)
                local_gradients[layer][batch_index] = temp1*temp2

        #Calculate weight and bias gradients
        for batch_index in range(self.batch_size):
            for layer in reversed(range(self.network_size)):
                temp1 = local_gradients[layer][batch_index].reshape(1, self.network[layer+1])
                temp2 = self.activations[layer][batch_index].reshape(self.network[layer], 1)
                #print(temp1.shape, temp2.shape)
                #print((np.matmul(temp2, temp1)).shape)
                weight_gradients[layer] += np.matmul(temp2, temp1).reshape(self.network[layer], self.network[layer + 1])
                bias_gradients[layer] += local_gradients[layer][batch_index]

        #Calculate mean gradients
        for layer in reversed(range(self.network_size)):
            weight_gradients[layer] /= self.batch_size
            bias_gradients[layer] /= self.batch_size

        #Update weights and bias
        for layer in reversed(range(self.network_size)):
            self.weights[layer] -= self.learning_rate * weight_gradients[layer]

    def __str__(self):
        return "Network: %s, Bias: %d, Learning Rate: %f, Activation Type: %s"% (self.network, 
                                                                                    self.enable_bias, 
                                                                                    self.learning_rate, 
                                                                                    self.activation_type)

def mean_square_error(ground_truth, desired):
    temp = np.square(ground_truth - desired)
    temp = np.mean( temp )
    return temp

def create_color_list(data_set):
    return ["b" if data==1 else "r" for data in data_set]

dimension = 2
size = 100
X_train = np.concatenate((np.random.multivariate_normal(np.ones(dimension), 0.3*np.eye(dimension), size), 
                          np.random.multivariate_normal(-1 * np.ones(dimension), 0.3*np.eye(dimension), size)))

y_train = np.concatenate((np.ones(size), -1*np.ones(size)))

print(y_train.shape)
#Create perceptron
network = [2, 10, 20, 1]
mlp = MLP(network, learning_rate = 1e-1)

plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
plt.title("Training End")
plt.show()
mse = []
data_size = len(X_train)
batch_size = 4
#Train loop
for i in range(100):
    iter_index = np.arange(data_size/batch_size)
    random.shuffle(iter_index)
    error = 0
    for j in iter_index:
        window = np.arange(j * batch_size, dtype=np.int)
        yd = mlp.forward(X_train[window])
        error += mean_square_error(y_train[window], yd)
        #mse[-1] is the last error
        #Backpropagate error
        mlp.backward(y_train[window])

    mse.append( error/50 )
    print("Iteration: {}, Error is: {}".format(i, mse[-1]))
    #Stop condition
    if mse[-1] < 0.026:
        break

mse = np.array(mse)
plt.plot(np.arange(len(mse)), mse)
plt.grid()
plt.title("Training MSE")
plt.show()