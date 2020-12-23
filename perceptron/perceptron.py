import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#Create a perceptron
class Perceptron():
    def __init__(self, dimensions, learning_rate = 1e-2, bias = True, activation_type = "tanh", momentum = 0.9, batch_type = "batch", batch_size = 4):
        #Initialize
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.bias = bias
        self.momentum = momentum
        self.activation_type = activation_type
        self.batch_size = batch_size
        self.batch_type = batch_type

        self.reset()
        print(self.__str__())

    def reset(self):
        #Reset perceptron variables
        if self.bias is True:
            self.weights = np.random.rand(self.dimensions + 1) - 0.5
        else:
            self.weights = np.random.rand(self.dimensions) - 0.5
            
        self.data = np.zeros((1, self.dimensions))
        self.lin_comb = 0
        self.activation = 0
        self.old_weights = self.weights

    def forward(self, data):
        #Forward pass data through the perceptron
        if self.bias is True:
            self.data = np.concatenate((data, np.ones((len(data), 1))), axis = 1)
        else:
            self.data = np.array(data)

        self.lin_comb = self.data.dot(self.weights)
        self.activation = self.activation_function(self.lin_comb)
        return self.activation

    def activation_function(self, data):
        #Select activation function and use it
        if self.activation_type is "tanh":
            return np.tanh(data)
        elif self.activation_type is "sigmoid":
            return 1/ (1 + np.power(np.e, -1 * data))
        else:
            return data

    def clear_gradients(self):
        #Clear the gradients
        self.gradients = 0

    def backward(self, error):
        #Calculate gradients for all the data
        self.gradients = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(self.data.shape[0]):
            if self.activation_type is "tanh":
                self.gradients[i] = -1 * error[i] * (1 - self.activation[i]**2) * self.data[i]
            elif self.activation_type is "sigmoid":
                self.gradients[i] = -1 * error[i] * (self.activation[i] * (1 - self.activation[i])) * self.data[i]
            else:
                self.gradients[i] = -1 * error[i] * 1 * self.data[i]

        #Update the weights
        if self.batch_type is "single":
            #Update weights for every data
            indices = np.arange(len(self.data))
            #Shuffle training set "Stochastic Descent"
            random.shuffle(indices)
            for i in indices:        
                #Calculate momentum term
                momentum_term = self.momentum * (self.weights - self.old_weights)
                self.old_weights = self.weights
                self.weights = self.weights - self.learning_rate * self.gradients[i]
                #Lastly add momentum term
                self.weights += momentum_term

        if self.batch_type is "all":
            #Calculate momentum term
            momentum_term = self.momentum * (self.weights - self.old_weights)
            self.old_weights = self.weights
            #Find the mean of all gradients and one time update weights
            mean_gradient = self.gradients.mean(axis = 0)
            self.weights = self.weights - self.learning_rate * mean_gradient
            #Lastly add momentum term
            self.weights += momentum_term

        if self.batch_type is "batch":
            #Select random batches and update weights
            indices = list(np.arange(len(self.data)))
            #Shuffle training set
            random.shuffle(indices)
            batch_steps = int(len(self.data) / self.batch_size)
            for i in range(batch_steps):
                batch_indices = []
                #First sample random indices
                for _ in range(self.batch_size):
                    batch_indices.append(indices.pop())

                #Calculate momentum term
                momentum_term = self.momentum * (self.weights - self.old_weights)
                self.old_weights = self.weights
                batch_gradient = self.gradients[batch_indices].mean(axis = 0)
                self.weights = self.weights - self.learning_rate * batch_gradient
                #Lastly add momentum term
                self.weights += momentum_term

    def training(self, ground_truth):
        #calculate error 
        error = ground_truth - self.activation
        #run backward code
        self.backward(error)

    def __str__(self):
        return "Dimensions: %d, Bias: %d, Learning Rate: %f, Activation Type: %s"% (self.dimensions, 
                                                                                    self.bias, 
                                                                                    self.learning_rate, 
                                                                                    self.activation_type)

def mean_square_error(ground_truth, desired):
    temp = np.square(ground_truth - desired)
    temp = np.mean( temp )
    return temp

def create_color_list(data_set):
    return ["b" if data==1 else "r" for data in data_set]

if __name__ == "__main__":
    dimension = 2
    size = 100
    X_train = np.concatenate((np.random.multivariate_normal(np.ones(dimension), 0.3*np.eye(dimension), size), 
                            np.random.multivariate_normal(-1 * np.ones(dimension), 0.3*np.eye(dimension), size)))

    y_train = np.concatenate((np.ones(size), -1*np.ones(size)))

    print(y_train.shape)
    #Create perceptron
    perceptron = Perceptron(dimension, learning_rate = 1e-3, activation_type = "tanh", batch_type= "batch")

    plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
    plt.title("Dataset")
    plt.show()
    mse = []
    #Train loop
    for i in range(1000):
        yd = perceptron.forward(X_train)
        #print(yd)
        mse.append( mean_square_error(y_train, yd) )
        #mse[-1] is the last error
        print("Iteration: {}, Error is: {}".format(i, mse[-1]))
        #Stop condition
        if mse[-1] < 0.01:
            break
        error = y_train - yd
        #print(error)
        #Backpropagate error
        perceptron.training(y_train)

    mse = np.array(mse)
    plt.plot(np.arange(len(mse)), mse)
    plt.grid()
    plt.title("Training MSE")
    plt.show()

    #Draw class separating line
    discriminator = np.zeros((2,2))
    discriminator[0,0] = -10
    discriminator[1,0] = 10
    discriminator[0,1] = -1 * (perceptron.weights[0] * discriminator[0,0] + perceptron.weights[2]) / perceptron.weights[1]
    discriminator[1,1] = -1 * (perceptron.weights[0] * discriminator[1,0] + perceptron.weights[2]) / perceptron.weights[1]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=create_color_list(y_train), s=5)
    plt.plot(discriminator[:, 0], discriminator[:, 1], c='k')
    plt.axis([-3, 3, -3, 3])
    plt.title("Training End")
    plt.show()