import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

#Create a MLP network input is a list
class MLP():
    def __init__(self, network, learning_rate = 1e-2, activation_type = "tanh"):
        #Initialize
        self.network = network
        self.network_size = len(network) - 1
        self.learning_rate = learning_rate
        self.activation_type = activation_type

        self.reset()
        print(self.__str__())

    def reset(self):
        #Reset MLP variables
        self.weights = []

        for i in range(self.network_size):
            self.weights.append( np.random.rand(self.network[i + 1], self.network[i]+1) - 0.5 )

    def forward(self, data):
        #Check the dimension of the data
        if data.shape[1] != self.network[0]:
            raise ValueError("Error, wrong input dimension")

        #First get batch size
        self.batch_size = data.shape[0]

        #Create forward pass buffers
        self.outputs = []
        for _ in range(self.batch_size):
            batch_outputs = []
            for i in range(self.network_size + 1):
                batch_outputs.append(np.concatenate((np.zeros((self.network[i], 1)), np.ones((1,1)))))
            self.outputs.append(batch_outputs)

        #Load data to outputs[0] that is the inputs 
        for batch_index in range(self.batch_size):
            self.outputs[batch_index][0][:-1] = data[batch_index, :].reshape(-1, 1)

        #Inference
        for batch_index in range(self.batch_size):
            #For every layer
            for layer_index in range(self.network_size):
                #Make multiplications and tanh
                self.outputs[batch_index][layer_index + 1][:-1] = np.tanh( 
                    np.matmul(self.weights[layer_index], self.outputs[batch_index][layer_index]) )

        #Output buffer
        self.output = np.zeros((self.batch_size, self.network[-1]))

        for batch_index in range(self.batch_size):
            self.output[batch_index] = self.outputs[batch_index][-1][:-1].squeeze()

        return self.output[batch_index]

    def backward(self, ground_truth):
        #Intialize training buffers
        errors = []
        local_gradients = []
        derivatives = []
        for _ in range(self.batch_size):
            errors.append (np.zeros((self.network[-1], 1)))
            batch_gradients = []
            batch_derivatives = []
            for layer_index in range(self.network_size):
                batch_gradients.append(np.zeros((self.network[layer_index+1], 1)))
                batch_derivatives.append(np.zeros((self.network[layer_index + 1], self.network[layer_index]+1)))

            local_gradients.append(batch_gradients)
            derivatives.append(batch_derivatives)

        for batch_index in range(self.batch_size):
            #Calculate errors
            errors[batch_index] = ground_truth[batch_index] - self.output[batch_index]

            #Calculate gradients in reverse
            for layer_index in range(self.network_size):
                local_gradients[batch_index][layer_index] = 0



    def __str__(self):
        return "Network: %s, Learning Rate: %f, Activation Type: %s"% (self.network, self.learning_rate, 
                                                                                    self.activation_type)

def labels_to_vector(labels):
    result = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        result[i, int(labels[i])] = 1
    return result

def vector_to_labels(labels):
    return np.array([np.argmax(labels)])

def mean_square_error(ground_truth, desired):
    temp = np.square(ground_truth - desired)
    temp = np.mean( temp )
    return temp

#First open and read the file
f = open("iris.data", "r")
data = f.read()
#Split each line and create the list
data_list = data.split("\n")

#Seperate each data
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

#Now we will seperate 40 of each class to train set and remaining 10 to test set, we will also shuffle the selections
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

#Lastly birng all the data
dataset_train = np.array(dataset[np.concatenate((cls1_train, cls2_train, cls3_train)), :])
dataset_test = np.array(dataset[np.concatenate((cls1_test, cls2_test, cls3_test)), :])
print(dataset_train.shape, dataset_test.shape)
print(dataset_train[0])

#Create perceptron
network = [4, 20, 10, 3]
mlp = MLP(network, learning_rate = 1e-1)

input_data = dataset_train[[0, 42, 82], 0:-1]
ground_truth = labels_to_vector( dataset_train[[0, 42, 82], -1] )

print(input_data, ground_truth)

mlp.forward(input_data)
mlp.backward(ground_truth)