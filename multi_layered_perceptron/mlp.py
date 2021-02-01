import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import copy

#Create a MLP network input is a list
class MLP():
    def __init__(self, network, momentum = 0.9, learning_rate = 1e-2, activation_type = "tanh"):
        #Initialize
        self.network = network
        self.network_size = len(network) - 1
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_type = activation_type

        self.reset()
        print(self.__str__())

    def reset(self):
        #Reset MLP variables
        self.weights = []

        #Xavier initialization
        for i in range(self.network_size):
            self.weights.append( (np.random.rand(self.network[i + 1], self.network[i]+1) * 2  - 1 ) * np.sqrt(6/(self.network[i + 1] + self.network[i])))

        #Copy previous and new weights
        self.previous_weights = copy.deepcopy(self.weights)
        self.new_weights = copy.deepcopy(self.weights)

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

        #Load input data to outputs[0]
        #outputs[:-1] because outputs[-1] is bias
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

        #Fill the output buffers
        for batch_index in range(self.batch_size):
            self.output[batch_index] = self.outputs[batch_index][-1][:-1].squeeze()

        return self.output

    def backward(self, ground_truth):
        #Intialize training buffers
        errors = []
        local_gradients = []
        derivatives = []
        for _ in range(self.batch_size):
            errors.append(np.zeros((self.network[-1], 1)))
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

            #Gradients of output layer
            local_gradients[batch_index][-1] = errors[batch_index] * -1 * (1 - self.outputs[batch_index][-1][:-1].squeeze())

            #Output derivative
            derivatives[batch_index][-1] = np.matmul(local_gradients[batch_index][-1].reshape(-1, 1), self.outputs[batch_index][-2].transpose())

            #Calculate local gradients and derivatives in reverse
            for layer_index in reversed(range(self.network_size - 1)):
                #No bias and transpose
                weight_temp = np.transpose(self.weights[layer_index + 1][:, :-1])

                temp = np.matmul(weight_temp, local_gradients[batch_index][layer_index + 1])

                local_gradients[batch_index][layer_index] = temp * (1 - (self.outputs[batch_index][layer_index+1][:-1].squeeze())**2)
                
                #Calculate derivatives
                derivatives[batch_index][layer_index] = np.matmul(local_gradients[batch_index][layer_index].reshape(-1, 1), self.outputs[batch_index][layer_index].transpose())

        #Calculate the mean of gradients inside the batch
        mean_derivatives = []
        for layer_index in range(self.network_size):
            mean_derivatives.append( np.zeros((self.network[layer_index + 1], self.network[layer_index]+1)) )

        for batch_index in range(self.batch_size):
            for layer_index in range(self.network_size):
                mean_derivatives[layer_index] += derivatives[batch_index][layer_index] / self.batch_size

        #Update weights
        for layer_index in range(self.network_size):
            momentum_term = self.momentum * (self.weights[layer_index] - self.previous_weights[layer_index])
            self.new_weights[layer_index] -= self.learning_rate * mean_derivatives[layer_index] + momentum_term

        #Copy weights
        self.previous_weights = copy.deepcopy(self.weights)
        self.weights = copy.deepcopy(self.new_weights)

    def __str__(self):
        return "Network: %s, Learning Rate: %f, Activation Type: %s"% (self.network, self.learning_rate, 
                                                                                    self.activation_type)

def labels_to_vector(labels):
    result = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        result[i, int(labels[i])] = 1
    return result

def vector_to_labels(labels):
    return np.argmax(labels, axis=1)

def mean_square_error(ground_truth, predicted):
    temp = np.square(ground_truth - predicted)
    temp = np.mean( temp )
    return temp

#This function creates batch windows
def create_batch_windows(batch_size, train_size, shuffle = True):
    #Shuffle the indices
    indices = list(np.arange(train_size))

    if(shuffle is True):   
        random.shuffle(indices)

    batch_windows = []
    while len(indices)>batch_size:
        window = []
        for _ in range(batch_size):
            window.append(indices.pop())
        batch_windows.append(window)

    if indices != []:
        window = []    
        while indices != []:
            window.append(indices.pop())
        batch_windows.append(window)

    return batch_windows

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

#Different trails can be defined here momentum tested
different_momentum = [0, 0.1, 0.5, 0.9]
trial_size = len(different_momentum)
trial_train_mse = []
trial_test_mse = []
c_matrix = []
try:
    for trial_number in range(trial_size):
        #Create perceptron
        network = [4, 10, 3]
        mlp = MLP(network, learning_rate = 1e-2, momentum=different_momentum[trial_number])

        #Training paramters
        mse_train = [] 
        mse_test = []
        batch_size = 2
        max_epoch = 1000

        #Main Loop
        for epoch in range(max_epoch):
            #Train Loop
            batch_windows = create_batch_windows(batch_size, len(dataset_train))
            train_epoch_mse = 0
            for batch in batch_windows:
                input_data = dataset_train[batch, 0:-1]
                ground_truth =  labels_to_vector( dataset_train[batch, -1] )
                #Inference
                out = mlp.forward(input_data)
                #Train
                mlp.backward(ground_truth)
                train_epoch_mse += mean_square_error(ground_truth, out)

            mse_train.append(train_epoch_mse/len(batch_windows))

            #Test Loop
            batch_windows = create_batch_windows(batch_size, len(dataset_test))
            test_epoch_mse = 0
            #For confussion matrix
            true_label = np.array([])
            predicted_label = np.array([])
            for batch in batch_windows:
                input_data = dataset_test[batch, 0:-1]
                ground_truth =  labels_to_vector( dataset_test[batch, -1] )
                #Inference
                out = mlp.forward(input_data)
                test_epoch_mse += mean_square_error(ground_truth, out)
                
                true_label = np.concatenate((true_label, vector_to_labels(ground_truth)), axis=0)
                #Maximum of the output is predicted
                predicted_label = np.concatenate((predicted_label, vector_to_labels(out)), axis=0)

            mse_test.append(test_epoch_mse/len(batch_windows))

            print("Trial: %d, Epoch: %d, Train: %.3f, Test %.3f"%(trial_number, epoch, mse_train[-1], mse_test[-1]))

            #Stop criteria
            if(mse_train[-1] < 0.025 and mse_test[-1] < 0.04):
                break

        print("Test Cofussion Matrix: ")
        c_matrix.append(confusion_matrix(true_label, predicted_label))
        print(c_matrix[-1])

        #Append trail mse
        trial_test_mse.append(mse_test)
        trial_train_mse.append(mse_train)

except KeyboardInterrupt:
    pass

plt.figure()
plt.title("Train Comparison")

for trial_number in range(trial_size):
    plt.plot(trial_train_mse[trial_number], label="%.1f"%(different_momentum[trial_number]))

plt.grid()
plt.legend()

plt.figure()
plt.title("Test Comparison")

for trial_number in range(trial_size):
    plt.plot(trial_test_mse[trial_number], label="%.1f"%(different_momentum[trial_number]))
    print("Confussion Matrix for Momentum: %.f"%(different_momentum[trial_number]))
    print(c_matrix[trial_number])

plt.grid()
plt.legend()
plt.show()