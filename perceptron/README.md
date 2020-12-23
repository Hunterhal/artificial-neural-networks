# Perceptron - Adaline Algorithm

Beware that the data is created everytime program runs and sampled randomly. For better analysis, create and save data then change the parameters.   
Example code is written for two dimensional data, change accordingly.
Tanh is selected as default, if you change beware of class (-1, 1) or (0, 1).  
Figures are created from numpy based perceptron, results can be generated for torch.  
For the torch code most important thing is torch has custom data type called torch tensor.  
Also torch natively built on batch processing, data has to be given in one by one.  

## Perceptron - Adaline

![Adaline](adaline.PNG)

The perceptron and Adaline are similar perceptron algortihm uses step function as activation function that is non differentiable, the Adaline has logistic function with limits and can be differantiated.  

### Inference - Forward Pass

First linear combination of the inputs is calculated with additional bias term.  
$$v=x_0w_0+x_1w_1+...+x_{n-1}w_{n-1}+w_n$$
Then this linear combination is given to logistic function.  
$$y=tanh(v)$$

### Training

In training phase, weights are updated by using backpropagation algorithm. The mean squared error, between output of the adaline $y$ and the ground truth $y_d$, is calculated by using
$$mse=\frac{1}{N}\overset{N}{\sum_{j}}(y_{dj}-y_j)^2$$
In the equation, $N$ is the number of samples in batch. Since the training is performed for each sample it is 1. The error between the output and the ground truth is
$$e=y_d-y$$
For the weight $w_i$ at the time step $k$, weight is defined by $w_i(k)$. The backpropagation algorithm calculates how the error is changed with respect to the weight $\frac{\partial E}{\partial w_i(k)}$. To calculate the partial derivative of error with respect to weights, the chain rule is implemented.
$$\frac{\partial E}{\partial w_i(k)}=\frac{\partial E}{\partial e}\frac{\partial e}{\partial y}\frac{\partial y}{\partial v}\frac{\partial v}{\partial w_i(k)}$$
In this equation,  
*$\frac{\partial E}{\partial e}$ is the derivative of mean square error formulation,  
*$\frac{\partial e}{\partial y}$ is the derivative of error,  
*$\frac{\partial y}{\partial v}$ is the derivative of activation function tanh  
*$\frac{\partial v}{\partial w}$ is the derivative of linear combination  
If this terms are substituted,
$$\frac{\partial E}{\partial w_i(k)}=e(-1)(1-tanh^2(v))x_i$$
Here the derivative of tanh function is used, for other activtion function this part must be changed. After this calculation the weights canbe updated by using  
$$w_i(k+1)=w_i(k)-lr\frac{\partial E}{\partial w_i(k)}$$
Unsuprisingly, the lr is learning rate.

### Batch Size

In data perspective, there are three ways to train neural networks  

 1. Batch Gradient Descent: All the data in training set is used for training.  
 2. Stochastic Gradient Descent: Only single randomly selected data is used for training  
 3. Mini-Batch Gradient Descent: Batch size is a number between 1 and whole training set.  

In the code the batch size is actually the mini-batch size. When the batch size is used, the derivatives are calculated for each sample then mean is calculated. $N$ is the batch size.  

$$\frac{\partial E}{\partial w_i(k)} = \frac{1}{N}\sum_{j}\frac{\partial E}{\partial w_{i,j}(k)}$$

### Momentum

Momentum is another useful for tool to improve convergence of neural networks. It is inspired from the physics. Please, look at the class-examples section. With momentum $m$, weight update rule is redefined as 

$$w_i(k+1)=w_i(k)-lr\frac{\partial E}{\partial w_i(k)}+m(w_i(k)-w_i(k-1))$$

## Results

The data is created artificially by two Gaussian distrbutions.  

![Training End](training_end.PNG)
![MSE](perceptron_training_mse.PNG)

## To-Do List

All Updated, add new features as challenge  
Adam Optimizer
