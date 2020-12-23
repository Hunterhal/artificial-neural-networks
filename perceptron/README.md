# Perceptron - Adaline Algorithm

Refer to Jupyter notebootk for the theory and basic coding.  
Beware that the data is created everytime program runs and sampled randomly. For better analysis, create and save data then change the parameters.  
Example code is written for two dimensional data, change accordingly.
Tanh is selected as default, if you change beware of class (-1, 1) or (0, 1).  
Figures are created from numpy based perceptron, results can be generated for torch.  
For the torch code most important thing is torch has custom data type called torch tensor.  
Also torch natively built on batch processing, data has to be given in one by one.  

## Perceptron - Adaline

![Adaline](adaline.PNG)

## Results

The data is created artificially by two Gaussian distrbutions.  

![Training End](training_end.PNG)
![MSE](perceptron_training_mse.PNG)

## To-Do List

All Updated, add new features as challenge  
Add basic codes to jupyter notebook  
Create test set and run codes
Adam Optimizer
