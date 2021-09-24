

## 1. What is a neural network neuron?
Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.


## 2. What is the use of the learning rate?
Learning rate is one of the most important hyperparameter used while training deep learning models. Learning rate (```alpha```) determines the size of the steps which a Gradient Descent optimizer takes along the direction of the slope of the surface derived from the loss function towards local minima.

Basic steps involved in training a machine learning model is shown in below figure. Learning rate (```alpha```) is used in **updation of parameters** which are then used to re-calculate the loss and this cycle continues till we achieve desired accuracy (acceptable loss).

![Learning_rate](https://user-images.githubusercontent.com/30425824/133917777-318f0de1-1607-433c-89f7-efbc4d195e76.jpg)

* **Small Learning rate** will slow down the training progress, but are also more desirable when the loss keeps getting worse or oscillates wildly. 
* **Large Learning rate** will accelerate the training progress rate and are beneficial when the loss is dropping fairly consistently in several consecutive iterations. However, large Learning rate value may also cause the training getting stuck at a local minimum, preventing the model from improving, resulting in either a low accuracy or not able to converge. 

Finding a constant learning rate that would improve the model is not always feasible and would involved several trails and errors to find the best values. Inspite of these iterations there may not be a fixed learning rate. There are several techniques that are evolved in the last few years and described below are some of the techniques available in Pytorch.

```torch.optim.lr_scheduler``` provides several methods to adjust the learning rate based on the number of epochs. ```torch.optim.lr_scheduler.ReduceLROnPlateau``` allows dynamic learning rate reducing based on some validation measurements.
* ```lr_scheduler.LambdaLR```
* ```lr_scheduler.MultiplicativeLR```
* ```lr_scheduler.StepLR```
* ```lr_scheduler.MultiStepLR```
* ```lr_scheduler.ExponentialLR```
* ```lr_scheduler.CyclicLR```
* ```lr_scheduler.OneCycleLR```
* ```lr_scheduler.CosineAnnealingLR```
* ```lr_scheduler.CosineAnnealingWarmRestarts```

### Sample code for ```lr_scheduler.StepLR```
```
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```
As the number of epochs increase by a step of 30, Learning rate is multiplied by a factor of 0.1 (```gamma```).

## 3. How are weights initialized?

Weight initialization plays a significant role not only in convergence of results but also in faster convergence of results. Below table shows the effect of different weight initializations

| Initialization Method        | Explanation           | Effects  |
|:----------------------------: |:---------------------:| :--------|
| Zero/ Constant values to weights | All the weights initialized to ```zero``` or ```some constant value C``` | * Initializing all the weights with zeros or constant leads all the neurons to learn the same features during training since all the neurons have identical gradients resulting in an inefficient model. All neurons evolve symmetrically instead of learning different features.|
| Too small values to weights | All the weights initialized with lower values such as 0.5  |  The gradients of the cost with respect to the parameters are too small, leading to convergence of the cost before it has reached the minimum value. Leads to vanishing gradients problem |
| Too large values to weights | All the weights initialized with larger values such as 1.5  |  the gradients of the cost with the respect to the parameters are too big. This leads the cost to oscillate around its minimum value. Leads to exploding gradients problem |

Inorder to overcome the vanishing and exploding gradient problem the general rules of thumb used are:
* 1. The **Mean** of activations should be ZERO.
* 2. The **Variance** of the activations should stay same across different layers.

For more information on mathematical and visualization aspects refer to this link https://www.deeplearning.ai/ai-notes/initialization/

### Observing how the Weights are initialized in Pytorch
Based on the type of the layer (Linear/Conv1D/Conv2D etc) the ranges of values considered for initializing Weights and Biases are different. Example
* Linear layer in Pytorch - refer this link for more details : https://pytorch.org/docs/master/generated/torch.nn.Linear.html#torch.nn.Linear
![image](https://user-images.githubusercontent.com/30425824/133922032-1dcd4863-79b4-4f92-a37d-e62ada2636c0.png)
* Conv1D layer - https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
![image](https://user-images.githubusercontent.com/30425824/133922106-a8a5a7d3-6b35-43fb-b971-10f1af8b3e74.png)

## 4. What is "loss" in a neural network?
Calculating Loss Function is one of the key steps in any machine learning process. 
Loss function helps to 
* Calculate how off the machine learning model predictions (```y_hat```) are from the actual values (```y_actual```).
* Partial differentiation of loss function with model parameters provides the direction and magnitude of correction that needs to be done on the paramters such that the model converges to minimum.
* Penalizes the model such that the model parameters are modified so as to make the model predictions closer to actual values after few iterations.

| Loss Value        | Indicates |
|:----------------: |:----------:|
|Higher Loss Value  | Model predictions ```y_hat``` is far from actual values ```y_actual```|
|Lower Loss Value | Model predictions ```y_hat``` is closer to actual values ```y_actual```|

There are several different types of loss functions and the choice of loss function depends on the problem the user is trying to solve. There is a flexibility to write user define loss functions. Some of the most commonly used loss functions are shown in below table.

|Type of Problem | Output Type | Final Activation Function | Loss Function | Pytorch Loss Function |Comments|
|:------------: |:----------:|:----------------: |:----------:|:----------------: |:-------------------------------------:|
| Regression | Numerical | Linear | Mean Squared Error (L2 loss); Mean Absolute Error (L1 loss) |```torch.nn.MSELoss```; ```torch.nn.L1Loss```;```torch.nn.HuberLoss```| L1 loss robust to outliers, L2 loss is more stable, Huber losss combines the benefits of L1 & L2 loss|
| Classification | Binary Outcome | Sigmoid | Binary Cross Entropy; Cross Entropy |```torch.nn.BCELoss```;```torch.nn.CrossEntropyLoss```|
| Classification | Single label, Multiple Classes | Softmax | Cross Entropy; Negative Log Likelihood loss |```torch.nn.CrossEntropyLoss```;```torch.nn.LLLoss```|In NLL, the model is punished for making the correct prediction with smaller probabilities and encouraged for making the prediction with higher probabilities. The logarithm does the punishment.|

Different loss functions that are readily availabe in pytorch could be found in this link: https://pytorch.org/docs/stable/nn.html#loss-functions


## 5. What is the "chain rule" in gradient flow?
The chain rule is essentially a mathematical formula that helps you calculate the derivative of a composite function. 
In Neural Networks, we use Chain rule to calculate the derivative of loss function with respect to parameters(weights) in optimization algorithm. 

Above diagram represents a NN with two hidden layers, having 3 and 2 neurons respectively. 
Forward Propagation: During the feed forward step of NN, predictions are made using input node values, weights and bias.
Mathematically:
  O = f (w*x + b)
Where O is the output of neuron, f is the activation function, w is weight, x is input, and b is bias.
The similar type of computation will occur at each neuron and the computation will propagate forward till the last neuron.




Back-Propagation: Once we finish the forward propagation, we get prediction at the output layer as O31 . With this output and the actual label, error/loss is calculated. 
 
Here L is loss, y is the predicted output and yhat is the actual output.
Our objective is to fine tune the network parameters to minimize the error term(cost function), in such a way that the predictions get nearer to actual labels.
 
In our neural network, in order to minimize the cost, we need to find the weight and bias values for which the loss/error returns the smallest value possible. The smaller the loss, the more accurate our predictions are. This is an optimization problem where we have to find the function minima. 
To find the minima of a function, we can use the gradient descent algorithm as shown in the above figure.
To calculate the derivative, we use chain rule. This is illustrated in the figure below.
  
 

