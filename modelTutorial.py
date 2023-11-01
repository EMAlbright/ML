#%%
import torch
from torch import nn #nn contains building blocks for neural networks
import matplotlib.pyplot as plts
# PyTorch WorkFlow (01)

"""
What we covering:
1. Data preparing and loading
3. Build model
3. Fit model to the data (training)
4. Making predictions and evaluating a mode (inference)
5. Saving and loading a model
6. Save and reload trained model
"""
# 1. Data Preparing and Loading : Data can be anything in ML

# 2 parts of ML : Numerically encode data, Build model to learn patterns

# Use linear regression to make straight line (y= mx + b)

# Create known parameters
weight = 0.7  #b
bias = 0.3    #a

#Create 
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]

# Splitting data into training and test sets (one of most important concepts)

# 3 datasets: Training set (Course materials), Validation set (Practice exam), Test set (Final exam)

#Create a train test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test) # 40 40 10 10
#%%
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    plts.figure(figsize=(10, 7))

    #Plot training data in blue
    plts.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    #Plot test data in green
    plts.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        #Plot predictions if they exist
        plts.scatter(test_data, predictions, c="r", s=4, label="Predictions")

        #show legend
        plts.legend(prop={"size": 14})
plot_predictions()
#%%
# 2. Build model
# What the model does:
"""
Start with random values (weight and bias)
Look at training data and adjust the random values to better 
represent  the ideal values (weight and bias we used to create data)

Does so through 2 algorithms:
- Gradient Descent - Adjusts weights based on how far off prediction was from 
actual value of Y

- Back Propagation - Uses gradient descent but also takes into account error

"""
# Create a linear regression model class
class LinearRegressionModel(nn.Module):
    #nn.module is the class for neural networks

    #Requires grad means pyTorch tracks the gradients of specific parameter
    # Does automatically
    def __init__(self):
        super().__init__()

        #Initialize model parameters
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=None,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=None,
                                             dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # X is input data
       return self.weights * x + self.bias  # Linear regression formula
#Any subclass of nn.module needs to override forward
#%%
#Pytorch model building essentials
"""
torch.nn - computational graphing (neural networks can be considered a graph)
torch.nn.Parameter - what parameters should our model try and learn, often
a pytorch later from torch.nn will set these for us
torch.nn.Module - base class for all neural network modules
torch.optim - where the optimizers in pytorch live, help with gradient descent
def forward() - all nn.module subclasses require you to overwrite forward(),
this method defines what happens in forward computation
"""

#Check whats in our model using .parameters

#Create a random seed
torch.manual_seed(42)

#Create an instance of the model (subclass of nn.Module)
model_0 = LinearRegressionModel()

list(model_0.parameters())
#%%
#List named parameters
model_0.state_dict()

#move the 2 random values to our ideal values (weight=0.7, bias=0.3)

#Making predictions using torch.inference_mode()
#Lets see predictioni of y test based on x test
#When we pass data through the model, its going to run through forward() method

with torch.inference_mode():
    y_preds = model_0(X_test)

#Not implemented error: forward method not in line with class method
plot_predictions(predictions=y_preds)
#%%
# 3. Train model


# %%
# Loss function : function to measure how wrong 
# models prediction is to ideal output, lower is better

# Optimizer: Takes into account the loss of model and adjusts model
# parameters (weight + bias in our case) to improve loss function

# For pytorch, need a TRAINING loop and a TESTING loop

#setup loss function
loss_fn = nn.L1Loss()

#setup optimizer : SGD most common
# smaller the lr, the smaller the change in the parameter
# larger the lr, the larger the change in the parameter
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) #lr - learning rate = most important hyperparameter you set

# Building a training loop (and testing loop)
"""
0. Loop through data
1. Forward pass (data moving through model function forward() - forward propagation)
2. Calculate loss (compare forward pass predictions to ground truth labels)
3. Optimize zero grad
4. Loss backward
5. Optimizer step - use optimizer to adjust models parameters to minimize loss

5 = gradient descent, 4 = back propagation  (gradient is slope)
"""
#epochs is one loop through the data
torch.manual_seed(42)

epochs = 100
epoch_count = []
train_loss_values = []
test_loss_values = []

# 0. Loop through data
for epoch in range(epochs):
     #Set model to training mode
     model_0.train() # train mode in pyTorc

#1. Forward pass
     y_pred = model_0(X_train)

#2. Calculate loss
     loss = loss_fn(y_pred, y_train)
     
#3. optimizer zero grad
     optimizer.zero_grad
#4. Perform backpropagation on loss with respect to parameters of model
     torch.Tensor.backward
    
#5. step optimizer (perform gradient descent)
     optimizer.step()

     model_0.eval() # turn off gradient tracking
     
     with torch.inference_mode():
        #1. Do the forward pass
        test_pred = model_0(X_test)

        #2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)
# Print whats happening
if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    #Print states
    print(model_0.state_dict())

# Plot the loss curves
plts.plot(epoch_count, train_loss_values, label="Train loss")
plts.plot(epoch_count, test_loss_values, label="Test loss")
plts.title("Training and test loss curves")
plts.ylabel("Loss")
plts.xlabel("Epochs")
plts.legend()
#%%
 #create data using linear regression formula
weight = .7
bias = .3

start = 0
end = 1
step = .02

# create x and y
X = torch.arange(start, end, step).unsqueeze(dim=1)
y =weight * X + bias
X[:10], y[:10]

# Create data (same as everything above but again) in
# a more common way using torch.nn (dont manually type as much as above)

train_split = int(.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)

plot_predictions(X_train, y_train, X_test, y_test)

#Create linear model by sublassing nn.Module
class LinearRegressionModelV2(nn.Module):
     def __init__(self):
        super().__init__()
        #use nn.Linear() for creating parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
     def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()

# TRAINING: Loss function, Optimizer, Training loop, Testing loop

#Setup loss function
loss_fn = nn.L1Loss()

#Setup optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=.01)

# Training loop
torch.manual_seed(42)

epochs = 200

for epoch in range(epochs):
    model_1.train()

    #1. forward pass
    y_pred = model_1(X_train)

    #2. calculate loss
    loss= loss_fn(y_pred, y_train)

    #3. optimizer zero grad
    optimizer.zero_grad()

    #4. perform back propagation
    loss.backward()

    #5. optimizer step
    optimizer.step()

    #Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

     #print whats happening

if epoch % 10 == 0:
     print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
#%%