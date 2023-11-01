"""
CHAPTER 2: NEURAL NETWORK CLASSIFICATION

BINARY CLASSIFICATION: is it one thing or another (split into 2)
MULTICLASS CLASSIFICATION: is more than one thing or another (photo of a cat, dog, snake, lettuce, etc.)
MULTILABEL CLASSIFICATION: multiple things for one thing 
"""
#%%
import torch
import sklearn
from sklearn.datasets import make_circles

n_samples = 1000
#Random state is like random seed

X, y = make_circles(n_samples,
                    noise=.03,
                    random_state=42)
len(X), len(y)
print(f"First 5 samples of X {X[:5]}")
print(f"First 5 samples of y {y[:5]}")
print(y)
#%%

#Make dataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[:, 1],
                        "label": y})
circles.head(10)
#%%

#Visualize, visualize, visualize
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)

#Data we are working with is refered to as a
# toy dataset, small enough to experiment on, but
# sizable enough to practice with fundamentals

# %%

### 1.1 CHECK INPUT AND OUTPUT SHAPES
X.shape, y.shape
# %%

#View first example of features and labels
X_sample = X[0]
y_sample = y[0]
# 2 features of x predicting 1 y value
print(f"Values of X and y for one sample {X_sample} for y: {y_sample}")

# 1.2 TURN DATA INTO TENSORS AND CREATE TRAIN/TEST SPLITS
#%%
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]

# %%
#SPLIT data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% of data will be test, 80% will be train)
                                                    random_state=42)
len(X_train), len(X_test), len(y_train), len(y_test)
#%% #Trains have 800 (so model can learn patterns)
    #Test has 200 (so model can improve training)

# 2. BUILDING A MODEL
"""
Build model to classify blue and red dots
Steps:
1. Setup device agnostic code so our code runs on GPU if there is one
2. Construct a model (by subclassing 'nn.Module')
3. Define a loss function and optimizer
4. Create a training and test loop
"""
# 1
#%%
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
device
#%%
"""
2.1 Subclass 'nn.Module' (almost all models sublclass this)
2.2 create 2 'nn.Linear()' layers capable of handling shapes of our data
2.3 Defines a 'forward()' method to outline forward pass of model
2.4 Instantiate an instance of model class and send to the target device
"""
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        #2 create nn.Linears
        # takes in 2 features and upscale to 5 features
        # 2nd layer must take in first layers output (outputs 1 which is same shape as 7)
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    
    # 3. define a forward method that outlines forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # X -> layer1 -> layer2 -> output
#4. instantiate instance of model class and send to device
model_0 = CircleModelV0().to(device)
model_0
#%%

# REPLICATE THE ABOVE (starting from class CircleModeV1) using nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_0
# %%

#Make some predictions
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predicitions: {untrained_preds[:10]}")
print(f"\nFIrst 10 labels: {y_test[:10]}")
# %%

# setup loss function and uptimizer
"""
For regression (picking number) MAE (Mean Absolute Error)
For classification (this one) binary cross entrapy or categorical cross entrapy

Loss function measures how wrong models predictions are
For optimizers, 2 of most common are SGD and Adam
"""
loss_function = nn.BCEWithLogitsLoss() # has sigmoid activation function built in
#%%
#sigmoid activation for non linear data (images, NL, etc)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

#calculate accuracy
def accuracy_fn(y_true, y_pred):
     correct = torch.eq(y_true, y_pred).sum().item()
     acc = (correct / len(y_pred)) * 100
     return acc

# Training Loop
"""
1. Forward pass
2. Calculate loss
3. Optimizer zero grad
4. Loss backward (backpropagation)
5. Optimizer (gradient descent)
"""

# Logits - raw outputs of model

#3.1: Logits -> prediction probabilities -> prediction labels

# We convert logits to prediction probabilities by passing through
# activation function (sigmoid for binary classification (like now)
# and softmax for multiclass classification). Then convert to labels.

#First 5 logits
#%%

y_logits = model_0(X_test.to(device))[:5]
y_logits
    
# %%

y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs

# %%
#Find predicted labels 
y_preds = torch.round(y_pred_probs)

#in full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

#Get rid of extra dimension
y_preds.squeeze()
# %%

y_test[:5]
#%%
# 3.2 Building and Training Testing Loop

torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_function(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_function(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:}")
#%%

# 4. Make predictions and evaluate model
# Looks like model is not learning anything so far
# Make them VISUAL! VISUALIZE!

import requests
from pathlib import Path
# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("Function already exists, skipping download")
else:
    print("Downloading")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundary of model
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
#%%

#5. Improving the model
"""
Different ways:
- Add more layers (give model more chances to learn patterns)
- Add more hidden units - go from 5 hidden units to 10 hidden units
  (more parameters means the same as above)
- Fit for longer (more epochs)
- Changing the activation functions (sigmoid for us)
- Change the learning rate
- Change the loss function

Hidden units: 5 to 10
Number of layers: 2 to 3
Number of epochs: 100 to 1000
"""
# Out is 1 because just 1 output (y)

#%%
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        #these 3 are the same as the return, speeds up code (less code)
       # z = self.layer_1(x)
        #z = self.layer_2(z)
        #z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_1 = CircleModelV1().to(device)
model_1

#More neurons and better represents
#%%
model_0.state_dict()

model_1.state_dict()
# %%

#Create loss function
loss_fn = nn.BCEWithLogitsLoss()
#Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

#Training and Evaluation Loop 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_1.train()

    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")
# %%
#%%
#Plot decision boundary
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
# %%

# One way to troubleshoot a large problem is to test smaller problem

#Create some data (same as linear regression model)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = X_regression * weight + bias

#Create train and Test splits
train_split = int(0.8 * len(X_regression))  # 80% train
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
#%%
#%%
# Adjusting model 1 to fit a straight line
# Same architecture as model 1 
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_2
# %%
# Loss and Optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

# Train Model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set epochs
epochs = 1000

#Put data on target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss: .2f}")
# %%

#Turn on eval mode
model_2.eval()

with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# Plot data
plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression,
                 predictions=y_preds.cpu())
# %%

# Missing piece is non - linearity, only have been using that thus far
# RECREATING non-linear data
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples=1000,
                    noise=0.03,
                    random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# %%
#Convert data to tensors and train/test splits

import torch
from sklearn.model_selection import train_test_split
#Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#Split into train / test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
#%%
#Build a model with non linear activation functions
from torch import nn
class CircleModeV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #Where to put our non linear activation function (ReLU)
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModeV2().to(device)
model_3

loss_fn = nn.BCEWithLogitsLoss()
#Create optimizer
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.1)

#Training and Evaluation Loop 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_3.train()
    
    #This line is forward pass
    y_logits = model_3(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        #This line is forward pass
        # Need to squeeze so that test logits squeezed to 1
        # Before its at 200
        test_logits = model_3(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# %%

#6.4 : VISUALIZE VISUALIZE VISUALIZE
# Evaluating non linear model above (acc went from 50 to 75%)

model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

#Plot decision boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
#Model 1 = no non-linearity
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
#Model 3 = non-linearity
plot_decision_boundary(model_3, X_test, y_test)
# %%

A = torch.arange(-10, 10, 1, dtype=torch.float32)
A.dtype
# %%

#Visualize tensor
plt.plot(A) #Straight line

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0), x) # inputs must be tensors
relu(A)

# %%

#Plot ReLU activation function
plt.plot(relu(A))
# %%
# Now do the same for sigmoid
def sigmoid(x):
    return 1 / 1 + torch.exp(-x)

plt.plot(torch.sigmoid(A))
# %%

# SECTION 8: Putting it all together with a multi-class classification
"""
Binary classficiation : one thing or another (fraud vs not, cat vs dog)
MultiClass classification: Multiple things against eachother (pizza v cake v spaghetti)
"""

# 8.1 Creating a toy multi-class dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)
# 2. turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into test and train splits

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. plot data (visualize)
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

# %%

# 8.2 : Building a multi class classification model

#Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
#Build a multi class model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
    
       super().__init__()
       self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
       )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

model_4
# %%

# NOW, loss function and training loop

# 8.3 : Loss function and optimizer
#%%
loss_fn = nn.CrossEntropyLoss() # For multi class classification

#Optimizer
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

# 8.4 : Prediction probabilities for multi class
model_4(X_blob_train.to(device))[:5]
# %%
y_logits[:10]
# %%

# Logits (raw output of model) -> Prediction Probabilities (torch.softmax activation function)-> 
# (take argmax of pred probabilities) Prediction Labels

y_logits = model_4(X_blob_test.to(device))
# Change raw data to prediction probabilities through activation function
# ABOVE used sigmoid activation function

y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
# %%
print(y_pred_probs[:5])

# %%
y_blob_test

# %%
# 8.5 : Create training loop and testing loop

torch.manual_seed(42)

epochs = 100

X_blob_train, y_blob_train, = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                            y_pred=test_pred)

    if epoch % 10 == 0:
       print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")
# %%
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
# GO FROM LOGITS -> PREDICTION PROBABILITIES
y_pred_probs = torch.softmax(y_logits, dim=1)

#Plot decision boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
#Model 1 = no non-linearity
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
#Model 3 = non-linearity
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
# %%

# See TORCH METRICS for many pyTorch metrics (scikit also)