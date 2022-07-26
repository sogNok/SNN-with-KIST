#Importing Library

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import os.path
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from classets import ECG, CNN

print("init model done")

# Set Hyper parameters and other variables to train the model.

learning_rate = 0.001
training_epochs = 10
batch_size = 100
test_batch_size = 1000
no_cuda = False
seed = 1

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

n_workers = 4 * torch.cuda.device_count()
kwargs = {'num_workers': n_workers, 'pin_memory': True} if use_cuda else {}

print("set vars and device done")

# Load MNIST data.
train_dataset = ECG(
    root=os.path.join("..", "..", "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=True,
)

# Load MNIST data.
test_dataset = ECG(
    root=os.path.join("..", "..", "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=False,
)

#Prepare Data Loader for Training and Validation
train_loader = data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle=True, **kwargs)

test_loader = data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)

model = CNN(1280, 6).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in train_loader:        
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.permute(0,2,1).to(device, dtype=torch.float)
        Y = Y.to(device)


        optimizer.zero_grad()
        hypothesis = model(X)
        #print('x', hypothesis)
        #print('y', Y)
        #print(hypothesis.shape,":", Y.shape)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

model.eval()
with torch.no_grad():
    X_test = test_dataset.data.view(len(test_dataset), 1, 1280).float().to(device)
    Y_test = test_dataset.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
