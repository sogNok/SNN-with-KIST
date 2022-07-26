from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import MNIST
import warnings
import os.path
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from classets import ECG, Outlayer, CNN, NN, ANN, reservoir

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.encoding import Encoder, NullEncoder, PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection

from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
    )

interV	= 1 #128 * 10 * 1
classN	= 6
trainN	= 10000
testN	= 4040
timeT   = 1280
batch_train = 500
batch_FC    = 64
batch_test  = 512
learning_rate = 0.001

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=95)
parser.add_argument("--n_neurons", type=int, default=512)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=timeT)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_workers = 4 * torch.cuda.device_count()
kwargs = {"num_workers": 0, "pin_memory": True} if torch.cuda.is_available() else {}

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Create simple Torch NN
network = Network(dt=dt)
inpt = Input(interV, shape=(1, interV))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52) #np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")

# For GA
#weights = np.load('solution.npy', allow_pickle=True)
#weights = torch.from_numpy(weights).float()
#c1_w, c2_w = weights[0:n_neurons].view(1, n_neurons), weights[n_neurons:].view(n_neurons, n_neurons)

C1 = Connection(source=inpt, target=output, w=torch.randn(inpt.n, output.n)) #0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=torch.randn(output.n, output.n)) #0.5 * torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

# Monitors for visualizing activity

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Get MNIST training images and labels.
# Load MNIST data.
train_dataset = ECG(
    root=os.path.join("..", "..",  "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=True
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_train, shuffle=False, **kwargs
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = trainN / batch_train
out_data = torch.zeros(timeT, trainN, n_neurons, device=device)
out_label = torch.zeros(trainN, dtype=torch.int64, device=device)

#pbar = tqdm(total=trainN,file=sys.stdout)
for (i, dataPoint) in enumerate(train_loader):
    #pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    #pbar.update(batch_train)
    
    print("Train progress: (%d / %d)" % (i, n_iters))
    if i > n_iters-0.001:
        pass
        #pbar.close()
        #break
    
    # Extract & resize the MNIST samples image data for training
    #       int(time / dt)  -> length of spike train
    #       28 x 28         -> size of sample
    datum = {"I": dataPoint[0].permute(1,0,2)}
    
    #exit(0)
    if gpu:
        datum = {k: v.cuda() for k, v in datum.items()}
    label = dataPoint[1].cuda()

    # Run network on sample image
    network.run(inputs=datum, time=time, input_time_dim=1)
    
    print(spikes["O"].get("s").shape)
    
    spikes_sum = spikes["O"].get("s")
    print('1')
    out_data = torch.cat([out_data,spikes_sum], dim=1)
    print('2')
    out_label = torch.cat([out_label,label], dim=0)

    network.reset_state_variables()
    if gpu:
        datum = {k: v.cpu() for k, v in datum.items()}

out_data = out_data[:, 1:,].unsqueeze(2)
out_label = out_label[1:]
out_dataset = Outlayer(out_data, out_label)


print(out_data.shape)
print(out_label.shape)
'''
print(training_pairs[0][0][0][0:128])
print(training_pairs[0][0][0][128:256])
print(training_pairs[0][0][0][256:384])
print(training_pairs[0][0][0][384:512])
'''
# Create and train logistic regression model on reservoir outputs.
model = CNN(n_neurons, classN).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, momentum=0.5)

torch.set_printoptions(precision=2)
# Training the Model
print("\n Training the read out")
for epoch in range(n_epochs):
    avg_loss = 0
    out_loader = torch.utils.data.DataLoader(
    out_dataset, batch_size=batch_FC, shuffle=True, **kwargs
    )

    for (i, dataPoint) in enumerate(out_loader):
        X, Y = dataPoint
        
        # Reset gradients to 0
        optimizer.zero_grad()

        X = X.to(device, dtype=torch.float)
        Y = Y.to(device)

        # Run spikes through logistic regression model
        outputs = model(X)
        #print('x', outputs)#.sum(dim=1))
        #print('y', Y)

        # Calculate MSE
        loss = criterion(outputs, Y)
            # Optimize parameters
        loss.backward()
        optimizer.step()
     
        #print(loss)
        avg_loss += loss / trainN

   
    print("Epoch: {0}/{1}, Loss: {2:>.9}".format(epoch+1, n_epochs, avg_loss))
#exit(0)
# Run same simulation on reservoir with testing data instead of training data
# (see training section for intuition)

test_dataset = ECG(
    root=os.path.join("..", "..",  "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_test, shuffle=True, **kwargs
)


n_iters = testN / batch_test
out_data = torch.zeros(1, n_neurons, device=device)
out_label = torch.zeros(1, dtype=torch.int64, device=device)

#pbarr = tqdm(total=testN, file=sys.stdout)
for (i, dataPoint) in enumerate(test_loader):
    #pbarr.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))
    #pbarr.update(batch_test)
    print("Train progress: (%d / %d)" % (i, n_iters))

    if i > n_iters-0.001:
        pass
        #pbarr.close()
        #break
    
    datum = {"I": dataPoint[0].permute(1,0,2)}
    if gpu:
        datum = {k: v.cuda() for k, v in datum.items()}
    label = dataPoint[1].cuda()

    # Run network on sample image
    network.run(inputs=datum, time=time, input_time_dim=1)
   
    spikes_sum = spikes["O"].get("s").sum(0)
    out_data = torch.cat([out_data,spikes_sum], dim=0)
    out_label = torch.cat([out_label,label], dim=0)

    network.reset_state_variables()

out_data = out_data[1: ,].unsqueeze(1).cpu()
out_label = out_label[1:].cpu()
out_dataset = Outlayer(out_data, out_label)
out_loader = torch.utils.data.DataLoader(
    out_dataset, batch_size=batch_test, shuffle=False, **kwargs    
)

accuracy = 0
# Test model with previously trained logistic regression classifier
for (i, dataPointer) in enumerate(out_loader):
    X_test, Y_test = dataPointer
    
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    prediction = model(X_test)
    
    
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    #print('x', prediction)
    #print('arg x', torch.argmax(prediction,1))
    #print('y', Y_test)
    accuracy += correct_prediction.sum()

print(f'Accuracy: {accuracy} / {testN}')
print(f'Acuuracy: {accuracy / testN}')

torch.save(accuracy, 'fitness')

