from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import MNIST
import warnings
import os.path
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from classets import ECG, Outlayer, CNN, NN, ANN

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
batch_train = 512
batch_FC    = 64
batch_test  = 512
learning_rate = 0.001

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=512)
parser.add_argument("--n_epochs", type=int, default=300)
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

print(f'seed: {seed}')
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_workers = 4 * torch.cuda.device_count()
kwargs = {"num_workers": n_workers, "pin_memory": True} if torch.cuda.is_available() else {}

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Load data.
train_dataset = ECG(
    root=os.path.join("..", "..",  "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=True
)
test_dataset = ECG(
    root=os.path.join("..", "..",  "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
    train=False
)

# Create and train logistic regression model on reservoir outputs.
model = CNN(n_neurons, classN).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, momentum=0.5)

torch.set_printoptions(precision=2)

# Create simple Torch NN
network = Network(dt=dt)
inpt = Input(interV, shape=(1, interV))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, reset=-60, thresh=-52)
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))
C3 = Connection(source=inpt, target=output, w=torch.ones(inpt.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")
network.add_connection(C3, source="I", target="O")

# Monitors for visualizing activity
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time, device=device)}
network.add_monitor(voltages["O"], name="O_voltages")

weights = C2.w
print(torch.randn(inpt.n, output.n).shape)
print(torch.randn(output.n, output.n).shape)
print(C2.w.shape)
