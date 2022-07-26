import warnings
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from classets import NN, ANN, CNN, ECG, TorchvisionDatasetWrapper

import argparse
import numpy as np

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)

interV	= 128 * 4 #* 4
epochN  = 5
trainN	= 2000 # 60000 # 600 # 2000
testN	= 340 # 7938 # 240 # 340
neuronN = 512
timeT   = 150
batch_size = 16
interval = 1

DT = 'svdb'
TR = 'training_fft_1r.pt' # './training_fft_slide_3.pt # './training_fft_2n2.pt' # './training_fft_1r.pt' # './training_fft_slide_1.pt'
TE = 'test_fft_1r.pt' # './test_fft_slide_3.pt' # './test_fft_2n2.pt' # './test_fft_1r.pt' # './test_fft_slide_1.pt'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=neuronN)
parser.add_argument("--n_epochs", type=int, default=epochN)
parser.add_argument("--n_test", type=int, default=testN)
parser.add_argument("--n_train", type=int, default=trainN)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=timeT)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=50)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=interV,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm= interV / 10.0,
    theta_plus=theta_plus,
    inpt_shape=(1, interV),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = TorchvisionDatasetWrapper(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "ECG"),
    data_type=DT,
    TR=TR,
    TE=TE,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    ),
)


# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)

#inh_voltage_monitor = Monitor(
#    network.layers["Ai"], ["v"], time=int(time / dt), device=device
#)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
#network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)


# Train the network.
print("\nBegin training.\n")
start = t()
labels = []
for epoch in range(n_epochs):
    print("train:", epoch+1)
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].permute(1, 0, 2).unsqueeze(dim=2)}
        #print(inputs["X"].shape)
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        network.reset_state_variables()  # Reset state variables.


    network.save(f"network_N{neuronN}_T{timeT}_{TR}")

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")


