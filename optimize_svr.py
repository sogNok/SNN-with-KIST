import warnings

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classets import NN, ANN, CNN, ECG, TorchvisionDatasetWrapper

import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.encoding import PoissonEncoder
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)

from sklearn.svm import SVR

interV	= 128 * 4 # * 4
trainN	= 60000 # 600 # 60000 # 2000
testN	= 7938 # 240 # 7938 # 340
neuronN = 6144
timeT   = 600
batch_size = 64
interval = 1

TR = 'training_fft_slide_3.pt' # './training_fft_slide_3.pt # './training_fft_2n2.pt' # './training_fft_1r.pt' # './training_fft_slide_1.pt'
TE = 'test_fft_slide_3.pt' # './test_fft_slide_3.pt' tr# './test_fft_2n2.pt' # './test_fft_1r.pt' # './test_fft_slide_1.pt'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=neuronN)
parser.add_argument("--n_epochs", type=int, default=1)
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

network = load(f"network_N{neuronN}_T{timeT}_{TR}")

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = TorchvisionDatasetWrapper(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "ECG"),
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


# Inference optimizing the network
print("\nInference caculating.\n") 
network.train(mode=False)

start = t()

X_data = np.zeros((trainN, neuronN))
labels = np.zeros((trainN))

for epoch in range(1):
    print("infer:", epoch+1)
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
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        #print(inputs["X"].sum())
        Y = batch["label"]
        
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        X = spikes["Ae"].get("s").permute(1, 0, 2).sum(dim=1)
        
        index = (step+1) * batch_size
        X_data[index - batch_size:index if index < trainN else trainN] = X.cpu().numpy()
        labels[index - batch_size:index if index < trainN else trainN] = Y.cpu().numpy()

        network.reset_state_variables()  # Reset state variables.


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Caculating complete.\n")

print(X_data.shape)
print(labels.shape)

model = SVR()
#X_data = np.concatenate((X_data, X_data), axis=0)
#labels = np.concatenate((labels, labels), axis=0)
model.fit(X_data, labels)
relation_square = model.score(X_data, labels)

# Load MNIST data.
test_dataset = TorchvisionDatasetWrapper(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "ECG"),
    TR=TR,
    TE=TE,
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor()]#, transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
mse = 0
mae = 0
mape = 0

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
start = t()

with torch.no_grad():

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )


    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_test:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].permute(1, 0, 2).unsqueeze(dim=2)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        X = spikes["Ae"].get("s").permute(1, 0, 2).sum(dim=1).cpu().numpy()
        Y = batch["label"].cpu().numpy()
        
        hypothesis = model.predict(X)
        mse += np.power(hypothesis - Y, 2).sum()
        mae += np.abs(hypothesis - Y).sum()
        mape += (np.abs(hypothesis - Y) / Y).sum()

        network.reset_state_variables()  # Reset state variables.
        #pbar.set_description_str("Test progress: ")
        #pbar.update()

mse /= testN
mae /= testN
mape /= testN
mape *= 100

print("\nMSE: %.4f" % (mse))
print("\nRMSE: %.4f" % (np.sqrt(mse)))
print("\nMAE: %.4f" % (mae))
print("\nMAPE: %.4f" % (mape))
print("\nR: %.4f" % (relation_square)) 

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

