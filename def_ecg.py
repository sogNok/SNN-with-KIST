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
import pygad

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
learning_rate = 0.0005

seed = 95
n_neurons = 512
n_epochs = 100
examples = 500
n_workers = -1
time = timeT
dt = 1.0
intensity = 64
progress_interval = 10
update_interval = 250
train = True
plot = False
gpu = True

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_workers = 4 * torch.cuda.device_count()
kwargs = {"num_workers": n_workers, "pin_memory": True} if torch.cuda.is_available() else {}

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

count = 13

def fitness_func(solution, solution_idx):
    print(f'G{count+1}-{solution_idx}')
    # Create simple Torch NN
    network = Network(dt=dt)
    inpt = Input(interV, shape=(1, interV))
    network.add_layer(inpt, name="I")
    output = LIFNodes(n_neurons, thresh=-52) #np.random.randn(n_neurons).astype(float))
    network.add_layer(output, name="O")

    # For GA
    #weights = np.load('solution.npy', allow_pickle=True)
    weights = torch.from_numpy(solution).float()
    c1_w, c2_w = weights[0:n_neurons].view(1, n_neurons), weights[n_neurons:].view(n_neurons, n_neurons)

    C1 = Connection(source=inpt, target=output, w=c1_w)#torch.randn(inpt.n, output.n)) #0.5 * torch.randn(inpt.n, output.n))
    C2 = Connection(source=output, target=output, w=c2_w)#torch.randn(output.n, output.n)) #0.5 * torch.randn(output.n, output.n))

    network.add_connection(C1, source="I", target="O")
    network.add_connection(C2, source="O", target="O")

# Monitors for visualizing activity
    spikes = {}
    for l in network.layers:
        spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
        network.add_monitor(spikes[l], name="%s_spikes" % l)

    voltages = {"O": Monitor(network.layers["O"], ["v"], time=time, device=device)}
    network.add_monitor(voltages["O"], name="O_voltages")

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
    out_data = torch.zeros(1, n_neurons, device=device)
    out_label = torch.zeros(1, dtype=torch.int64, device=device)

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
        
        #print(spikes["O"].get("s").sum(0).shape)
        
        spikes_sum = spikes["O"].get("s").sum(0)
        out_data = torch.cat([out_data,spikes_sum], dim=0)
        out_label = torch.cat([out_label,label], dim=0)

        network.reset_state_variables()

    out_data = out_data[1: ,].unsqueeze(1).cpu()
    out_label = out_label[1:].cpu()
    out_dataset = Outlayer(out_data, out_label)

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
    
    return accuracy.item()

fitness_function = fitness_func

num_generations = 87 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 50 # Number of solutions in the population.
num_genes = 512 + 512 * 512

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    global count
    count += 1

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
    ga_instance.save(f'./sol2/utionG{count}')

gas = pygad.load('./sol2/utionG13')
init_pop = gas.population
# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       initial_population=init_pop,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       keep_parents=1,
                       save_best_solutions=True,
                       on_generation=callback_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'real_result' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)
