import os
import torch
import pygad
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t
from classets import ECG, Outlayer, CNN, NN, ANN, reservoir

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_assignments,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=96)
parser.add_argument("--n_neurons", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=4040)
parser.add_argument("--n_train", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=25)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=1280)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu

update_interval = update_steps * batch_size

device = "cpu"
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

count = 0

def fitness_func(solution, solution_idx):
    print(f'G{count+1}-{solution_idx}')

    weights = torch.from_numpy(solution).float()
    c1_w, c2_w = weights[0:n_neurons].view(1, n_neurons), weights[n_neurons:].view(n_neurons, n_neurons)

# Build network.
    network = reservoir(
        n_inpt=1,
        w_inpt=c1_w,#torch.randn(1, 512),
        n_liquid=512,
        w_liquid=c2_w,#torch.randn(512, 512),
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        norm=51.2,
        nu=(1e-4, 1e-2),
        theta_plus=theta_plus,
        inpt_shape=(1, 1),
    )

# Directs network to GPU
    if gpu:
        network.to("cuda")

# Load MNIST data.
    dataset = ECG(
        root=os.path.join("..", "..",  "data", "ECG", "TorchvisionDatasetWrapper", "processed"),
        train=True
    )

# Neuron assignments and spike proportions.
    n_classes = 6
    assignments = -torch.ones(n_neurons, device=device)
    proportions = torch.zeros((n_neurons, n_classes), device=device)
    rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
    accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
    exc_voltage_monitor = Monitor(
        network.layers["Ae"], ["v"], time=int(time / dt), device=device
    )
    inh_voltage_monitor = Monitor(
        network.layers["Ai"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"I", "X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
        )
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None

    spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
    print("\nBegin training.\n")
    start = t()

    for epoch in range(n_epochs):
        labels = []

        if epoch % progress_interval == 0:
            print("\n Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
            start = t()

        # Create a dataloader to iterate and batch data
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=gpu,
        )

        pbar_training = tqdm(total=n_train)
        for step, batch in enumerate(train_dataloader):
            if step > n_train:
                break
            # Get next input sample.
            inputs = {"I": batch[0]}#.permute(1,0,2)}
            #print(inputs["I"].shape)
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            if step is n_train - 1 or step % update_steps == 0 and step > 0:
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels, device=device)

                # Get network predictions.
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )

                # Compute network accuracy according to available classification strategies.
                accuracy["all"].append(
                    100
                    * torch.sum(label_tensor.long() == all_activity_pred).item()
                    / len(label_tensor)
                )
                accuracy["proportion"].append(
                    100
                    * torch.sum(label_tensor.long() == proportion_pred).item()
                    / len(label_tensor)
                )

                print(
                    "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                    % (
                        accuracy["all"][-1],
                        np.mean(accuracy["all"]),
                        np.max(accuracy["all"]),
                    )
                )
                print(
                    "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                    " (best)\n"
                    % (
                        accuracy["proportion"][-1],
                        np.mean(accuracy["proportion"]),
                        np.max(accuracy["proportion"]),
                    )
                )

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            labels.extend(batch[1].tolist())

            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1)

            # Add to spikes recording.
            s = spikes["Ae"].get("s").permute((1, 0, 2))
            spike_record[
                (step * batch_size)
                % update_interval : (step * batch_size % update_interval)
                + s.size(0)
            ] = s
            #print(s.size(0))
            #print(s.shape)
            # Get voltage recording.
            exc_voltages = exc_voltage_monitor.get("v")
            inh_voltages = inh_voltage_monitor.get("v")

            network.reset_state_variables()  # Reset state variables.
            pbar_training.update(batch_size)

    print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
    print("Training complete.\n")

    return accuracy["all"][-1]

fitness_function = fitness_func

num_generations = 100 # Number of generations.
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
    ga_instance.save(f'./sol3/utionG{count}')

#gas = pygad.load('./sol2/utionG13')
#init_pop = gas.population
# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       #initial_population=init_pop,
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

