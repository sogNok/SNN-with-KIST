import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy
import torch
import pygad
import subprocess
import sys

#from def_ecg import main_run

fitness = 0

def fitness_func(solution, solution_idx):
    #print(solution_idx, sys._getframe(3).f_code.co_name+'()')
    #numpy.save('solution', solution)
    #subprocess.call(['python', 'brecg.py', '--n_epoch=1'])
    #exec(open('test2.py').read())
    global fitness
    fitness += 1#main_run(solution, 2)
    
    #fitness = 100#torch.load('fitness')
    print(solution_idx, end='')

    return fitness#.item()

fitness_function = fitness_func

num_generations = 10 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 10 # Number of solutions in the population.
num_genes = 512 + 512 * 512

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print()
    best_solution = ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=best_solution))
    print("Change     = {change}".format(change=best_solution - last_fitness))
    last_fitness = best_solution

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       on_generation=on_generation)

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
