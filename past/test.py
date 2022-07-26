import torch
import subprocess
import numpy
import pygad

def fitness_func():
    pass
def callback_generation():
    pass


filename='./sol2/utionG27'
ga = pygad.load(filename=filename)
solution, a, b = ga.best_solution(ga.last_generation_fitness)
print(ga.__dict__)
#print(ga.best_solution_generation)
#print(solution)
