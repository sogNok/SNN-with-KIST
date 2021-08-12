import torch
import subprocess
import numpy

#subprocess.call(['python', 'test2.py'])


a = numpy.load('solution.npy', allow_pickle=True)

print(a.shape)
