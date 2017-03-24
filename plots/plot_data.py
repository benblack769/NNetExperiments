import matplotlib.pyplot as plt
import sys
import numpy as np

assert len(sys.argv) == 4, "needs 3 additional command line arguments"

filename = sys.argv[1]
use_every = int(sys.argv[2])
first_vecs = int(sys.argv[3])


with open(filename) as file:
    file_lines = file.readlines()

datalist = []
times = []
for line in file_lines:
    vals = [float(val) for val in line.split()]
    datalist.append(np.array(vals[1:]))
    times.append(vals[0])
use_data = datalist[::use_every]
use_times = times[::use_every]
file_data = list(np.transpose(np.vstack(use_data)))
first_data = file_data[:first_vecs]


time = np.array(use_times)
for d in first_data:
    plt.plot(time,d)
plt.show()
