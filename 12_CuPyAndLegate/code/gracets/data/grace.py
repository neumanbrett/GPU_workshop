import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Pull in the time series data
data = np.loadtxt("/glade/u/home/bneuman/codes/12_CuPyAndLegate/code/gracets/data/grace_raw.o")

# Plot TS
x = data[:,0]
y = data[:,1]
ts = plt.plot(x, y)
plt.xlabel("Years")
plt.ylabel("GRACE Annual Amplitude (mm)")

# Perform DFFT (cpu)
# ---
# Transfer data to GPU
# Stream setup
# Perform DFFT (gpu)
# Plot DFFT
# Plot performance differences