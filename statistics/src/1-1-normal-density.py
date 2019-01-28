# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Config
os.chdir("/home/jovyan/work/")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# actual plot
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x))
plt.savefig("./results/1-1-normal-density.png")