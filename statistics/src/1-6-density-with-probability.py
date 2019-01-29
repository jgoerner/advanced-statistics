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

# Plot
x = np.linspace(120, 220, 100)
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(x, norm(loc=175, scale=10).pdf(x), linewidth=5)
plt.fill_between(x[x < 180], 0, norm(loc=175, scale=10).pdf(x[x < 180]), alpha=.5)
plt.savefig("./results/1-6-normal-density-with-probability.png");