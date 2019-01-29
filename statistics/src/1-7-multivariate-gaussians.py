# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Config
os.chdir("/home/jovyan/work/")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
# Create Distributions
mvn_1 = multivariate_normal(mean=[.5, -.2], cov=[[2, .3], [.3, .5]])
mvn_2 = multivariate_normal(mean=[-2.5, 1.8], cov=[[1, -.9], [0, .1]])
mvn_3 = multivariate_normal(mean=[-2.5, -.5], cov=[[.5, .3], [.9, 2]])

# Draw Samples
n_samples = 3000
sample_1 = mvn_1.rvs(n_samples)
sample_2 = mvn_2.rvs(n_samples)
sample_3 = mvn_3.rvs(n_samples)

# Plot
plt.scatter(*sample_1.T, color="red", alpha=.25, marker=".")
plt.scatter(*sample_2.T, color="blue", alpha=.25, marker=".")
plt.scatter(*sample_3.T, color="green", alpha=.25, marker=".")
plt.savefig("./results/1-7-multivariate_gaussians.png");