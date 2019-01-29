# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Constructr Multivariate Gaussian based on parameter
mu = [175, 42, 78]
cov = [
    [6, 2.68, 6.62],
    [2.68, 1.48, 3.07],
    [6.62, 3.07, 18]
]
mvn = multivariate_normal(mean=mu, cov=cov)

# Draw a sample
samples = mvn.rvs(100)
x, y, z = samples.T

# plot the samples
f, axs = plt.subplots(1,3)
axs[0].scatter(x, y)
axs[0].set_title("Height vs Shoe size")
axs[1].scatter(x, z)
axs[1].set_title("Height vs Weight")
axs[2].scatter(y, z)
axs[2].set_title("Shoe size vs Weight")
plt.savefig("./results/1-8-multivariate-gaussians.png");