# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare distributions
N = [norm(loc=0, scale=0.5), norm(loc=1.5, scale=0.7)]
x = np.linspace(-2, 5, 200)

# Plot
cmap = ["b", "r"]
n_samples = 20
for n, c in zip(N, cmap):
    plt.plot(x, n.pdf(x), color=c)
    plt.fill_between(x, 0, n.pdf(x), alpha=0.25, color=c)
    plt.scatter(
        n.rvs(n_samples), 
        -.05*np.ones(n_samples), 
        marker="^",
        color=c,
        alpha=.25,
        s=100
    )
plt.savefig("./results/2-08-sample-with-dist.png")