# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare Distributions
mus = [-1, 0, 3]
sigs = [1.5, 1, 0.5]
pis = [0.3, 0.5, 0.2]
N = []
for mu, sig in zip(mus, sigs):
    N.append(norm(loc=mu, scale=sig))

# Sample
n_samples = 100000
samples = np.empty(0)
for pi, n in zip(pis, N):
    samples = np.concatenate((samples, n.rvs(int(n_samples*pi))))

# Evaluation
sns.distplot(
    samples, 
    bins=100, 
    kde_kws={"color":"red", "linewidth":5})
plt.title("10,000 samples from Gaussian Mixture Model")
plt.savefig("./results/2-01-gmm-sample.png")