# Imports
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import bernoulli
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(42)

# Prepare the data
samples = np.repeat(25, 4)
hits = np.repeat(15, 4)
data = [s for i,j in zip(samples, hits)
         for s in random.sample([1]*j+[0]*(i-j), 25)] 
grp = np.repeat(np.arange(4), 25)

# Sampling
with pm.Model() as model:
    alpha = pm.HalfCauchy("alpha", beta=10)
    beta = pm.HalfCauchy("beta", beta=10)
    theta = pm.Beta("theta", alpha=alpha, beta=beta, shape=4)
    y = pm.Bernoulli("y", p=theta[grp], observed=data)
    trace = pm.sample(5000)

# Plotting
fig, ax = plt.subplots(nrows=1, ncols=2)
sns.distplot(trace["alpha"], ax=ax[0], label=r"$\alpha$")
sns.distplot(trace["beta"], ax=ax[0], label=r"$\beta$")
ax[1].plot(range(len(trace["alpha"])), trace["alpha"], alpha=.25, label=r"$\alpha$")
ax[1].plot(range(len(trace["beta"])), trace["beta"], alpha=.25, label=r"$\beta$")
ax[0].legend()
ax[1].legend()
plt.savefig("./results/4-05-hierarhical-posterior.png")