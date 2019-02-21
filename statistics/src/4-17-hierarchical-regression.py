import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import norm, uniform
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(1)

# Prepare the data
N = 30
b_0 = [3, 3.2, 2.4, 4.1]
b_1 = [11, 10, 9.1, 14.1]
i = np.repeat(range(4), N)
x = np.array([])
y = np.array([])
for b0, b1 in zip(b_0, b_1):
    x_new = norm(0, 20).rvs(N)
    x = np.append(x, x_new)
    eps = norm(0, 10).rvs(N)
    y = np.append(y, b0 + b1*x_new + eps)
x = np.append(x, 10)
y = np.append(y, 3 + 11*10)
i = np.append(i, 4)

# Plot Initial Data
_, ax = plt.subplots(nrows=1, ncols=5, sharey=True)
for idx, a in enumerate(ax):
    a.scatter(x[idx*N:(idx+1)*N], y[idx*N:(idx+1)*N])
plt.savefig("./results/4-17-hierarchical-regression-data.png")

# Sampling
with pm.Model() as model:
    b_0 = pm.Normal("b_0", mu=0, sd=100, shape=5)
    b_1 = pm.Normal("b_1", mu=0, sd=100, shape=5)
    eps = pm.Uniform("e", 0, 40)
    mu = pm.Deterministic("mu", b_0[i] + b_1[i]*x)
    Y = pm.Normal("Y", mu=mu, sd=eps, observed=y)
    trace = pm.sample(5000)

# Evaluation
colors = sns.color_palette()
_, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
for c in range(5):
    sns.distplot(trace["b_1"].T[c], ax=ax[0][c], color=colors[c])
    sns.distplot(trace["b_0"].T[c], ax=ax[1][c], color=colors[c])
plt.ylim(0, 0.5)
plt.xlim(0, 20)
plt.savefig("./results/4-17-hierarchical-non-robust.png")