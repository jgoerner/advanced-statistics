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
np.random.seed(42)

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

# Sampling
with pm.Model() as model:
    mu_0 = pm.Normal("mu_0", mu=0, sd=10)
    sd_0 = pm.HalfCauchy("sd_0", 5)
    b_0 = pm.Normal("b_0", mu=mu_0, sd=sd_0, shape=5)
    mu_1 = pm.Normal("mu_1", mu=0, sd=10)
    sd_1 = pm.HalfCauchy("sd_1", 5)
    b_1 = pm.Normal("b_1", mu=mu_1, sd=sd_1, shape=5)
    nu = pm.Deterministic("nu", pm.Exponential("lam", 1/20) + 1)
    eps = pm.HalfCauchy("e", 5)
    mu = pm.Deterministic("mu", b_0[i] + b_1[i]*x)
    Y = pm.StudentT("Y", mu=mu, sd=eps, nu=nu, observed=y)
    trace = pm.sample(2000, tune=1000)

# Evaluation
print("HPD-Range B0: \n{}".format(np.diff(pm.hpd(trace["b_0"])).ravel()))
print("HPD-Range B1: \n{}".format(np.diff(pm.hpd(trace["b_1"])).ravel()))

# Plot
colors = sns.color_palette()
x_ = np.linspace(-50, 50, 100)
_, axes = plt.subplots(nrows=3, ncols=5)
for c in range(5):
    sns.distplot(trace["b_1"].T[c], ax=axes[0][c], color=colors[c])
    sns.distplot(trace["b_0"].T[c], ax=axes[1][c], color=colors[c])
    axes[2][c].scatter(x[i==c], y[i==c], color=colors[c], s=10)
    for _ in range(30):
        idx = np.random.randint(0, len(trace))
        axes[2][c].plot(
            x_, 
            trace["b_0"].T[c][idx] + trace["b_1"].T[c][idx] * x_,
            alpha=0.25,
            color="grey"
        )
plt.ylim(0, 0.5)
plt.xlim(0, 20)
plt.tight_layout()
plt.savefig("./results/4-18-hierarchical-robust.png")