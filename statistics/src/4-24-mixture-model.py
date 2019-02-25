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
n_cluster = [90, 50, 75]
std_devs = [2, 2, 2]
mus = [9, 21, 35]

mix = np.random.normal(
    np.repeat(mus, n_cluster), np.repeat(std_devs, n_cluster)
)

# Sampling
n = len(n_cluster)
with pm.Model() as model:
    p = pm.Dirichlet("p", np.ones(n))
    k = pm.Categorical("k", p=p, shape=sum(n_cluster))
    means = pm.Normal("means", mu=[10, 10, 10], sd=10, shape=n)
    sigmas = pm.HalfCauchy("sigmas", 5)
    y = pm.Normal("y", mu=means[k], sd=sigmas, observed=mix)
    trace = pm.sample(5000, tune=1000, chains=1)

# Plot
samples = pm.sample_posterior_predictive(
    trace=trace,
    samples=100,
    model=model)
for sample in samples["y"]:
    sns.kdeplot(sample, color="red", alpha=0.1)
sns.kdeplot(mix, color="blue", linewidth=3)
plt.savefig("./results/4-24-mixture-model.png")