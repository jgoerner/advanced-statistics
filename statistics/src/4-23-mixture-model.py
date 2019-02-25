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
Ns = [90, 50, 80]
mus = [4, 10, 20]
sds = [0.5, 4, 1.5]

mix = np.array([])
for N, mu, sd in zip(Ns, mus, sds):
    mix = np.append(mix, norm(mu, sd).rvs(N))

# Sampling
n = len(Ns)
with pm.Model() as model:
    p = pm.Dirichlet("p", np.ones(n))
    k = pm.Categorical("k", p=p, shape=sum(Ns))
    means = pm.Normal("means", mu=[10, 10, 10], sd=10, shape=n)
    sigmas = pm.HalfCauchy("sigmas", 5, shape=n)
    y = pm.Normal("y", mu=means[k], sd=sigmas[k], observed=mix)
    trace = pm.sample(5000, tune=1000)
    
# Plot
pm.traceplot(
    trace, 
    varnames=["means", "p", "sigmas"],
    lines={"means": mus, "sigmas":sds}
)
plt.savefig("./results/4-23-mixture-model.png")