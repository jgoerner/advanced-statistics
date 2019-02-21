# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import norm
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(42)

# Prepare the data
data = norm(50, 5).rvs(30)
data = np.append(data, [79, 83, 99])

# Sampling
with pm.Model() as model:
    nu = pm.Exponential("nu", lam=1/10)
    mu = pm.Uniform("mu", 0, 100)
    sigma = pm.HalfNormal("sigma", 5)
    y = pm.StudentT("y", mu=mu, sd=sigma, nu=nu, observed=data)
    trace = pm.sample(5000)

sns.kdeplot(data, label="original")
preds = pm.sample_posterior_predictive(trace, samples=100, model=model)["y"]
for p in preds:
    sns.kdeplot(p, alpha=.1, color="red")
plt.xlim((0, 100))
plt.savefig("./results/4-08-robust-outlier.png")