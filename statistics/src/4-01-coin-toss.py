# Imports
import os
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import bernoulli

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare the data
y = bernoulli(0.22).rvs(10)
print(y)

# Inferece
with pm.Model() as model:
    theta = pm.Beta("theta", alpha=1, beta=1)
    throw = pm.Bernoulli("throw", theta, observed=y)
    trace = pm.sample(10000, step=pm.Metropolis(), progressbar=False)

pm.traceplot(trace[2000:])
plt.savefig("./results/4-01-coin-toss.png")