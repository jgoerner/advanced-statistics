# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import bernoulli

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(42)

# Prepare the data
y = bernoulli(0.22).rvs(10)
print(y)
chains = 1000 * 2**np.arange(6)
cols = ['mean', 'sd', 'mc_error', 'hpd_2.5', 'hpd_97.5', 'n_eff', 'Rhat']
df_summaries = pd.DataFrame(columns=cols)

# Inferece
for chain in chains:
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        throw = pm.Bernoulli("throw", theta, observed=y)
        trace = pm.sample(chain, step=pm.Metropolis(), progressbar=False)
        df_summaries = df_summaries.append(pm.summary(trace))

# Calculate the HPD interval range
df_summaries["hpd"] = df_summaries["hpd_97.5"] - df_summaries["hpd_2.5"]

plt.plot(chains, df_summaries["hpd"])
plt.title("HPD range based on different MCMC lengths")
plt.savefig("./results/4-04-hpd-chain-length.png")