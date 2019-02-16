# Imports
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 10)

# Generate data
data = norm(450, 50).rvs(25)

step_sizes = [100, 125, 150, 175, 200]
traces = []

# Sample
for step_size in step_sizes:
    
    traces.append(mcmc(
        data=data, 
        dist=norm, 
        target="loc", 
        init=500, 
        proposal_width=step_size, 
        params_prior={"loc": 500, "scale": 70},
        params_const={"scale": 50},
        n_iter=10000,
        desc="Step Size - {}".format(step_size)
    ))

# Plot
fig, ax = plt.subplots(nrows=len(step_sizes), ncols=2)
for idx, step in enumerate(step_sizes):
    ax[idx][0].plot(range(len(traces[idx])), traces[idx])
    ax[idx][0].set_title("Step Size: {}".format(step))
    sns.distplot(traces[idx], ax=ax[idx][1])
    ax[idx][1].set_title("Step Size: {}".format(step))
plt.tight_layout()
plt.savefig("./results/3-06-mcmc-step-sizes.png")