# Imports
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Generate data
data = norm(450, 50).rvs(25)

# Sample
trace = mcmc(
    data=data, 
    dist=norm, 
    target="loc", 
    init=500, 
    proposal_width=100, 
    params_prior={"loc": 500, "scale": 70},
    params_const={"scale": 50},
    n_iter=10000
)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(range(len(trace)), trace)
sns.distplot(trace, ax=ax[1])
plt.savefig("./results/3-05-mcmc-results.png")