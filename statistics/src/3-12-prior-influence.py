# Import
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import norm

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (12, 3)

# Preparation
data = norm(450, 50).rvs(25)
lams = [0.2, 1., 5.]
traces = []

# Sampling
for lam in lams:
    with pm.Model() as model:
        mu = pm.StudentT("mu", mu=500, lam=lam, nu=2)
        weight = pm.Normal("weight", mu, 50, observed=data)
        trace = pm.sample(
            draws=10000, 
            step=pm.Metropolis(), 
            chains=10)
        traces.append(trace[1000:])

# Plot
for idx, lam in enumerate(lams):
    sns.distplot(traces[idx]["mu"], label="λ = {}".format(lam))
plt.ylim(0, 0.04)
plt.legend(loc=0)
plt.savefig("./results/3-12-prior-influence.png")