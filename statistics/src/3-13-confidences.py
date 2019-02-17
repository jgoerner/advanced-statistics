# %load src/3-13-confidences.py
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
sigs = [200, 20, 2, 0.2]
traces = []

# Sampling
for sig in sigs:
    with pm.Model() as model:
        mu = pm.Normal("mu", 500, sig)
        weight = pm.Normal("weight", mu, 50, observed=data)
        trace = pm.sample(
            draws=10000, 
            step=pm.Metropolis(), 
            chains=10)
        traces.append(trace[1000:])
        pm.traceplot(trace[1000:], grid=True)

# Plot
plt.savefig("./results/3-13-confidences.png")