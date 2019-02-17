# Import
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
from scipy.stats import poisson
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (12, 3)

# Preparation
N = 100
true_lams = [20, 50]
true_tau = 30
data = np.hstack([
    poisson(true_lams[0]).rvs(true_tau),
    poisson(true_lams[1]).rvs(N - true_tau),
])

# Modeling
with pm.Model() as model:
    lam_1 = pm.Exponential("lam_1", data.mean())
    lam_2 = pm.Exponential("lam_2", data.mean())
    tau = pm.DiscreteUniform("tau", lower=0, upper=N-1)
    idx = np.arange(N)
    lam = pm.math.switch(tau > idx, lam_1, lam_2)
    female = pm.Poisson("target", lam, observed=data)
    step = pm.Metropolis()
    trace = pm.sample(20000, tune=5000, step=step, chains=10)
    pm.traceplot(trace[1000:], grid=True)
plt.savefig("./results/3-15-a-inference.png")