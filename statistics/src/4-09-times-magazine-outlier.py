import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(42)

# Prepare the data
df = pd.read_csv("./data/times_magazine.csv")

# Sampling
with pm.Model() as model:
    mu = pm.Uniform("mu", 0, 18)
    sigma = pm.HalfNormal("sigma", 3)
    y = pm.Normal("y", mu=mu, sd=sigma, observed=df.Female)
    trace = pm.sample(5000)

sns.kdeplot(df.Female, label="original")
preds = pm.sample_posterior_predictive(trace, samples=100, model=model)["y"]
for p in preds:
    sns.kdeplot(p, alpha=.1, color="red")
plt.savefig("./results/4-09-times-outlier.png")