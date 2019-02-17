# Import
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
from scipy.stats import norm
import seaborn as sns

# Config
os.chdir("/home/jovyan/work")
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (12, 3)

# Preparation
data = pd.read_csv("./data/times_magazine.csv")
print(tabulate(data.head(), headers="keys", tablefmt="psql"))

# Modeling
N = len(data.Female)
lam_ = data.Female.mean()
with pm.Model() as model:
    lam_1 = pm.Exponential("lam_1", lam_)
    lam_2 = pm.Exponential("lam_2", lam_)
    tau = pm.DiscreteUniform("tau", lower=1923, upper=1923+N)
    idx = np.arange(1923, 1923+N)
    lam = pm.math.switch(tau > idx, lam_1, lam_2)
    female = pm.Poisson("female", lam, observed=data.Female)
    step = pm.Metropolis()
    trace = pm.sample(20000, tune=5000, step=step)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2)
sns.distplot(trace["lam_1"], label="λ1", ax=ax[0])
sns.distplot(trace["lam_2"], label="λ2", ax=ax[0])
sns.countplot(trace["tau"], ax=ax[1])
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("./results/3-14-times-magazine.png")