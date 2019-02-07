# Import
import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare Distributions
mus = [[-1.5, -1.0], [0.0, 2.0], [-4.0, 0.0], [3.0, 0.0]]
sigs = [
    [[2.0, .4], [.4, .5]],
    [[1.0, -.9], [0, .1]],
    [[.5, .3], [.9, 2.0]],
    [[1.0, .0], [.0, 1.0]],
]
pis = [.15, .2, .5, .15]
N = []
for mu, sig in zip(mus, sigs):
    N.append(multivariate_normal(mean=mu, cov=sig))

# Sample
n_samples = 20000
samples = np.concatenate(
    [n.rvs(int(n_samples * pi)) 
     for pi, n 
     in zip(pis, N)])

# Evaluate
g = sns.jointplot(samples.T[0], samples.T[1], s=.1)
g.ax_joint.legend_.remove()
plt.savefig("./results/2-03-multivariable-gmm.png")