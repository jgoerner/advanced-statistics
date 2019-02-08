# Import
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba_array
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

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

# GMM
gmm = GaussianMixture(4)
gmm.fit(samples)

# Plotting
LiCmap = ListedColormap(["r", "b", "g", "k"])
fig, ax = plt.subplots(ncols=2, nrows=1)

# Left Plot
ax[0].scatter(*samples.T, s=1, c=gmm.predict(samples), cmap=LiCmap)

# Right Plot
for cls, c in zip(range(4), LiCmap.colors):
    rgba = np.repeat(to_rgba_array(c), samples.shape[0], axis=0)
    rgba[:, 3] = gmm.predict_proba(samples).T[cls]
    ax[1].scatter(*samples.T, color=rgba, s=1)
plt.savefig("./results/2-15-soft-clustring.png")