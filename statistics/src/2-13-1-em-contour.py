# Import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare Distributions
mus = [
    [-1.5, -1.0], 
    [0.0, 2.0], 
    [-4.0, 0.0], 
    [3.0, 0.0]
]
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

# Fit GMM
gmm = GaussianMixture(4)
gmm.fit(samples)

# Prepare Plot
x = y = np.linspace(-8, 8, 100)
xx, yy = np.meshgrid(x, y)
zz = gmm.score_samples(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)

# Plot
plt.scatter(*samples.T, alpha=.1, s=1)
cs = plt.contour(
    xx, 
    yy, 
    zz, 
    levels=-np.logspace(0, 1, 15)[::-1]
)
plt.clabel(cs)
plt.savefig("./results/2-13-contour-2d.png")