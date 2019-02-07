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

# Plot
xx, yy = np.meshgrid(np.linspace(-6, 6, 300), np.linspace(-6, 6, 300))
fig, axes = plt.subplots(nrows=2, ncols=2)
for idx, (n, ax) in enumerate(zip(N, axes.ravel()), 1):
    zz = n.pdf(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap="inferno")  
    ax.set_title("Distribution {}".format(idx))
plt.tight_layout()
plt.savefig("./results/2-06-contour-single-dist.png")