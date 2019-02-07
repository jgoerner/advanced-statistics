# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare distributions
mus = [-1, 0, 3]
sigs = [1.5, 1, .5]
pis = [.3, .5, .2]
N = []

for mu, sig in zip(mus, sigs):
    N.append(norm(loc=mu, scale=sig))

n_samples = 10000
samples = np.concatenate([
    n.rvs(int(pi*n_samples))
    for n, pi
    in zip(N, pis)
])
x = np.linspace(-6, 4, 500)

gmm = GaussianMixture(3)
gmm.fit(samples.reshape(-1, 1))
pdf_mix_gmm = np.zeros_like(x.reshape(1, -1))
for w, mu, sig in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    pdf_mix_gmm += w * norm(loc=mu, scale=sig).pdf(x)

pdf_mix = np.sum([pi * n.pdf(x) for pi, n in zip(pis, N)], axis=0)
plt.hist(samples, bins=50, normed=True, alpha=.25)
plt.plot(x, pdf_mix, linewidth=3, color="r", label="Real")
plt.plot(x, pdf_mix_gmm.T, linewidth=3, color="b", label="GMM")
plt.legend(loc=0)
plt.savefig("./results/2-10-sklearn-gmm.png")