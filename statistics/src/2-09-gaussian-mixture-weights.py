# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare distributions
mus = [-1, 1.5, 3]
sigs = [1, .7, .5]
pis = [.25, .25, .5]
N = []

for mu, sig in zip(mus, sigs):
    N.append(norm(loc=mu, scale=sig))

x = np.linspace(-4, 6, 100)
PDF = np.zeros_like(x)

for n, pi, idx in zip(N, pis, ["a", "b", "c"]):
    plt.plot(x, n.pdf(x), label="Gaussian {}".format(idx))
    PDF += pi * n.pdf(x)
plt.plot(x, PDF, label="Mixture")
plt.legend(loc=0)
plt.savefig("./results/2-09-single-and-mixture.png")

noms = []
for n, pi in zip(N, pis):
    noms.append(n.pdf(3) * pi)
pis_ = list(map(lambda x: x / sum(noms), noms))
for idx, pi in zip(["a", "b", "c"], pis_):
    print("P({}=1|3) ~ {:.05f}".format(idx, pi))