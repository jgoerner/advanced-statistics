# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)

# Prepare the Distributions
mus = np.arange(500, 1500, 250)
sigs = np.array([70, 90, 120, 150]) # given
N = []
for mu, sig in zip(mus, sigs):
    N.append(norm(loc=mu, scale=sig))