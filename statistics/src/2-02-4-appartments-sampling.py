# Sample
n_samples = 10000
samples = np.empty(0)
for pi, n in zip(pis, N):
    samples = np.concatenate((samples, n.rvs(int(pi*n_samples))))

# Evaluate
sns.distplot(
    samples, 
    bins=50, 
    kde_kws={"color":"red", "linewidth":5})
plt.title("10,000 samples from Gaussian Mixture Model")
plt.savefig("./results/2-02-3-sample-with-mixed-distribution.png")