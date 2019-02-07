# Prepare the mixture
pis = [.3, .3, .25, .15]
PDF = np.zeros_like(x, dtype="float64")
for pi, n in zip(pis, N):
    PDF += pi * n.pdf(x)

# Evaluate
plt.plot(x, PDF)
plt.title("GMM for weights [30%, 30%, 25%, 15%]")
plt.savefig("./results/2-02-2-mixed-distributions.png")