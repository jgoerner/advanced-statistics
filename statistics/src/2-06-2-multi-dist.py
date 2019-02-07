zz = np.zeros_like(xx)
for pi, n in zip(pis, N):
    zz += pi * n.pdf(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap="inferno")
plt.title("Multivariate Gausssian Mixture Model")
plt.savefig("./results/2-06-contour-multi-dist.png")