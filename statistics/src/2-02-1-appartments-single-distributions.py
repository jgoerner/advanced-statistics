x = np.arange(2000)
for idx, n in enumerate(N):
    plt.plot(x, n.pdf(x), color="red", alpha=1-idx*0.25)
plt.legend(["{}-Zimmer Whg".format(i) for i in range(1, 5)])
plt.title("Density Plot for housing prices in AlbSig")
plt.savefig("./results/2-02-1-single-distributions.png")