from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xx, yy, zz, rcount=20, ccount=20)
plt.savefig("./results/2-06-3d-multi-dist.png")