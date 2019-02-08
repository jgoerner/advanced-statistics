fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40)
ax.plot_surface(xx, yy, -zz)
plt.savefig("./results/2-13-contour-3d.png")