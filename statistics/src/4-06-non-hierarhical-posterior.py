# Sampling
with pm.Model() as model:
    theta = pm.Beta("theta", alpha=1, beta=1, shape=4)
    y = pm.Bernoulli("y", p=theta[grp], observed=data)
    trace = pm.sample(5000)

# Plotting
pm.traceplot(trace)
plt.savefig("./results/4-06-non-hierarhical-posterior.png")