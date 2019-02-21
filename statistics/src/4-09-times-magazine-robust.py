# Sampling
with pm.Model() as model:
    nu = pm.Exponential("nu", 1/10)
    mu = pm.Uniform("mu", 0, 18)
    sigma = pm.HalfNormal("sigma", 3)
    y = pm.StudentT("y", mu=mu, sd=sigma, nu=nu, observed=df.Female)
    trace = pm.sample(5000)

sns.kdeplot(df.Female, label="original")
preds = pm.sample_posterior_predictive(trace, samples=100, model=model)["y"]
for p in preds:
    sns.kdeplot(p, alpha=.1, color="red")
plt.xlim(-10, 20)
plt.savefig("./results/4-09-times-robust.png")