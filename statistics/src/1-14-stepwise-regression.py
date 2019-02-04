# imports
import matplotlib.pyplot as plt
import os
import pandas as pd

# config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
os.chdir("/home/jovyan/work")

# prepare data
df_cars = pd.read_csv("./data/mtcars.csv")
df_cars.drop("model", axis=1, inplace=True) # no OHE

# helper function
def stepwise_regression(df, target):
    # prepare stepwise regression
    lr = LinearRegression()
    features = df.columns.tolist()
    features.remove(target)
    var = []
    R2 = []

    # fit stepwise
    while len(features) != 0:
        scores = []
        for f in features:
            data = (df[var + [f]], df[[target]])
            scores.append((lr.fit(*data).score(*data), f))
        scores.sort()
        r2, f = scores.pop()
        R2.append(r2)
        var.append(f)
        features.remove(f)
    return var, R2

# stepwise regression for mpg
var_mpg, r2_mpg = stepwise_regression(df_cars, "mpg")

# stepwise regresssion for hp
var_hp, r2_hp = stepwise_regression(df_cars, "hp")

# plots
f, axs = plt.subplots(1, 2)

# mpg plot
axs[0].step(range(len(r2_mpg)+1), [0] + r2_mpg)
axs[0].set_xticks(range(len(var_mpg)+1))
axs[0].set_xticklabels(["-"] + var_mpg, rotation=45)
axs[0].set_title("mpg")
axs[0].set_ylabel("R2")

# hp plot
axs[1].step(range(len(r2_hp)+1), [0] + r2_hp)
axs[1].set_xticks(range(len(var_hp)+1))
axs[1].set_xticklabels(["-"] + var_hp, rotation=45)
axs[1].set_title("hp")
axs[1].set_ylabel("R2")

plt.savefig("./results/1-14-stepwise-regression.png");