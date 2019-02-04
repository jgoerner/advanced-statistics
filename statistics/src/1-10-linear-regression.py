# Imports
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Config
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
os.chdir("/home/jovyan/work")

# read dataframe
df_boston = pd.read_csv("./data/boston.csv", index_col=0)

# perform logistic regression
X, y = df_boston[["medv"]].values, df_boston["lstat"].values
lr = LinearRegression()
lr.fit(X, y)

# graphical evaluation
plt.scatter(X, y)
plt.xlabel("medv")
plt.ylabel("lstat")
plt.title("Intercerpt: {}, Coefficient: {}".format(lr.intercept_, lr.coef_))
plt.xticks([])
plt.yticks([])
plt.plot(X, lr.predict(X), color="red", linewidth=3)
plt.savefig("./results/1-10-linear-regression.png");