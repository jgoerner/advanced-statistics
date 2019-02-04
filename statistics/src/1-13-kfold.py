# imports
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold

# load the data
diabetes = load_diabetes()
X, y = diabetes.data[:, np.newaxis, 2], diabetes.target

# prepare the k-fold
kfold = KFold(100)
lr = LinearRegression()

# split & fit
intercerpts = []
coefficients = []
for train, test in kfold.split(X):
    lr.fit(X[train], y[train])
    intercerpts.append(lr.intercept_)
    coefficients.append(lr.coef_[0])

# Evaluate
print("90% of Intercerpt between {:.5f} and {:.5f}"
      .format(*np.percentile(intercerpts, [5, 95])))
print("90% of Coefficients between {:.5f} and {:.5f}"
      .format(*np.percentile(coefficients, [5, 95])))