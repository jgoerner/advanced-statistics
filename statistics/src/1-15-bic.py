# Imports
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoLarsIC

# Prepare the data
boston = load_boston()
X = boston.data
y = boston.target

# Perform the Regression
lr = LassoLarsIC(criterion="bic")
lr.fit(X, y)

# Evalute
print(lr.alpha_)