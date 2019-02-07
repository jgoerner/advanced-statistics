# Imports
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge

# Prepare the data
boston = load_boston()
X = boston.data
y = boston.target

# Perform the Regression
lr = Ridge()
lr.fit(X, y)

# Evalute
print("prediction = " + " + ".join(
    ['{:.2}*{}'.format(c, n) 
     for n, c 
     in zip(boston.feature_names, lr.coef_)]))