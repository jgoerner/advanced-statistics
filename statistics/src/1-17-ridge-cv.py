# Imports
from sklearn.datasets import load_boston
from sklearn.linear_model import RidgeCV

# Prepare the data
boston = load_boston()
X = boston.data
y = boston.target

# Perform the Regression
lr = RidgeCV(cv=50)
lr.fit(X, y)

# Evalute
print("prediction = " + " + ".join(
    ['{:.2}*{}'.format(c, n) 
     for n, c 
     in zip(boston.feature_names, lr.coef_)]))