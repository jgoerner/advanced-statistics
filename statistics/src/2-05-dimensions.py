# Import
import numpy as np

# Build Datasets
x = y = np.linspace(-4.5, 1.5, 200)
X, Y = np.meshgrid(x, y)

# Shape of X and Y
# ==> (200, 200)
print(X.shape)

# Shape of X.ravel(), shape of X.ravel().T
# ==> (40000, )
print(X.ravel().shape)
print(X.ravel().T.shape)

# Shape of np.array([X.ravel(), Y.ravel()]).T
# ==> (40000, 2)
print(np.array([X.ravel(), Y.ravel().T]).T.shape)

# How to "square" the range np.arange(0, 25)
print(np.arange(0, 25).reshape(-1, 5).shape)