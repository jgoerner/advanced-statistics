import numpy as np
import theano.tensor as T
from theano import function

# Define Function
x = T.vector("x")
y = T.vector("y")
z = x + y
f = function([x, y], z)

# Evaluate Function
print(f(np.array([1, 2, 3]), np.array([4, 5, 6])))