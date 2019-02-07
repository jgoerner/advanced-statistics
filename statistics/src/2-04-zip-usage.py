# Import
import numpy as np

# prepare meshgrid
xx, yy = np.meshgrid(np.linspace(0, 5, 5), np.linspace(0, 5, 5))

# create numpy and zip array
numpy_usage = np.array([xx.ravel(), yy.ravel()]).T # only numpy usage
zip_usage = list(zip(xx.ravel(), yy.ravel())) # classical 'zip' usage

# compare
print(np.array_equal(np.array(zip_usage), numpy_usage))