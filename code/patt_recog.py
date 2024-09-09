import numpy as np

mean = (1, 1)

mean2 = (1.5, 1.5)

cov = [[0.2, 0], [0, 0.2]]

x = np.random.multivariate_normal(mean, cov, size=500)

y = np.random.multivariate_normal(mean2, cov, size=500)


import matplotlib.pyplot as plt

plt.plot(x[:,0], x[:,1], 'b.', alpha=0.5)
plt.plot(y[:,0], y[:,1], 'r.', alpha=0.5)

plt.axis('equal')

plt.grid()

plt.show()

print("test")

from scipy import stats

