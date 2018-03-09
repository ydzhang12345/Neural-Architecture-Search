import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

import pdb


x, y = np.mgrid[-10:10:0.02, -10:10:0.02]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv1 = multivariate_normal([5, -5], [[10, 0], [0, 10]])
rv2 = multivariate_normal([2, -2], [[7, 2], [2, 5]])
rv3 = multivariate_normal([7, -7], [[1, 0], [0, 1]])
rv4 = multivariate_normal([3, -3], [[1, 0], [0, 1]])
rv11 = multivariate_normal([-5, 5], [[3, 1], [1, 2]]) 
rv22 = multivariate_normal([-2, 2], [[7, 2], [2, 5]])
rv33 = multivariate_normal([-7, 7], [[1, 0], [0,1]])
rv44 = multivariate_normal([-3, 3], [[4, 0], [0, 4]])
rv = rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos) + rv4.pdf(pos) + rv11.pdf(pos) + rv22.pdf(pos) + rv33.pdf(pos) + rv44.pdf(pos)



#z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
fig = plt.figure()
#plt.contourf(x, y, rv)
#plt.show()

ax = Axes3D(fig) #fig.add_subplot(111, projection='3d')
ax.plot_surface(x,  y, rv)
plt.show()

pdb.set_trace()
a = 1