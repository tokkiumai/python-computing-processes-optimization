import numpy as np

from scipy.optimize import minimize
from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fn(x):
  x1, x2 = x[0], x[1]
  return 2 * x1 * x1 + 3 * x2 * x2 + 4 * x1 * x2 - 6 * x1 - 3 * x2

def constraint(x):
  x1, x2 = x[0], x[1]
  return (x1 + x2 - 1, 2 * x1 + 3 * x2 - 4)

def callback_fn(xi):
  print(xi, fn(xi))

def main():
  b = (0, float('inf'))
  bnds = (b, b)

  x0 = (4, 2)
  con = { 'type': 'eq', 'fn': constraint }

  res = minimize(fn, x0, method = ' SLSQP', bounds = bnds, constraints = con, callback = callback_fn)

  print('Min: ' + '\n x: ' + str(res.x[0] + '\n y: ' + str(res.x[1]) + '\n z: ' + str(res.fun)))

  i = np.arange(-5, 5, 0.01)

  X, Y = np.meshgrid(i, i)
  Z = 2 * X * X + 3 * Y * Y + 4 * X * Y - 6 * X - 3 * Y

  figure = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')

  ax.scatter(res.x[0], res.x[1], res.fun, color = 'orange', s = 50, marker = 'o')
  ax.plot_surface(X, Y, Z, cmap = cm.coolwarm)

  plt.show()