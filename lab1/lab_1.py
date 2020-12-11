from math import sqrt, exp, cos, sin, pi, e
import matplotlib.pyplot as plt

import numpy as np
from numpy import arange

from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D

def partial_derivative(func, var = 0, point = []):
  args = point[:]

  def wraps(x):
    args[var] = x
    return func(*args)
  
  return derivative(wraps, point[var], dx = 1e-6)

def norm(x, y = 0):
  return sqrt(x * x + y * y)

def next(x0, y0, tk, d0, d1):
  return x0 - tk * d0, y0 - tk * d1

def df(fn, *args):
  xn = np.array([*args])
  return tuple((partial_derivative(fn, i, xn) for i in range(len(args))))

def generate_data():
  x = np.arange(-5, 5, 0.1)
  y = np.arange(-5, 5, 0.1)

  xGrid, yGrid = np.meshgrid(x, y)
  zGrid = np.sin(xGrid) * np.sin(yGrid) / (xGrid * yGrid)

  return xGrid, yGrid, zGrid

def main(fn):
  x0, y0 = 0.5, 1

  e1 = 0.0001
  e2 = 0.0001
  tk = 0.01

  maxiter = 200
  k = 0

  x1, y1 = x0, y0

  flag = False

  while True:
    d0, d1 = df(fn, x0, y0)
    dNorm = norm(d0, d1)

    if dNorm < e1:
      result = x0, y0
      break

    if k >= maxiter:
      result = x0, y0
      break

    x1, y1 = next(x0, y0, tk, d0, d1)

    while not (fn(x1, y1) < fn(x0, y0)):
      tk = tk / 2.0
      k += 1

      if k >= maxiter:
        result = x0, y0
        break

      x1, y1 = next(x0, y0, tk, d0, d1)
    
    if (norm(x1 - x0, y1 - y0) < e2) and (abs(fn(x1, y1) - fn(x0, y0)) < e2):
      if flag:
        result = x1, y1
        break
      else:
        flag = True
        k = k + 1
    else:
      k = k + 1
      flat = False
    
    x0, y0 = x1, y1

  i = np.arange(-5, 5, 0.01)

  X, Y = np.meshgrid(i, i)
  Z = f1, X, Y

  print('Min \n x: ' + str(x0) + '\n y: ' + str(y0) + '\n z: ' + str(fn(x0, y0)))
  print(k)

  figure = plt.figure()

  ax = figure.add_subplot(111, projection = '3d')
  ax.plot_surface(X, Y, Z, color = 'grey')

  ax.scatter(x0, y0, fn(x0, y0), color = 'green', s = 40, marker = 'o')
  plt.show()
