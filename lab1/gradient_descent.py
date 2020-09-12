import numpy as np

from utils import cal_cost

def gradient_descent(X, y, theta, step = 0.01, iterations = 100):
  m = len(y)

  cost_history = np.zeros(iterations)
  theta_history = np.zeros((iterations, 2))
  
  for it in range(iterations):
    prediction = np.dot(X, theta)

    theta = theta - (1 / m) * step * (X.T.dot((prediction - y)))
    theta_history[it,:] = theta.T

    cost_history[it] = cal_cost(theta, X, y)

  return theta, cost_history, theta_history
