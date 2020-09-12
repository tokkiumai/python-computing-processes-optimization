import numpy as np

from utils import cal_cost

def stocashtic_gradient_descent(X, y, theta, step = 0.01, iterations = 10):
  m = len(y)
  cost_history = np.zeros(iterations)

  for it in range(iterations):
    cost = 0.0

    for i in range(m):
      rand_ind = np.random.randint(0, m)
      X_i = X[rand_ind, :].reshape(1, X.shape[1])
      y_i = y[rand_ind].reshape(1, 1)

      prediction = np.dot(X_i, theta)

      theta = theta - (1 / m) * step * (X_i.T.dot(( prediction - y_i )))
      cost += cal_cost(theta, X_i, y_i)
    
    cost_history[it] = cost
  
  return theta, cost_history
