import numpy as np

from gradient_descent import gradient_descent

step = 0.01
iterations = 100

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((len(X), 1)), X]

theta, cost_history, theta_history = gradient_descent(X_b, y, theta, step, iterations)

print('Theta0: {:0.3f} \nTheta1: {:0.3f}'.format(theta[0][0], theta[1][0]))
print('Final cost: {:0.3f}'.format(cost_history[-1]))
