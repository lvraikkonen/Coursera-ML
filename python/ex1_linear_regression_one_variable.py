# excerise 1 ----- Linear Regression with One Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression
import os


def hypothesis(theta, x):
    return np.dot(x, theta)


def cost_function(theta, x, y):
    loss = hypothesis(theta, x) - y
    cost = np.sum(np.power(loss, 2)) / (2 * len(y))
    return cost


def gradient_descent(alpha, x, y, iters):
    # number of training dataset
    m = x.shape[0]
    theta = np.zeros(2)
    cost_iter = []
    for iter in range(iters):
        h_theta = hypothesis(theta, x)
        loss = h_theta - y
        J = np.sum(loss ** 2) / (2 * m)
        cost_iter.append([iter, J])
        # print "iter %s | J: %.3f" % (iter, J)
        gradient = np.dot(x.T, loss) / m
        theta -= alpha * gradient
    return theta, cost_iter


# generate sample data using scikit-learn
x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                       random_state=0, noise=35)
m, n = np.shape(x)
# add column as x0
x = np.c_(np.ones(m), x)
print np.shape(x)

alpha = 0.01
theta, cost_iter = gradient_descent(alpha, x, y, 1000)

# plot the result
for i in range(x.shape[1]):
    y_pred = theta[0] + theta[1] * x
plt.plot(x[:, 1], y, 'o')
plt.plot(x, y_pred, 'k-')

# plot cost trend
plt.plot(cost_iter[:, 0], cost_iter[:, 1])
plt.xlabel("Iteration Number")
plt.ylabel("Cost")


# Homework1 linear regression with one variable
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

data.insert(0, 'X0', 1)  # add one column as x0
cols = data.shape[1]
X = data.iloc[:, 0: cols - 1]
y = data.iloc[:, cols - 1]

X = np.array(X.values)
y = np.array(y.values)

alpha = 0.01
theta_home, cost_home = gradient_descent(alpha, X, y, 1000)
print theta_home

# plot result
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta_home[0] + theta_home[1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# plot cost trend
cost_home = np.array(cost_home)
plt.plot(cost_home[:5, 0], cost_home[:5, 1])
plt.xlabel("Iteration Number")
plt.ylabel("Cost")
