import numpy as np
import matplotlib.pyplot as plt


def sigmoid(theta, x):
    return 1.0 / (1 + np.exp(-x.dot(theta)))


def gradient(theta, x, y):
    first_part = sigmoid(theta, x) - np.squeeze(y)
    return first_part.T.dot(x)


def cost_function(theta, x, y):
    h_theta = sigmoid(theta, x)
    y = np.squeeze(y)
    first = y * np.log(h_theta)
    second = (1 - y) * np.log(1 - h_theta)
    return np.mean(-first - second)


def gradient_descent(theta, X, y, alpha=0.001, converge=0.001):
    # attribute normailization
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    cost_iter = []
    # initial cost for given theta
    cost = cost_function(theta, X, y)
    cost_iter.append([0, cost])
    i = 1
    while(cost > converge):
        theta -= alpha * gradient(theta, X, y)
        cost = cost_function(theta, X, y)
        cost_iter.append([i, cost])
        i += 1
    return theta, np.array(cost_iter)


def predict_function(theta, x):
    # normailization
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    pred_prob = sigmoid(theta, x)
    pred_value = np.where(pred_prob >= 0.5, 1, 0)
    return pred_value
