# coding:utf-8

import numpy as np
from sklearn.metrics import accuracy_score

y = np.array([1, 0, 0, 1])
y_hat = np.array([0.6, 0.51, 0.4, 0.55])
print(sum(y != (y_hat > 0.5)))

y_test = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
y_hat = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
accuracy = accuracy_score(y_test, y_hat)
print(accuracy)