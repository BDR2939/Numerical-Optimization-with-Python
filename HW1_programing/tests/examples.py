import numpy as np
from math import e


def circles(x, flag):
    Q = [[1, 0][0, 1]]

    value = np.matmul(np.matmul(x.transpose, Q), x)
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def ellipses(x, flag):
    Q = [[1, 0][0, 100]]

    value = np.matmul(np.matmul(x.transpose, Q), x)
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def rotated_ellipses(x, flag):
    side = [[np.sqrt(3) / 2, -0.5][0.5, np.sqrt(3) / 2]]
    Q = [[100, 0][0, 1]]
    Q = np.matmul(np.matmul(side.transpose, Q), side)

    value = np.matmul(np.matmul(x.transpose, Q), x)
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def rosenbrock(x, flag):
    value = 100 * ((x[1] - x[0] ** 2) ** 2) + ((1 - x[0]) ** 2)
    grad = [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]

    if flag:
        hess = [[-400 * x[1] + 1200 * x[0] ** 2 - 2, -400 * x[0]], [-400 * x[0], 200]]
        return value, grad.transpose, hess

    return value, grad.transpose


def linear(x, flag):
    a = [2, 9]
    a = a.transpose

    value = a.transpose * x
    grad = a

    if flag:
        return value, grad, 0

    return value, flag


def triangles(x, flag):
    first_pow = x[0] + 3 * x[1] - 0.1
    second_pow = x[0] - 3 * x[1] - 0.1
    third_pow = -x[0] - 0.1

    value = e ** first_pow + e ** second_pow + e ** third_pow
    grad = [e ** x[0], 3 * e ** (3 * x[1]) - 3 * e ** (-3 * x[1])]
