import numpy as np
from math import e


def circles(x, flag):
    Q = np.array([[1, 0], [0, 1]])

    value = np.matmul(x.transpose(), np.matmul(Q, x))
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def ellipses(x, flag):
    Q = np.array([[1, 0], [0, 100]])

    value = np.matmul(x.transpose(), np.matmul(Q, x))
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def rotated_ellipses(x, flag):
    side = np.array([[(3**0.5) / 2, -0.5], [0.5, (3**0.5) / 2]])
    Q = np.array([[100, 0], [0, 1]])
    Q = np.matmul(side.transpose(), np.matmul(Q, side))

    value = np.matmul(x.transpose(), np.matmul(Q, x))
    grad = 2 * np.matmul(Q, x)
    if flag:
        return value, grad, 2 * Q

    return value, grad


def rosenbrock(x, flag):
    value = 100 * ((x[1] - x[0] ** 2) ** 2) + ((1 - x[0]) ** 2)
    grad = np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    )

    if flag:
        hess = np.array(
            [[-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]]
        )
        return value, grad.transpose(), hess

    return value, grad.transpose()


def linear(x, flag):
    a = np.array([2, 9])
    a = a.transpose()

    value = np.matmul(a.transpose(), x)
    grad = a

    if flag:
        return value, grad, 0

    return value, grad


def triangles(x, flag):
    first_pow = x[0] + 3 * x[1] - 0.1
    second_pow = x[0] - 3 * x[1] - 0.1
    third_pow = -x[0] - 0.1

    value = e**first_pow + e**second_pow + e**third_pow
    grad = np.array(
        [2 * e ** x[0] - e ** -x[0], 3 * e ** (3 * x[1]) - 3 * e ** (-3 * x[1])]
    )

    if flag:
        hess = np.array(
            [
                [e * e ** x[0] + e ** -x[0], 0],
                [0, 9 * e ** (3 * x[1]) + 9 * e ** (-3 * x[1])],
            ]
        )
        return value, grad.transpose(), hess

    return value, grad.transpose()


def qp(x, flag):
    value = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2
    grad = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])
    if flag:
        hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        return value, grad.transpose(), hess

    return value, grad.transpose()


def ineq_constraint_1_1(x, flag):
    value = -x[0]
    grad = np.array([-1, 0, 0])

    if flag:
        hess = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        return value, grad.transpose(), hess

    return value, grad


def ineq_constraint_1_2(x, flag):
    value = -x[1]
    grad = np.array([0, -1, 0])

    if flag:
        hess = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        return value, grad.transpose(), hess

    return value, grad


def ineq_constraint_1_3(x, flag):
    value = -x[2]
    grad = np.array([0, 0, -1])

    if flag:
        hess = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        return value, grad.transpose(), hess

    return value, grad


def lp(x, flag):
    value = -x[0] - x[1]
    grad = np.array([-1, -1])
    if flag:
        hess = np.array([[0, 0], [0, 0]])
        return value, grad.transpose(), hess

    return value, grad.transpose()


def ineq_constraint_2_1(x, flag):
    value = -x[0] - x[1] + 1
    grad = np.array([-1, -1])

    if flag:
        hess = np.array([[0, 0], [0, 0]])
        return value, grad.transpose(), hess

    return value, grad.transpose()


def ineq_constraint_2_2(x, flag):
    value = x[1] - 1
    grad = np.array([0, 1])

    if flag:
        hess = np.array([[0, 0], [0, 0]])
        return value, grad.transpose(), hess

    return value, grad.transpose()


def ineq_constraint_2_3(x, flag):
    value = x[0] - 2
    grad = np.array([1, 0])

    if flag:
        hess = np.array([[0, 0], [0, 0]])
        return value, grad.transpose(), hess

    return value, grad.transpose()


def ineq_constraint_2_4(x, flag):
    value = -x[1]
    grad = np.array([0, -1])

    if flag:
        hess = np.array([[0, 0], [0, 0]])
        return value, grad.transpose(), hess

    return value, grad.transpose()
