import numpy as np


def cholesky_decomp(A):
    """
    compute the Cholesky decomposition of a real, squared, symetric matrix

    Args:
    A: Real valued NxN symetric matrix

    Returns:
    L: lower triangular matrix
    """

    n = A.shape[0]
    L = np.zeros_like(A)

    L[:, 0] = A[:, 0] / (A[0, 0] ** 0.5)

    for i in range(1, n):
        for j in range(1, i + 1):
            sum = np.matmul(L[i, :], L[j, :].transpose())

            if i == j:
                L[i, j] = (A[i, j] - sum) ** 0.5

            else:
                L[i, j] = (A[i, j] - sum) / L[j, j]

    return L
