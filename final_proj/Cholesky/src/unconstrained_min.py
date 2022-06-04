import sys

import numpy as np

sys.path.append("/Users/ronibendom/Master/Numerical Optimization with Python/")

from final_proj.Cholesky.src.Cholesky_decomp import cholesky_decomp

class LineSearchMinimization:

    WOLFE_COND_COSNT = 0.01
    BACKTRACKING_CONST = 0.5

    def __init__(self, method) -> None:
        self.method = method

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        x = x0
        if self.method == "Newton":
            f_x, g_x, h_x = f(x, True)
        elif self.method == "Quasi-Newton":
            f_x, g_x= f(x, False)
            L = np.identity(len(g_x))
        else:
            f_x, g_x = f(x, False)

        x_prev = x
        f_prev = f_x

        x_s = [x0]
        obj_values = [f_x]

        iter = 0
        while iter < max_iter:

            if iter != 0 and sum(abs(x - x_prev)) < param_tol:
                return x, f_x, x_s, obj_values, True

            if self.method == "Newton":
                p = np.linalg.solve(h_x, -g_x)
                _lambda = np.matmul(p.transpose(), np.matmul(h_x, p)) ** 0.5
                if 0.5 * (_lambda ** 2) < obj_tol:
                    return x, f_x, x_s, obj_values, True

            elif self.method == "Quasi-Newton":
                t = np.linalg.solve(L, -g_x)
                p = np.linalg.solve(L.T, t)

                # p = np.linalg.solve(B, -g_x)

                _lambda = np.matmul(p.transpose(), np.matmul(L, p)) ** 0.5
                if 0.5 * (_lambda ** 2) < obj_tol:
                    return x, f_x, x_s, obj_values, True

            else:
                p = -g_x

            if iter != 0 and (f_prev - f_x < obj_tol):
                return x, f_x, x_s, obj_values, True

            if step_len == "wolfe":
                alpha = self.__wolfe(f, p, x)

            else:
                alpha = step_len

            x_prev = x
            f_prev = f_x

            x = x + alpha * p
            if self.method == "Newton":
                f_x, g_x, h_x = f(x, True)
            elif self.method == "Quasi-Newton":
                g_prev = g_x
                f_x, g_x = f(x, False)
                s = alpha * p
                y = g_x - g_prev

                L = self.BFGS_update_for_L(L, s, y)
            else:
                f_x, g_x = f(x, False)

            x_s.append(x)
            obj_values.append(f_x)

            iter += 1

        return x, f_x, x_s, obj_values, False

    def __wolfe(self, f, p, x) -> int:
        alpha = 1

        while f(x + alpha * p, False)[0] > f(x, False)[
            0
        ] + self.WOLFE_COND_COSNT * alpha * np.matmul(f(x, False)[1].transpose(), p):
            alpha = alpha * self.BACKTRACKING_CONST

        return alpha
    
    def BFGS_update(self, B, s, y):
        epsilon = 1e-12
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        pos_term = np.matmul(y, y.T) / (np.matmul(y.T, s) + epsilon)

        neg_term = np.matmul(B, np.matmul(s, np.matmul(s.T, B)))
        neg_term = neg_term / (np.matmul(s.T, np.matmul(B, s)) + epsilon)

        return B + pos_term - neg_term

    def BFGS_update_for_L(self, L, s, y):
        epsilon = 1e-12
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        B = np.matmul(L, L.T)

        y_T_s = np.matmul(y.T, s)
        s_T_B_s = np.matmul(s.T, np.matmul(B, s))
        beta = (y_T_s / s_T_B_s) ** 0.5

        first_upper_term = np.matmul(np.matmul(L.T, s), y.T)
        second_upper_term = beta * np.matmul(np.matmul(np.matmul(L.T, s), s.T), B.T)
        lower_term = beta * np.matmul(s.T ,np.matmul(B, s)) + epsilon

        J_T = L.T + first_upper_term / lower_term - second_upper_term / lower_term

        return np.linalg.qr(J_T, mode='r').T

