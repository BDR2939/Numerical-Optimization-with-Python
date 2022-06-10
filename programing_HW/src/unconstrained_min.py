import numpy as np


class LineSearchMinimization:

    WOLFE_COND_COSNT = 0.01
    BACKTRACKING_CONST = 0.5

    def __init__(self, method) -> None:
        self.method = method

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        x = x0
        f_x, g_x, h_x = f(x, True)

        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

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
            else:
                f_x, g_x = f(x, False)

            print(f"i = {iter + 1}, x{iter + 1} = {x}, f(x{iter + 1}) = {f_x}")

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
