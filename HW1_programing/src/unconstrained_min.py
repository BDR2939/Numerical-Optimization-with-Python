import numpy as np


class LineSearchMinimization:

    WOLFE_COND_COSNT = 0.01
    BACKTRACKING_CONST = 0.5

    def __init__(self, method) -> None:
        self.method = method

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        if self.method == "Newton":
            return self.__minimize_newton(f, x0, step_len, obj_tol, param_tol, max_iter)

        else:
            return self.__minimize_gd(f, x0, step_len, obj_tol, param_tol, max_iter)

    def __minimize_newton(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        x = x0
        f_x, g_x, h_x = f(x, True)

        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

        x_prev = x
        f_prev = f_x

        x_s = [x0]
        obj_values = [f_x]

        iter = 0
        while iter < max_iter:

            if abs(x - x_prev) < param_tol:
                return x, f_x, x_s, obj_values, True

            p = np.linalg.solve(h_x, -g_x)

            if f_prev - f_x < obj_tol or (p.transpose * h_x * p) ** 0.5 < obj_tol:
                return x, f_x, x_s, obj_values, True

            if step_len == "wolfe":
                alpha = self.__wolfe(f, p, x)

            else:
                alpha = step_len

            x_prev = x
            f_prev = f

            x = x + alpha * p
            f_x, g_x, h_x = f(x, True)
            print(f"i = {iter + 1}, x{iter + 1} = {x}, f(x{iter + 1}) = {f_x}")

            x_s.append(x)
            obj_values.append(f_x)

        return x, f_x, x_s, obj_values, False

    def __minimize_gd(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        x = x0
        f_x, g_x = f(x, False)

        print(f"i = 0, x0 = {x}, f(x0) = {f_x}")

        x_prev = x
        f_prev = f_x

        x_s = [x0]
        obj_values = [f_x]

        iter = 0
        while iter < max_iter:

            if abs(x - x_prev) < param_tol:
                return x, f_x, x_s, obj_values, True

            if f_prev - f_x < obj_tol:
                return x, f_x, x_s, obj_values, True

            p = -g_x
            if step_len == "wolfe":
                alpha = self.__wolfe(f, p, x)

            else:
                alpha = step_len

            x_prev = x
            f_prev = f

            x = x + alpha * p
            f_x, g_x = f(x, False)

            print(f"i = {iter + 1}, x{iter + 1} = {x}, f(x{iter + 1}) = {f_x}")

            x_s.append(x)
            obj_values.append(f_x)

        return x, f_x, x_s, obj_values, False

    def __wolfe(self, f, p, x) -> int:
        alpha = 1

        while (
            f(x + alpha * p)[0]
            > f(x, False)[0] + self.WOLFE_COND_COSNT * alpha * f(x, False)[1] * p
        ):
            alpha = alpha * self.BACKTRACKING_CONST

        return alpha
