from importlib_metadata import method_cache
import numpy as np
from sympy import true


class LineSearchMinimization:
    
    def __init__(self, method) -> None:
        self.method = method
        WOLFE_COND_COSNT = 0.01
        BACKTRACKING_CONST = 0.5

    def minimize(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        if self.method == 'Newton':
            x_s, values = self.minimize_newton(f, x0, step_len, obj_tol, param_tol, max_iter)
        
        else:
            x_s, values = self.minimize_gd(f, x0, step_len, obj_tol, param_tol, max_iter)

    def minimize_newton(self, f, x0, step_len, obj_tol, param_tol, max_iter):
        pass

    
    def minimize_gd(self, f, x0, step_len, obj_tol, param_tol, max_iter):

        # iter = 0
        # while iter < max_iter:
        pass

    def check_stop_conds_newton(self, obj_tol, param_tol, x, x_prev, p, h) -> bool:
        if abs(x - x_prev) < param_tol:
            return True
        

    def check_stop_conds_gd(self, obj_tol, param_tol, x, x_prev, f_x, f_prev) -> bool:
        if abs(x - x_prev) < param_tol:
            return True
        
        if f_prev - f_x < obj_tol:
            return True
        
        return False
        

    def wolfe(self, f, p, x) -> int:
        alpha = 1

        while f(x + alpha * p)[0] > f(x)[0] + self.WOLFE_COND_COSNT * alpha * f(x)[1] * p:
            alpha = alpha * self.BACKTRACKING_CONST