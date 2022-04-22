import sys

# sys.path.append("C:/Projects/HW/Numerical-Optimization-with-Python")
sys.path.append('/Users/ronibendom/Master/Numerical Optimization with Python/')

from HW1_programing.src.unconstrained_min import LineSearchMinimization
from HW1_programing.src.utils import (
    plot_contours,
    plot_iterations
)
from HW1_programing.tests.examples import (
    circles,
    ellipses,
    rotated_ellipses,
    rosenbrock,
    linear,
    triangles,
)

import unittest
import numpy as np


class TestLineSearchMethods(unittest.TestCase):
    START_POINT = np.array([1, 1])
    ROSENBROCK_START_POINT = np.array([-1, 2])
    gd_minimizer = LineSearchMinimization("Gradient descent")
    newton_minimizer = LineSearchMinimization("Newton")

    def test_circles(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}')

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(circles, 'Convergence over circular contour lines', x_s_gd, x_s_newton)
        plot_iterations('Objective function values of quadratic function 1 - Circular contour lines', obj_values_gd, obj_values_newton)


    def test_ellipses(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}')

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(ellipses, 'Convergence over elliptical contour lines', x_s_gd, x_s_newton)
        plot_iterations('Objective function values of quadratic function 2 - Elliptical contour lines', obj_values_gd, obj_values_newton)


    def test_rotated_ellipses(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}')

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(rotated_ellipses, 'Convergence over roatated elliptical contour lines', x_s_gd, x_s_newton,)
        plot_iterations('Objective function values of quadratic function 3 - Roatated Elliptical contour lines', obj_values_gd, obj_values_newton)


    def test_rosenbrock(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            100,
        )
        print(f'point of convergence - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}')

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            10000,
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(rosenbrock, 'Convergence over Rosenbrock function contour lines',  x_s_gd, x_s_newton)
        plot_iterations('Objective function values of Rosenbrock function', obj_values_gd, obj_values_newton)


    def test_linear(self):
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd =  self.gd_minimizer.minimize(
            linear, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(linear, 'Convergence over linear function contour lines',  x_s_gd)
        plot_iterations('Objective function values of linear function', obj_values_gd)


    def test_triangles(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}')

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(f'point of convergence - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}')

        plot_contours(triangles, 'Convergence over smothed corners triangles contour lines', x_s_gd, x_s_newton)
        plot_iterations('Objective function values of smothed corners triangles function', obj_values_gd, obj_values_newton)


if __name__ == "__main__":
    unittest.main()
