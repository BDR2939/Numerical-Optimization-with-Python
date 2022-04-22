import sys

# sys.path.append("C:/Projects/HW/Numerical-Optimization-with-Python")
sys.path.append('/Users/ronibendom/Master/Numerical Optimization with Python/')

from HW1_programing.src.unconstrained_min import LineSearchMinimization
from HW1_programing.src.utils import plot_contours
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

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        plot_contours(circles, x_s_gd, x_s_newton, 'gd circles')

    def test_ellipses(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        plot_contours(ellipses, x_s_gd, x_s_newton, 'gd circles')


    def test_rotated_ellipses(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        plot_contours(rotated_ellipses, x_s_gd, x_s_newton, 'gd circles')


    def test_rosenbrock(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            100,
        )

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            10000,
        )

        plot_contours(rosenbrock, x_s_gd, x_s_newton, 'gd circles')


    def test_linear(self):
        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd =  self.gd_minimizer.minimize(
            linear, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

    def test_triangles(self):
        x_newton, f_x_newton, x_s_newton, obj_values_newton, success_newton = self.newton_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        plot_contours(triangles, x_s_gd, x_s_newton, 'gd circles')




if __name__ == "__main__":
    unittest.main()
