import sys

# sys.path.append("C:/Projects/HW/Numerical-Optimization-with-Python")
sys.path.append("/Users/ronibendom/Master/Numerical Optimization with Python/")

from final_proj.src.unconstrained_min import LineSearchMinimization
from final_proj.src.utils import plot_contours, plot_iterations
from final_proj.tests.examples import (
    circles,
    ellipses,
    rotated_ellipses,
    rosenbrock,
    triangles,
)

import unittest
import numpy as np


class TestLineSearchMethods(unittest.TestCase):
    START_POINT = np.array([1, 1], dtype=np.float64)
    ROSENBROCK_START_POINT = np.array([-1, 2], dtype=np.float64)
    gd_minimizer = LineSearchMinimization("Gradient descent")
    newton_minimizer = LineSearchMinimization("Newton")
    quasi_newton_minimizer = LineSearchMinimization("Quasi-Newton")


    def test_circles(self):
        (
            x_newton,
            f_x_newton,
            x_s_newton,
            obj_values_newton,
            success_newton,
        ) = self.newton_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, circles functoin - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}"
        )
        print(f"stoped after {len(x_s_newton) - 1} steps")

        (
            x_quasi_newton,
            f_x_quasi_newton,
            x_s_quasi_newton,
            obj_values_quasi_newton,
            success_quasi_newton,
        ) = self.quasi_newton_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, circles functoin - quasi newton: {x_quasi_newton}, value: {f_x_quasi_newton}, success: {success_quasi_newton}"
        )
        print(f"stoped after {len(x_s_quasi_newton) - 1} steps")

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, circles functoin - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}"
        )
        print(f"stoped after {len(x_s_gd) - 1} steps")

        plot_contours(
            circles, "Convergence over circular contour lines", x_s_gd, x_s_newton, x_s_quasi_newton
        )
        plot_iterations(
            "Objective function values of quadratic function 1 - Circular contour lines",
            obj_values_gd,
            obj_values_newton,
            obj_values_quasi_newton
        )

    def test_ellipses(self):
        (
            x_newton,
            f_x_newton,
            x_s_newton,
            obj_values_newton,
            success_newton,
        ) = self.newton_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, ellipses function - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}"
        )
        print(f"stoped after {len(x_s_newton) - 1} steps")

        (
            x_quasi_newton,
            f_x_quasi_newton,
            x_s_quasi_newton,
            obj_values_quasi_newton,
            success_quasi_newton,
        ) = self.quasi_newton_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-30, 100
        )
        print(
            f"point of convergence, ellipses function - quasi newton: {x_quasi_newton}, value: {f_x_quasi_newton}, success: {success_quasi_newton}"
        )
        print(f"stoped after {len(x_s_quasi_newton) - 1} steps")

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, ellipses function - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}"
        )
        print(f"stoped after {len(x_s_gd) - 1} steps")

        plot_contours(
            ellipses, "Convergence over elliptical contour lines", x_s_gd, x_s_newton, x_s_quasi_newton
        )
        plot_iterations(
            "Objective function values of quadratic function 2 - Elliptical contour lines",
            obj_values_gd,
            obj_values_newton,
            obj_values_quasi_newton
        )

    def test_rotated_ellipses(self):
        (
            x_newton,
            f_x_newton,
            x_s_newton,
            obj_values_newton,
            success_newton,
        ) = self.newton_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, rotated ellipses function - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}"
        )
        print(f"stoped after {len(x_s_newton) - 1} steps")

        (
            x_quasi_newton,
            f_x_quasi_newton,
            x_s_quasi_newton,
            obj_values_quasi_newton,
            success_quasi_newton,
        ) = self.quasi_newton_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-30, 100
        )
        print(
            f"point of convergence, rotated ellipses function - quasi newton: {x_quasi_newton}, value: {f_x_quasi_newton}, success: {success_quasi_newton}"
        )
        print(f"stoped after {len(x_s_quasi_newton) - 1} steps")

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, rotated ellipses function - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}"
        )
        print(f"stoped after {len(x_s_gd) - 1} steps")

        plot_contours(
            rotated_ellipses,
            "Convergence over rotated elliptical contour lines",
            x_s_gd,
            x_s_newton,
            x_s_quasi_newton
        )
        plot_iterations(
            "Objective function values of quadratic function 3 - Rotated Elliptical contour lines",
            obj_values_gd,
            obj_values_newton,
            obj_values_quasi_newton
        )

    def test_rosenbrock(self):
        (
            x_newton,
            f_x_newton,
            x_s_newton,
            obj_values_newton,
            success_newton,
        ) = self.newton_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            100,
        )
        print(
            f"point of convergence, rosenbrock function - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}"
        )
        print(f"stoped after {len(x_s_newton) - 1} steps")

        (
            x_quasi_newton,
            f_x_quasi_newton,
            x_s_quasi_newton,
            obj_values_quasi_newton,
            success_quasi_newton,
        ) = self.quasi_newton_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            10000
        )
        print(
            f"point of convergence, rosenbrock function - quasi newton: {x_quasi_newton}, value: {f_x_quasi_newton}, success: {success_quasi_newton}"
        )
        print(f"stoped after {len(x_s_quasi_newton) - 1} steps")

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            rosenbrock,
            self.ROSENBROCK_START_POINT.transpose(),
            "wolfe",
            10e-12,
            10e-8,
            10000,
        )
        print(
            f"point of convergence, rosenbrock function - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}"
        )
        print(f"stoped after {len(x_s_gd) - 1} steps")

        plot_contours(
            rosenbrock,
            "Convergence over Rosenbrock function contour lines",
            x_s_gd,
            x_s_newton,
            x_s_quasi_newton
        )
        plot_iterations(
            "Objective function values of Rosenbrock function",
            obj_values_gd,
            obj_values_newton,
            obj_values_quasi_newton
        )

    def test_triangles(self):
        (
            x_newton,
            f_x_newton,
            x_s_newton,
            obj_values_newton,
            success_newton,
        ) = self.newton_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, triangles function - newton: {x_newton}, value: {f_x_newton}, success: {success_newton}"
        )
        print(f"stoped after {len(x_s_newton) - 1} steps")

        (
            x_quasi_newton,
            f_x_quasi_newton,
            x_s_quasi_newton,
            obj_values_quasi_newton,
            success_quasi_newton,
        ) = self.quasi_newton_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-30, 100
        )
        print(
            f"point of convergence, triangles function - quasi newton: {x_quasi_newton}, value: {f_x_quasi_newton}, success: {success_quasi_newton}"
        )
        print(f"stoped after {len(x_s_quasi_newton) - 1} steps")

        x_gd, f_x_gd, x_s_gd, obj_values_gd, success_gd = self.gd_minimizer.minimize(
            triangles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )
        print(
            f"point of convergence, triangles function - GD: {x_gd}, value: {f_x_gd}, success: {success_gd}"
        )
        print(f"stoped after {len(x_s_gd) - 1} steps")

        plot_contours(
            triangles,
            "Convergence over smothed corners triangles contour lines",
            x_s_gd,
            x_s_newton,
            x_s_quasi_newton
        )
        plot_iterations(
            "Objective function values of smothed corners triangles function",
            obj_values_gd,
            obj_values_newton,
            obj_values_quasi_newton
        )


if __name__ == "__main__":
    unittest.main()
