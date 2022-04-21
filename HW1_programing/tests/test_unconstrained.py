import sys

sys.path.append("C:/Projects/HW/Numerical-Optimization-with-Python")

from HW1_programing.src.unconstrained_min import LineSearchMinimization
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

    def test_circles(self):
        gd_minimizer = LineSearchMinimization("Gradient descent")
        newton_minimizer = LineSearchMinimization("Newton")

        newton_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        gd_minimizer.minimize(
            circles, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

    def test_ellipses(self):
        gd_minimizer = LineSearchMinimization("Gradient descent")
        newton_minimizer = LineSearchMinimization("Newton")

        newton_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        gd_minimizer.minimize(
            ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

    def test_rotated_ellipses(self):
        gd_minimizer = LineSearchMinimization("Gradient descent")
        newton_minimizer = LineSearchMinimization("Newton")

        newton_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )

        gd_minimizer.minimize(
            rotated_ellipses, self.START_POINT.transpose(), "wolfe", 10e-12, 10e-8, 100
        )


if __name__ == "__main__":
    unittest.main()
