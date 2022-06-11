import sys
from token import LPAR

# sys.path.append("C:/Projects/HW/Numerical-Optimization-with-Python")
sys.path.append("/Users/ronibendom/Master/Numerical Optimization with Python/")

from programing_HW.src.constrained_min import InteriorPointMinimization
from programing_HW.src.utils import plot_contours, plot_iterations, plot_feasible_set_2d
from programing_HW.tests.examples import (
    qp,
    ineq_constraint_1_1,
    ineq_constraint_1_2,
    ineq_constraint_1_3,
    lp,
    ineq_constraint_2_1,
    ineq_constraint_2_2,
    ineq_constraint_2_3,
    ineq_constraint_2_4,
)

import unittest
import numpy as np


class TestInteriorPointMethod(unittest.TestCase):
    START_POINT_qp = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    START_POINT_lp = np.array([0.5, 0.75], dtype=np.float64)
    minimizer = InteriorPointMinimization()

    def test_qp(self):
        eq_constraint_mat = np.array([1, 1, 1]).reshape(1, -1)
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.minimize(
            qp,
            self.START_POINT_qp,
            [
                ineq_constraint_1_1,
                ineq_constraint_1_2,
                ineq_constraint_1_3,
            ],
            eq_constraint_mat,
            np.array([1]),
            "wolfe",
            10e-12,
            10e-8,
            100,
            20,
            10e-10,
        )

        plot_iterations(
            "Objective function values of qp function",
            outer_obj_values,
            obj_values,
            "Outer objective values",
            "Objective values",
        )

    def test_lp(self):
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.minimize(
            lp,
            self.START_POINT_lp,
            [
                ineq_constraint_2_1,
                ineq_constraint_2_2,
                ineq_constraint_2_3,
                ineq_constraint_2_4,
            ],
            np.array([]),
            np.array([]),
            "wolfe",
            10e-12,
            10e-8,
            100,
            20,
            10e-10,
        )

        print(f"Point of convergence: {x_s[-1]}")
        print(f"Objective value at point of convergence: {lp(x_s[-1], False)[0]}")
        print(
            f"-y -x +1 <= 0 value at point of convergence: {ineq_constraint_2_1(x_s[-1], False)[0]}"
        )
        print(
            f"y - 1 <= 0 value at point of convergence: {ineq_constraint_2_2(x_s[-1], False)[0]}"
        )
        print(
            f"x - 2 <= 0 value at point of convergence: {ineq_constraint_2_3(x_s[-1], False)[0]}"
        )
        print(
            f"-y <= 0 value at point of convergence: {ineq_constraint_2_4(x_s[-1], False)[0]}"
        )

        plot_feasible_set_2d(x_s)

        plot_iterations(
            "Objective function values of lp function",
            outer_obj_values,
            obj_values,
            "Outer objective values",
            "Objective values",
        )


if __name__ == "__main__":
    unittest.main()
