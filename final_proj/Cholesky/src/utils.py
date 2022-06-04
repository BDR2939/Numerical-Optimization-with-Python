import numpy as np
import matplotlib.pyplot as plt


def plot_contours(f, title, xy_gd=None, xy_newton=None, xy_quasi_newton=None):

    if xy_gd is not None:
        x_gd = [xy_gd[i][0] for i in range(len(xy_gd))]
        y_gd = [xy_gd[i][1] for i in range(len(xy_gd))]

    if xy_newton is not None:
        x_newton = [xy_newton[i][0] for i in range(len(xy_newton))]
        y_newton = [xy_newton[i][1] for i in range(len(xy_newton))]

    if xy_quasi_newton is not None:
        x_quasi_newton = [xy_quasi_newton[i][0] for i in range(len(xy_quasi_newton))]
        y_quasi_newton = [xy_quasi_newton[i][1] for i in range(len(xy_quasi_newton))]

    if xy_gd is not None and xy_newton is not None and xy_quasi_newton is not None:
        xlist = np.linspace(
            min([min(x_gd), min(x_newton), min(x_quasi_newton)]), max([max(x_gd), max(x_newton), max(x_quasi_newton)]), 100
        )
        ylist = np.linspace(
            min([min(y_gd), min(y_newton), min(y_quasi_newton)]), max([max(y_gd), max(y_newton), max(y_quasi_newton)]), 100
        )

    elif xy_gd is not None:
        xlist = np.linspace(min(x_gd), max(x_gd), 100)
        ylist = np.linspace(min(y_gd), max(y_gd), 100)
    
    elif xy_quasi_newton is not None:
        xlist = np.linspace(min(x_quasi_newton), max(x_quasi_newton), 100)
        ylist = np.linspace(min(y_quasi_newton), max(y_quasi_newton), 100)

    else:
        xlist = np.linspace(min(x_newton), max(x_newton), 100)
        ylist = np.linspace(min(y_newton), max(y_newton), 100)

    X, Y = np.meshgrid(xlist, ylist)

    Z = np.full(X.shape, None)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]), False)[0]

    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(X, Y, Z)
    ax.clabel(cp, inline=True, fontsize=10)
    if xy_gd is not None:
        ax.plot(
            x_gd,
            y_gd,
            label="Gradient descent steps",
            color="r",
            marker=".",
            linestyle="--",
        )
    if xy_newton is not None:
        ax.plot(
            x_newton,
            y_newton,
            label="Newton's method steps",
            color="k",
            marker=".",
            linestyle="--",
        )
    if xy_quasi_newton is not None:
        ax.plot(
            x_quasi_newton,
            y_quasi_newton,
            label="Quasi Newton's method steps",
            color="b",
            marker=".",
            linestyle="--",
        )
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_iterations(title, obj_values_gd=None, obj_values_newton=None, obj_values_quasi_newton=None):

    fig, ax = plt.subplots()
    if obj_values_gd is not None:
        ax.plot(range(len(obj_values_gd)), obj_values_gd, label="Gardient descent")

    if obj_values_newton is not None:
        ax.plot(
            range(len(obj_values_newton)), obj_values_newton, label="Newton's method"
        )

    if obj_values_quasi_newton is not None:
        ax.plot(
            range(len(obj_values_quasi_newton)), obj_values_quasi_newton, label="Quasi Newton method"
        )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("# iterations")
    ax.set_ylabel("Objective function value")
    plt.show()
