import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import warnings


def plot_contours(f, title, xy_gd=None, xy_newton=None):

    if xy_gd is not None:
        x_gd = [xy_gd[i][0] for i in range(len(xy_gd))]
        y_gd = [xy_gd[i][1] for i in range(len(xy_gd))]

    if xy_newton is not None:
        x_newton = [xy_newton[i][0] for i in range(len(xy_newton))]
        y_newton = [xy_newton[i][1] for i in range(len(xy_newton))]

    if xy_gd is not None and xy_newton is not None:
        xlist = np.linspace(
            min([min(x_gd), min(x_newton)]), max([max(x_gd), max(x_newton)]), 100
        )
        ylist = np.linspace(
            min([min(y_gd), min(y_newton)]), max([max(y_gd), max(y_newton)]), 100
        )

    elif xy_gd is not None:
        xlist = np.linspace(min(x_gd), max(x_gd), 100)
        ylist = np.linspace(min(y_gd), max(y_gd), 100)

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
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_iterations(
    title, obj_values_1=None, obj_values_2=None, label_1=None, label_2=None
):

    fig, ax = plt.subplots()
    if obj_values_1 is not None:
        ax.plot(range(len(obj_values_1)), obj_values_1, label=label_1)

    if obj_values_2 is not None:
        ax.plot(range(len(obj_values_2)), obj_values_2, label=label_2)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("# iterations")
    ax.set_ylabel("Objective function value")
    plt.show()


def plot_feasible_set_2d(path_points):
    # plot the feasible region
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(
        ((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    # plot the lines defining the constraints
    x = np.linspace(0, 4, 2000)
    # y >= -x + 1
    y1 = -x + 1
    # y <= 1
    y2 = np.ones(x.size)
    # y >= 0
    y3 = np.zeros(x.size)

    if path_points is not None:
        x_path = [path_points[i][0] for i in range(len(path_points))]
        y_path = [path_points[i][1] for i in range(len(path_points))]

    # Make plot
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(np.ones(x.size) * 2, x)
    plt.plot(
        x_path,
        y_path,
        label="algorithm's path",
        color="k",
        marker=".",
        linestyle="--",
    )
    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")


def plot_feasible_set_3d(path_points):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig = plt.figure()
        ax = Axes3D(fig)
        x = [1, 0, 0]
        y = [0, 1, 0]
        z = [0, 0, 1]
        verts = [list(zip(x, y, z))]
        poly_3d_collection = Poly3DCollection(verts, alpha=0.5, edgecolors="k")
        ax.add_collection3d(poly_3d_collection)

        if path_points is not None:
            x_path = [path_points[i][0] for i in range(len(path_points))]
            y_path = [path_points[i][1] for i in range(len(path_points))]
            z_path = [path_points[i][2] for i in range(len(path_points))]

        ax.plot(
            x_path,
            y_path,
            z_path,
            label="algorithm's path",
            color="k",
            marker=".",
            linestyle="--",
        )

        plt.show()
