from turtle import color
import numpy as np
import matplotlib.pyplot as plt

def plot_contours(f, xy_gd, xy_newton,  title):

    x_gd = [xy_gd[i][0] for i in range(len(xy_gd))]
    y_gd = [xy_gd[i][1] for i in range(len(xy_gd))]

    x_newton = [xy_newton[i][0] for i in range(len(xy_newton))]
    y_newton= [xy_newton[i][1] for i in range(len(xy_newton))]

    xlist = np.linspace(min([min(x_gd), min(x_newton)]), max([max(x_gd), max(x_newton)]), 100)
    ylist = np.linspace(min([min(y_gd), min(y_newton)]), max([max(y_gd), max(y_newton)]), 100)

    X, Y = np.meshgrid(xlist, ylist)

    Z = np.full(X.shape, None)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]), False)[0]

    fig,ax=plt.subplots(1,1)
    cp = ax.contour(X, Y, Z)
    ax.plot(x_gd, y_gd, label = 'Gradient descent steps', color = 'r')
    ax.plot(x_newton, y_newton, label = 'Newton\'s method steps', color = 'k')
    ax.legend()
    plt.show()