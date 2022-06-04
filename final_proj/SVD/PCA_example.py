import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm


def feature_normalize(x):
    mu = np.mean(x, axis=0)
    x_norm = x - mu
    sigma = np.std(x, axis=0)
    x_norm = x_norm / sigma
    return x_norm, mu, sigma


def pca(x):
    m, n = x.shape
    cov_mat = 1 / m * x.T.dot(x)
    u, s, vh = np.linalg.svd(cov_mat)
    return u, s, vh


def display_data(x,fig):
    m, n = x.shape
    example_width = np.round(np.sqrt(x.shape[1]))
    example_height = n / example_width
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)
    pad = 1
    display_array = - np.ones((int(pad + display_rows * (example_height + pad)), int(pad + display_cols * (example_width + pad))))
    curr_ex = 0
    for j in range(int(display_rows)):
        for i in range(int(display_cols)):
            if curr_ex > m:
                break
            max_val = np.max(np.abs(x[curr_ex, :]))
            r_f = int(pad + j * (example_height + pad))
            r_e = int(pad + j * (example_height + pad) + example_height)
            c_f = int(pad + i * (example_width + pad))
            c_e = int(pad + i * (example_width + pad) + example_width)
            display_array[r_f:r_e, c_f:c_e] = np.reshape(x[curr_ex, :], (int(example_height), int(example_width))) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break
    fig.imshow(display_array.T,cmap='gray')


def project_data(x_norm, u, k):
    u_red = u[:, 0:k]
    z = np.matmul(x_norm, u_red)
    return z


def recover_data(z, u, k):
    u_red = u[:, 0:k]
    x_rec = np.matmul(z, u_red.T)
    return x_rec


def generate_correlated_rvs():
    num_samples = 50
    r = np.array([[1.45, 0.9],
                  [0.9, 1.05]])
    b = np.array([4,5])
    x = norm.rvs(size=(2, num_samples))
    c = cholesky(r, lower=True)
    y = np.dot(c, x).T + b
    return y


if __name__ == '__main__':
    # 2d example of dimensionality reduction
    x_2d = generate_correlated_rvs()
    x_2d_norm, mu, _ = feature_normalize(x_2d)
    u, s, _ = pca(x_2d_norm)
    y1 = mu + 1.5*s[0] * u[:,0].T
    y2 = mu + 1.5*s[1] * u[:,1].T

    plt.scatter(x_2d[:, 0], x_2d[:, 1],label='Data')
    plt.plot(np.vstack((mu,y1))[:,0],np.vstack((mu,y1))[:,1],'orange',label='First component')
    plt.plot(np.vstack((mu,y2))[:,0],np.vstack((mu,y2))[:,1],'r',label='Second component')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('2D dataset')
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(x_2d_norm[:,0],x_2d_norm[:,1],label='Data')
    k = 1
    z = project_data(x_2d_norm, u, k)

    x_2d_rec = recover_data(z, u, k)
    plt.scatter(x_2d_rec[:,0],x_2d_rec[:,1],label='Projected data')
    for i in range(x_2d_norm.shape[0]):
        p = np.vstack((x_2d_norm[i,:],x_2d_rec[i,:]))
        plt.plot(p[:,0],p[:,1],color='red', linestyle='dashed')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Projected data')
    plt.legend()
    plt.show()

    ## PCA on images
    # load data and plot
    x = pd.read_csv(r'C:\Users\tmrsh\PycharmProjects\pythonProject\Optimization Final Project\faces.csv')
    x = x.values
    fig, ax = plt.subplots()
    display_data(x[0:100, :],ax)
    plt.title('Original images')
    plt.show()

    # Perform PCA
    x_norm, mu, sigma = feature_normalize(x)
    u, s, vh = pca(x_norm)

    # plot components variance
    plt.figure()
    plt.stem(np.arange(0,120),s[0:120])
    plt.title('First 120 eigenvalues of $\Sigma$')
    plt.xlabel('Matrix diagonal entries [i,i]')
    plt.ylabel(r'$\lambda_i$')
    plt.show()

    # show principal components image
    fig, ax = plt.subplots()
    display_data(u[:, 0:35].T,ax)
    plt.title('Principal components only')
    plt.show()
    k = 110
    z = project_data(x_norm, u, k)

    # show recovered data
    fig, ax = plt.subplots(1, 2)

    x_rec = recover_data(z, u, k)
    display_data(x[0:100, :],ax[0])
    ax[0].set_title('Original images')

    display_data(x_rec[0:100, :],ax[1])
    ax[1].set_title('Recovered data from 100-dimensional projected vectors')
    plt.show()
    a=1
