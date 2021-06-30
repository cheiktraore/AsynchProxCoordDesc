import numpy as np
from numba import njit
from numpy.linalg import norm
from multiprocessing import shared_memory
# import time

@njit
def ST(x, u):
    """Soft-thresholding of vector x at level u, i.e., entrywise:
    x_i + u_i if x_i < -u_i, x_i - u_i if x_i > u_i and 0 else.
    """
    # TODO implement without if, using only np.abs, np.sign ad np.maximum
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


@njit
def lasso_loss(A, b, lbda, x):
    """Value of Lasso objective at x."""
    # TODO implement with the norm function imported above/
    return norm(A @ x - b) ** 2 / 2. + lbda * norm(x, ord=1)


def choose_coord(n_features, low_var=3, high_var=19, low_mean=6, high_mean=19, proc_nums=9):
    # sigma = np.random.randint(low_var,high_var)
    # mu = np.random.randint(low_mean,high_mean)
    # i_k = sigma * np.random.randn() + mu
    # i_k = np.int(np.floor(i_k) % proc_nums)
    i_k = np.random.randint(n_features)
    return i_k


def PRBCFB(A, b, x, lbda, max_iter, max_it, i_k):
    max_iteration = shared_memory.ShareableList(name=max_iter.shm.name)
    # global max_iter
    A = np.asfortranarray(A)
    n, m = A.shape
    gammas = 1. / norm(A, axis=0) ** 2

    all_x = np.zeros((max_it, m))
    u = A @ x - b  # negative residuals
    fx = np.zeros(max_it)
    indx = i_k
    # for t in range(max_iter):
    t = 0
    fx[t] = lasso_loss(A, b, lbda, x)
    all_x[t] = x
    while ((max_iteration[0]-1) > 0):
        old_x = x.copy()
        x[indx] = ST(x[indx] - A[:, indx].T @ u * gammas[indx],
                        lbda * gammas[indx])
        u += A[:, indx] * (x[indx] - old_x[indx])
        try:
            max_iteration[0] = max_iteration[0] - 1
        except ValueError:
            pass
        indx = choose_coord(n_features=A.shape[1])
        t += 1
        # END TODO
        try:
            all_x[t] = x
        except IndexError:
            print(max_iteration[0])
            raise IndexError
        fx[t] = lasso_loss(A, b, lbda, x)
    max_iteration.shm.close()
    return x, all_x, fx
