import numpy as np
from numba import njit
from numpy.linalg import norm
import pickle

# Functions save_object, load_object credited to https://www.askpython.com/python/examples/save-data-in-python.
def save_object(obj):
    try:
        with open("data_save.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

@njit
def ST(x, u):
    """Soft-thresholding of vector x at level u, i.e., entrywise:
    x_i + u_i if x_i < -u_i, x_i - u_i if x_i > u_i and 0 else.
    """
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


@njit
def lasso_loss(A, b, lbda, x):
    """Value of Lasso objective at x."""
    return norm(A @ x - b) ** 2 / 2. + lbda * norm(x, ord=1)


def choose_coord(n_features, low_var=3, high_var=19, low_mean=6, high_mean=19,
                 proc_nums=9, distr='uniform'):
    # sigma = np.random.randint(low_var,high_var)
    # mu = np.random.randint(low_mean,high_mean)
    # i_k = sigma * np.random.randn() + mu
    # i_k = np.int(np.floor(i_k) % proc_nums)
    if distr == 'uniform'.strip():
        i_k = np.random.randint(n_features)
    return i_k