import threading
import numpy as np
from numpy.linalg import norm
from multiprocessing import cpu_count
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

from async_prox_coord_desc import ST

from celer.datasets import make_correlated_data
from celer.utils import configure_plt
configure_plt()

n, m, s = 50, 100, 5

max_it = 50_000

A, y, x_true = make_correlated_data(100, 500, density=0.1, random_state=0)

lbda_max = norm(A.T @ y, ord=np.inf)
lbda = lbda_max / 100

N_PROC = cpu_count()
x = np.zeros(m)
R = y.copy()
lc = norm(A, axis=0) ** 2


# TODO maybe a closure and the threads are started inside
def cd_concu(A, lbda):
    global max_iter
    global x
    global R
    global lc
    while max_iter > 0:
        i = np.random.randint(x.shape[0])  # TODO cheik more refined choice
        old_xi = x[i]
        neg_grad_i = A[:, i] @ R
        new = ST(x[i] + neg_grad_i / lc[i], lbda / lc[i])
        x[i] = new
        R += A[:, i] * (old_xi - new)
        # TODO if you want to append the losses and iterates


threads = [threading.Thread(target=cd_concu) for _ in range(N_PROC)]

for thread in threads:
    thread.start()
