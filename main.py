import threading
import numpy as np
from numpy.linalg import norm
from multiprocessing import cpu_count

from utils import ST, choose_coord, lasso_loss, save_object

from celer.datasets import make_correlated_data
# from celer.utils import configure_plt
# configure_plt()

n, m, s = 50, 100, 5

max_iter = np.array([60_000])
max_it = max_iter[0]

A, y, x_true = make_correlated_data(n, m, density=0.1, random_state=0)
lbda_max = norm(A.T @ y, ord=np.inf)
lbda = lbda_max / 100

N_PROC = cpu_count()
x = np.zeros(m)
R = y.copy()
lc = norm(A, axis=0) ** 2
all_x = np.zeros((max_it, m))
fx = np.zeros(max_it)
t = np.array([0])
test_var = np.array([0])

# TODO maybe a closure and the threads are started inside
def cd_concu(A, y, lbda):
    global max_iter
    global x
    global R
    global lc
    global all_x
    global t
    global max_it
    global test_var
    while max_iter[0] > 0:
        i = choose_coord(n_features=m)
        old_xi = x[i]
        neg_grad_i = A[:, i] @ R
        new = ST(x[i] + neg_grad_i / lc[i], lbda / lc[i])
        x[i] = new
        R += A[:, i] * (old_xi - new)
        if t >= max_it:
            break
        fx[t[0]] = lasso_loss(A, y, lbda, x)
        all_x[t[0]] = x
        t[0] += 1
        max_iter[0] -= 1
    if test_var[0] ==0:
        test_var[0] += 1
        save_object([A, y, lbda, x, all_x, fx])

threads = [threading.Thread(target=cd_concu, args=(A,y,lbda)) for _ in range(N_PROC)]

for thread in threads:
    thread.start()