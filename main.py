from asyn_prox_coord_desc import PRBCFB, lasso_loss, np, norm, shared_memory
from multiprocessing import cpu_count
import concurrent
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
# import threading

# have more readable plots with increased fontsize:
fontsize = 16
plt.rcParams.update({'axes.labelsize': fontsize,
                     'font.size': fontsize,
                     'legend.fontsize': fontsize,
                     'xtick.labelsize': fontsize - 2,
                     'ytick.labelsize': fontsize - 2})

n, m, s = 50, 100, 5
# max_iter = np.zeros(1)
# max_iter[0] = 50000
max_it = 50000 #int(max_iter[0])
x_true = np.zeros(m)
support = np.random.choice(m, size=s, replace=False)
x_true[support] = (-1) ** np.arange(s)   # alternate 1 and -1
A = np.random.rand(n, m)
y = A @ x_true + 0.06 * np.random.randn(n)

lbda_max = norm(A.T @ y, ord=np.inf)

lbda = lbda_max / 100

x0 = np.zeros(m)

if __name__ == '__main__':
    max_iter = shared_memory.ShareableList([max_it])
    PROC_NUM = cpu_count()
    executors = concurrent.futures.ProcessPoolExecutor(PROC_NUM)
    list_of_processes = []
    results = list(np.zeros(PROC_NUM))

    for i in range(PROC_NUM):
        list_of_processes.append(executors.submit(
            PRBCFB, A, y, x0, lbda, max_iter, max_it, i))
    for i in range(PROC_NUM):
        results[i] = list_of_processes[i].result()

    clf = Lasso(alpha=lbda/len(y), fit_intercept=False, max_iter=10000, tol=1e-10)
    clf.fit(A, y)
    f_star = lasso_loss(A, y, lbda, clf.coef_)

    plt.figure(constrained_layout=True)
    x, all_x, fx = results[0]
    print("Last 15 function values: {} \n Optimal value: {}".format(fx[-15:-1], f_star))
    print(max_it, max_iter[0])
    print(x0)
    plt.semilogy(1 / m * np.arange(max_it), fx -
                f_star, label=r"PRBCFB")
    plt.ylabel("Distance to optimal objective")
    plt.xlabel("Iteration (scaled for fair comparison)")
    plt.legend()
    plt.show(block=False)
    max_iter.shm.close()
    max_iter.shm.unlink()
