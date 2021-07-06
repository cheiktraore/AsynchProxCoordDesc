import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from utils import lasso_loss, load_object

A, y, lbda, x, all_x, fx = load_object('data_save.pickle')
clf = Lasso(alpha=lbda/len(y), fit_intercept=False,
            max_iter=10000, tol=1e-10)
clf.fit(A, y)
f_star = lasso_loss(A, y, lbda, clf.coef_)

plt.figure(constrained_layout=True)
plt.semilogy(1 / A.shape[1] * np.arange(all_x.shape[0]), fx -
                f_star, label=r"Prox_Coord_Desc")
plt.ylabel("Distance to optimal objective")
plt.xlabel("Iteration (scaled for fair comparison)")
plt.legend()
plt.show(block=False)