import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pickle
from utils import lasso_loss

with open("data_saved.pickle", "rb") as f:
    A, y, lbda, x, all_x, fx = pickle.load(f) 
clf = Lasso(alpha=lbda/len(y), fit_intercept=False,
            max_iter=20000, tol=1e-10)
clf.fit(A, y)
f_star = lasso_loss(A, y, lbda, clf.coef_)

plt.figure(constrained_layout=True)
plt.semilogy(1 / A.shape[1] * np.arange(all_x.shape[0]), fx -
                f_star, label=r"Prox_Coord_Desc")
plt.ylabel("Distance to optimal objective")
plt.xlabel("Iteration (scaled for fair comparison)")
plt.legend()
plt.show(block=False)