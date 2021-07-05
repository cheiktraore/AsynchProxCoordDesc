clf = Lasso(alpha=lbda/len(y), fit_intercept=False,
            max_iter=10000, tol=1e-10)
clf.fit(A, y)
f_star = lasso_loss(A, y, lbda, clf.coef_)

plt.figure(constrained_layout=True)
x, all_x, fx = results[0]
print("Last 15 function values: {} \n Optimal value: {}".format(
    fx[-15:-1], f_star))
print(max_it, max_iter[0])
print(x0)
plt.semilogy(1 / m * np.arange(max_it), fx -
                f_star, label=r"PRBCFB")
plt.ylabel("Distance to optimal objective")
plt.xlabel("Iteration (scaled for fair comparison)")
plt.legend()
plt.show(block=False)