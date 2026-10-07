"""
gamma_s.py -- source margins of Section 6.1 (fixed configuration, lags 1..K).
gamma_s = max_{||alpha||=1} min_i r_i^t alpha with r_i = (rho_i(1), ..., rho_i(K))
(Section 2). For K >= 5 (feasible), gamma_s > 0 is computed exactly as the margin of the
diagonal matrices D_k by MNP (Proposition 2 applied to D_1..D_K). For K <= 4 there is
no PD combination, gamma_s < 0; it is computed for K = 1 in closed form and for K = 2..4
by maximizing the piecewise-linear function min_i r_i^t alpha over the unit sphere,
a nonconvex problem solved by multistart
(400 000 random starts, then Nelder-Mead refinement of the 20 best), seed 0.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from model import *
from scipy.optimize import minimize

md = make_model(); ac = md['ac']; rng = np.random.default_rng(0)
for K in range(1, 11):
    Rm = ac[:, 1:K+1]                                     # rows r_i^t
    if K == 1:
        print(K, 'gamma_s = %.3f' % max(Rm[:, 0].min(), (-Rm[:, 0]).min())); continue
    Ds = [np.diag(ac[:, k]) for k in range(1, K+1)]
    y, lo, up, it, _ = mnp(Ds, maxit=200000, tol=1e-7)
    if lo > 0:
        print(K, 'gamma_s = %.4f (certified bracket [%.5f, %.5f])' % (lo, lo, up)); continue
    f = lambda a: -np.min(Rm@(a/np.linalg.norm(a)))
    X = rng.standard_normal((400000, K)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = np.min(X@Rm.T, axis=1); best = -np.inf
    for j in np.argsort(v)[-20:]:
        r = minimize(f, X[j], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
        best = max(best, -r.fun)
    print(K, 'gamma_s approx %.3f (numerical, no PD combination)' % best)
