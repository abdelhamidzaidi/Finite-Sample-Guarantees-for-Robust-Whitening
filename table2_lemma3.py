"""
table2_lemma3.py -- Table 2 (Section 6.2): Monte Carlo check of Lemma 3,
    E || hat Gamma_X(tau) - Gamma_X(tau) ||_F^2 <= (p+1) S_X / (T - tau).
For K = 10 lags and T in {1000, 5000, 20000}, 300 replications each, it prints
max_k E||E_k||_F^2 (estimated), the bound min_k (p+1) S_X/(T - tau_k), their ratio,
and T * max_k E||E_k||_F^2 (constant if the rate is 1/T). Seeds: 7 + T.
Output: printed rows of Table 2; S_X = 6.21 is printed first.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from model import *

md = make_model(); S = S_const(md); K = 10        # S_X of Assumption 2
print('S_X = %.2f' % S)
A = pop_matrices(md, K)                           # population A_k, equation (2)
for T in [1000, 5000, 20000]:
    rng = np.random.default_rng(7+T)
    acc = np.zeros(K); R = 300
    for _ in range(R):
        Ah = sample_matrices(simulate(md, T, rng), K)             # hat A_k (Section 4)
        acc += np.array([np.linalg.norm(a-b, 'fro')**2 for a, b in zip(Ah, A)])   # ||E_k||_F^2
    emp = acc/R                                                   # Monte Carlo E||E_k||_F^2
    # Lemma 3 bound for lag tau_k (symmetrization does not increase the Frobenius norm);
    # the common bound max_k (p+1) S_X/(T - tau_k) = (p+1) S_X/(T - tau_K) bounds every k,
    # and the ratio column is max_k of E||E_k||_F^2 divided by its own bound (Table 2 caption)
    bnd = np.array([(P+1)*S/(T-t) for t in range(1, K+1)])
    print('T=%6d  max_k E||E_k||_F^2 = %.3e  common bound = %.3e  max_k ratio = %.2f  T*E = %.2f'
          % (T, emp.max(), bnd.max(), (emp/bnd).max(), emp.max()*T))
