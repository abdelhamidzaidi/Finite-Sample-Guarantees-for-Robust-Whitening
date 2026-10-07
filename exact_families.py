"""
exact_families.py -- Section 7.2 (exact matrices): Table 4 and Figure 2.
Usage:  python exact_families.py p K NTRIALS
Paper: (p,K) in {(5,3),(5,5),(10,5),(10,10)} with 100 trials, (20,10) with 60 and
(20,20) with 50. Seed: 1000 p + K. Writes exact_{p}_{K}.pkl.

The random families follow the appendix of [5]: A_1..A_{K-1} = R_k + R_k^t with
uniform entries, and A_K = U Delta U^t chosen so that B* = A_K + sum alpha_k A_k is PD;
hence gamma > 0. The unspecified Delta of [5] is taken as -D + U(0.1, 1).

Stored per trial:
  gl, gu   certified bounds gamma_- <= gamma <= bar gamma from MNP (Proposition 3(i));
  Rp       R_+ = (sum_k ||A_k||_2^2)^{1/2} >= R   (so kappa = R_+/gamma_- bounds R/gamma);
  it_max, t_max, status_max, m_max   iterations, time, status, and lambda_min(A(alpha))
           of the MMRW output (MNP to tolerance 1e-3, cap 20 000);
  MNP, PC, PDC   (iterations to first PD iterate, CPU time, terminated?, lambda_min(A(alpha))).
Figure 2 plots the iterations against kappa^2 with the bounds 3 kappa^2 (PC,
Proposition 4(i)) and 2 kappa^2 (MNP, Proposition 3(ii)).
"""
import numpy as np
import core, time, warnings, pickle, sys
warnings.filterwarnings('ignore')
from core import *


def gen(p, K, rng):
    """Random family of [5], appendix 'Generation of the set A'."""
    a = rng.standard_normal(K-1)
    As = []
    for k in range(K-1):
        Rk = rng.uniform(-1, 1, (p, p)); As.append(Rk + Rk.T)
    M = np.tensordot(a, np.stack(As), 1)
    D, U = np.linalg.eigh(M)
    while True:                                   # Delta + D PD and Delta sign-indefinite
        Delta = -D + rng.uniform(0.1, 1.0, p)
        if (Delta > 0).any() and (Delta < 0).any(): break
    As.append(U @ np.diag(Delta) @ U.T)
    return As


if __name__ == '__main__':
    CAP = 20000
    p, K, NTR = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(1000*p + K)
    rows = []
    for tr in range(NTR):
        As = gen(p, K, rng); Ast = np.stack(As)
        Rp = np.sqrt(sum(np.linalg.norm(A, 2)**2 for A in As))          # R_+ >= R
        t0 = time.perf_counter(); ym, lo, up, itm, first = mnp(As, maxit=CAP, tol=1e-3); tmax = time.perf_counter()-t0
        row = dict(gl=lo, gu=up, Rp=Rp, it_max=itm, t_max=tmax)
        row['status_max'] = core.LAST_STATUS
        def marg(a):                                                      # lambda_min(A(alpha)), ||alpha|| = 1
            a = a/np.linalg.norm(a); return np.linalg.eigvalsh(np.tensordot(a, Ast, 1))[0]
        row['m_max'] = marg(ym) if lo > 0 else np.nan
        t0 = time.perf_counter(); r = mnp(As, maxit=CAP, stop_first=True); tt = time.perf_counter()-t0
        row['MNP'] = (r[3], tt, r[4] is not None, marg(r[0]) if r[4] is not None else np.nan)
        t0 = time.perf_counter(); a, n = pc(As, maxit=CAP); tt = time.perf_counter()-t0
        row['PC'] = (n, tt, n < CAP, marg(a) if n < CAP else np.nan)
        t0 = time.perf_counter(); a, n = pdc(As, maxit=CAP); tt = time.perf_counter()-t0
        row['PDC'] = (n, tt, n < CAP, marg(a) if n < CAP else np.nan)
        rows.append(row)
    pickle.dump(rows, open(f'exact_{p}_{K}.pkl', 'wb'))
    print(p, K, 'done', flush=True)
