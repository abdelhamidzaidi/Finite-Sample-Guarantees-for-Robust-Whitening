"""
sec6_runs.py -- simulations of Section 6 (validation of the theory).
Usage:  python sec6_runs.py K T1,T2,... NTRIALS      (paper: K in {10, 6}, 200 trials)
Seed of the trials at (K, T): 1000 K + T. Writes res_{K}_{T}.pkl, read by
make_tables.py (Theorem 1(a) check, Table 3, capped-run counts) and by
make_figures.py (Figure 1).

For each trial it stores, for the unit direction alpha returned by each method,
  (A(alpha) PD?, lambda_min(A(alpha)), lambda_min(hat A(alpha)), relative error of Q,
   eps(alpha) = ||hat A(alpha) - A(alpha)||_2   [Section 4, true directional error],
   hat eps(alpha)                               [Section 5, split-half estimate]).
The test with the true error is Theorem 1(b): lambda_min(hat A(alpha)) > eps(alpha).
"""
import numpy as np
import core, time, warnings, pickle, sys
warnings.filterwarnings('ignore')
from model import *

md = make_model()
K = int(sys.argv[1]); Ts = [int(x) for x in sys.argv[2].split(',')]; NTR = int(sys.argv[3])
MAXIT = 3000                                   # iteration cap of every method
A = pop_matrices(md, K); Ast = np.stack(A)     # population A_k, equation (2)
# population margin gamma (1), certified to relative tolerance 1e-6 (Proposition 3(i))
y, lo, up, it, _ = mnp(A, maxit=200000, tol=1e-6); rho = lo
for T in Ts:
    t0 = time.time()
    rng = np.random.default_rng(1000*K + T)
    rows = []
    for tr in range(NTR):
        X = simulate(md, T, rng); Ah = sample_matrices(X, K); Ahst = np.stack(Ah)
        # split-half estimates hat A_k^(1), hat A_k^(2) and their half difference (Section 5)
        H1 = np.stack(sample_matrices(X[:, :T//2], K)); H2 = np.stack(sample_matrices(X[:, T//2:], K))
        Dst = 0.5*(H1 - H2)
        # bar eps = (sum_k ||E_k||_2^2)^{1/2} >= eps (Section 4), used in the Theorem 1(a) check
        eps = np.sqrt(sum(np.linalg.norm(a-b, 2)**2 for a, b in zip(Ah, A)))
        row = dict(eps=eps)
        # MNP to tolerance eta = 1e-3 (MNP-max) on the estimated matrices (Algorithm 1)
        ym, lom, upm, itm, first = mnp(Ah, maxit=MAXIT, tol=1e-3)
        row['feas_hat'] = first is not None             # estimated problem has a PD combination
        row['rho_hat_lo'] = lom; row['rho_hat_up'] = upm; row['it'] = itm   # certified bounds on hat gamma
        row['status'] = core.LAST_STATUS
        outs = {}
        outs['MNP-max'] = ym if lom > 0 else None
        outs['MNP-first'] = first[0] if first is not None else None
        if first is not None:
            a, n = pc(Ah, maxit=MAXIT);  outs['PC'] = a if n < MAXIT else None
            a, n = pdc(Ah, maxit=MAXIT); outs['PDC'] = a if n < MAXIT else None
        else:
            outs['PC'] = outs['PDC'] = None
        for name, a in outs.items():
            if a is None:
                row[name] = (False, np.nan, np.nan, np.nan, np.nan, np.nan); continue
            a = a/np.linalg.norm(a)                               # unit direction
            B = np.tensordot(a, Ast, 1); Bh = np.tensordot(a, Ahst, 1)   # A(alpha), hat A(alpha)
            tl = np.linalg.eigvalsh(B)[0]; el = np.linalg.eigvalsh(Bh)[0]
            werr = np.nan
            if tl > 0 and el > 0:                                 # ||hat Q - Q|| / ||Q||, (6)
                werr = np.linalg.norm(isqrt(Bh)-isqrt(B), 2)/np.linalg.norm(isqrt(B), 2)
            ed = np.linalg.norm(Bh - B, 2)                        # eps(alpha), true error
            edh = np.linalg.norm(np.tensordot(a, Dst, 1), 2)      # hat eps(alpha), split-half
            row[name] = (tl > 0, tl, el, werr, ed, edh)
        rows.append(row)
    pickle.dump(dict(rho=rho, rows=rows), open(f'res_{K}_{T}.pkl', 'wb'))
    print(K, T, 'gamma %.5f' % rho, 'done in %.1fs' % (time.time()-t0), flush=True)
