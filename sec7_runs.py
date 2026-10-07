"""
sec7_runs.py -- estimated matrices in the fixed configuration (Sections 7.3-7.4):
Table 5, Figures 3 and 4, and the numbers quoted in Sections 7.3-7.4.
Usage:  python sec7_runs.py TAG VALUES NTRIALS
  TAG = T    : VALUES = 500,1000,2000,5000,10000,20000,50000,100000  (SNR 5 dB, cond 4)
  TAG = snr  : VALUES = -5,0,5,10,20,40                              (T = 5000, cond 4)
  TAG = cond : VALUES = 1,2,4,8,16                                   (T = 5000, SNR 5 dB)
Paper: 500 trials. Seed: crc32("TAG|T|SNR|cond|0"). Writes cmpR_{TAG}_{value}.pkl.

Per trial and method (PC, PDC, MNP-first, MMRW) it stores, for the unit direction alpha:
  truepd   A(alpha) PD for the true matrices;
  tl, el   lambda_min(A(alpha)) and lambda_min(hat A(alpha));
  defect   whitening defect ||hat Q B hat Q - I||_2 with B = A(alpha), hat Q = hat A(alpha)^{-1/2}
           (bounded by eps(alpha)/lambda_hat in (5) when Theorem 3 applies);
  werr     ||hat Q - Q||_2/||Q||_2 of (6), only when A(alpha) is PD;
  pi       Amari index after joint diagonalization of hat Q hat A_k hat Q (Section 7.1);
  cert, cert2   split-half screen passed with c = 1 and c = 2 (Section 5);
  it, time iterations and CPU time of the combination step.
It also stores the Amari index of SOBI with classical whitening hat R_X(0)^{-1/2}.
"""
import numpy as np
import core, time, warnings, pickle, sys, zlib
warnings.filterwarnings('ignore')
from model import *; from sep import *
MAXIT = 3000


def run_setting(tag, T, snr, cond, K=10, NTR=200, seed=0):
    md = make_model(snr_db=snr, cond=cond)
    A = pop_matrices(md, K); Ast = np.stack(A)                          # equation (2)
    y, lo, up, it, _ = mnp(A, maxit=200000, tol=1e-6); gamma = lo       # population margin (1)
    rng = np.random.default_rng(zlib.crc32(f'{tag}|{T}|{snr:g}|{cond:g}|{seed}'.encode()))
    rows = []
    for tr in range(NTR):
        X = simulate(md, T, rng); Ah = sample_matrices(X, K); Ahst = np.stack(Ah)
        H1 = np.stack(sample_matrices(X[:, :T//2], K)); H2 = np.stack(sample_matrices(X[:, T//2:], K))
        Dst = 0.5*(H1-H2)                                               # for hat eps(alpha), Section 5
        row = {}
        # baseline: SOBI with classical whitening hat R_X(0)^{-1/2}, biased by the noise
        R0 = sym(X @ X.T / T); Q0 = isqrt(R0)
        V = joint_diag([Q0@Ak@Q0 for Ak in Ah]); row['SOBI'] = amari(V.T@Q0@md['M'])
        outs = {}
        # MMRW = MNP to tolerance 1e-3 (Algorithms 4-5); MNP-first = its first PD iterate
        t0 = time.perf_counter(); ym, lom, upm, itm, first = mnp(Ah, maxit=MAXIT, tol=1e-3); tm = time.perf_counter()-t0
        outs['MMRW'] = (ym if lom > 0 else None, itm, tm)
        row['mmrw_capped'] = bool(lom > 0 and itm >= MAXIT)
        row['mmrw_status'] = core.LAST_STATUS
        outs['MNP-first'] = (first[0] if first else None, first[1] if first else MAXIT, np.nan)
        t0 = time.perf_counter(); a, n = pc(Ah, maxit=MAXIT); outs['PC'] = (a if n < MAXIT else None, n, time.perf_counter()-t0)
        t0 = time.perf_counter(); a, n = pdc(Ah, maxit=MAXIT); outs['PDC'] = (a if n < MAXIT else None, n, time.perf_counter()-t0)
        for name, (a, n, tt) in outs.items():
            if a is None:                                               # no PD combination found
                row[name] = dict(ok=False, it=n, time=tt); continue
            a = a/np.linalg.norm(a)
            B = np.tensordot(a, Ast, 1); Bh = np.tensordot(a, Ahst, 1)  # A(alpha), hat A(alpha)
            tl = np.linalg.eigvalsh(B)[0]; el = np.linalg.eigvalsh(Bh)[0]
            edh = np.linalg.norm(np.tensordot(a, Dst, 1), 2)            # hat eps(alpha)
            d = dict(ok=True, it=n, time=tt, truepd=tl > 0, tl=tl, el=el, cert=el > edh, cert2=el > 2*edh,
                     werr=np.nan, pi=np.nan, defect=np.nan)
            if el > 0:
                Qh = isqrt(Bh)                                          # hat Q = hat A(alpha)^{-1/2}
                d['defect'] = np.linalg.norm(Qh@B@Qh - np.eye(B.shape[0]), 2)   # left side of (5)
                if tl > 0: d['werr'] = np.linalg.norm(Qh-isqrt(B), 2)/np.linalg.norm(isqrt(B), 2)   # (6)
                V = joint_diag([Qh@Ak@Qh for Ak in Ah]); d['pi'] = amari(V.T@Qh@md['M'])       # W = V^t hat Q
            row[name] = d
        rows.append(row)
    return dict(gamma=gamma, rows=rows, T=T, snr=snr, cond=cond)


if __name__ == '__main__':
    tag = sys.argv[1]; vals = [float(v) for v in sys.argv[2].split(',')]
    NTR = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    for v in vals:
        if tag == 'T':    r = run_setting(tag, int(v), 5.0, 4.0, NTR=NTR)
        if tag == 'snr':  r = run_setting(tag, 5000, v, 4.0, NTR=NTR)
        if tag == 'cond': r = run_setting(tag, 5000, 5.0, v, NTR=NTR)
        pickle.dump(r, open(f'cmpR_{tag}_{v:g}.pkl', 'wb'))
        print(tag, v, 'gamma %.4f' % r['gamma'], 'done', flush=True)
