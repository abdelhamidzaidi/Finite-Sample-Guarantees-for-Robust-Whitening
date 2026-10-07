"""
robust_runs.py -- Section 7.5: 100 independently drawn configurations; Tables 6-7 and
Figure 5. Usage:  python robust_runs.py d0 d1     (paper: d = 0..99, run in chunks)
Model seed of draw d: crc32("sens-model|d"); trial seed: crc32("sens-trials|d|T").
Writes sens_{d}.pkl.

Each draw: M with Gaussian entries, V = c_V G G^t at SNR 5 dB, five AR(2) sources with
pole radii U[0.75, 0.9] and frequencies U[0.2, 2.94]; K = 10; T in {2000, 10^4}; 100 trials.
The population margin gamma and direction alpha* = y*/||y*|| (Proposition 2(ii)) are
computed by MNP to tolerance 1e-6. Per trial and method it stores truepd, the split-half
screen with c = 1, 2, the whitening defect, the Amari index, and the direction error
||alpha - alpha*|| used in Table 7 and Figure 5(c).
"""
import numpy as np, time, warnings, pickle, sys, zlib
import core
warnings.filterwarnings('ignore')
from model import *; from sep import *
MAXIT = 3000; P5 = 5; K = 10; SNR = 5.0


def draw_model(d):
    rng = np.random.default_rng(zlib.crc32(f'sens-model|{d}'.encode()))
    ws = np.sort(rng.uniform(0.2, 2.94, P5)); rads = rng.uniform(0.75, 0.9, P5)
    M = rng.standard_normal((P5, P5))
    G = rng.standard_normal((P5, P5)); V = G @ G.T
    V *= np.trace(M @ M.T)/np.trace(V) * 10**(-SNR/10)                  # SNR definition, Section 6.1
    a_list = [ar2_coeffs(r, w) for r, w in zip(rads, ws)]
    H = 400; ac = np.array([acov(a, H) for a in a_list]); scale = 1/np.sqrt(ac[:, 0]); ac = ac*scale[:, None]**2
    return dict(M=M, V=V, a_list=a_list, scale=scale, ac=ac, H=H, ws=ws, rads=rads)


def run_draw(d, Ts=(2000, 10000), NTR=100):
    md = draw_model(d); A = pop_matrices(md, K); Ast = np.stack(A)
    y, lo, up, it, _ = core.mnp(A, maxit=2000000, tol=1e-6)
    gamma = lo if core.LAST_STATUS == 'converged' else np.nan          # population margin (1)
    astar = y/np.linalg.norm(y)                                         # alpha* (Proposition 2(ii))
    Ds = [np.diag(md['ac'][:, k]) for k in range(1, K+1)]; _, gs, *_ = core.mnp(Ds, maxit=200000, tol=1e-6)  # gamma_s
    sv = np.linalg.svd(md['M'], compute_uv=False)
    out = dict(draw=d, gamma=gamma, gamma_s=gs if gs > 0 else np.nan, cond=sv[0]/sv[-1], res={})
    if not (gamma > 0): return out
    for T in Ts:
        rng = np.random.default_rng(zlib.crc32(f'sens-trials|{d}|{T}'.encode()))
        rows = []
        for tr in range(NTR):
            X = simulate(md, T, rng); Ah = sample_matrices(X, K); Ahst = np.stack(Ah)
            Dst = 0.5*(np.stack(sample_matrices(X[:, :T//2], K)) - np.stack(sample_matrices(X[:, T//2:], K)))
            R0 = sym(X @ X.T / T); Q0 = isqrt(R0); Vj = joint_diag([Q0@Ak@Q0 for Ak in Ah])
            row = {'SOBI': amari(Vj.T@Q0@md['M'])}
            ym, lom, upm, itm, first = core.mnp(Ah, maxit=MAXIT, tol=1e-3); st = core.LAST_STATUS
            outs = {'MMRW': ym if lom > 0 else None, 'MNP-first': first[0] if first else None}
            a, n = pc(Ah, maxit=MAXIT); outs['PC'] = a if n < MAXIT else None
            a, n = pdc(Ah, maxit=MAXIT); outs['PDC'] = a if n < MAXIT else None
            row['status'] = st
            for m, a in outs.items():
                if a is None: row[m] = dict(ok=False); continue
                a = a/np.linalg.norm(a); B = np.tensordot(a, Ast, 1); Bh = np.tensordot(a, Ahst, 1)
                tl = np.linalg.eigvalsh(B)[0]; el = np.linalg.eigvalsh(Bh)[0]
                edh = np.linalg.norm(np.tensordot(a, Dst, 1), 2)                       # hat eps(alpha)
                r = dict(ok=el > 0, truepd=tl > 0, tl=tl, pass1=el > edh, pass2=el > 2*edh,
                         derr=np.linalg.norm(a - astar))                               # ||alpha - alpha*||
                if el > 0:
                    Qh = isqrt(Bh); r['defect'] = np.linalg.norm(Qh@B@Qh - np.eye(P5), 2)   # (5)
                    Vj = joint_diag([Qh@Ak@Qh for Ak in Ah]); r['pi'] = amari(Vj.T@Qh@md['M'])
                row[m] = r
            rows.append(row)
        out['res'][T] = rows
    return out


if __name__ == '__main__':
    d0, d1 = int(sys.argv[1]), int(sys.argv[2])
    for d in range(d0, d1):
        o = run_draw(d); pickle.dump(o, open(f'sens_{d}.pkl', 'wb'))
        print('draw', d, 'cond %.1f gamma %.4f' % (o['cond'], o['gamma']), flush=True)
