"""
remarks_pdc.py -- numbers of Remark 6 (Section 7.3): PDC on estimated matrices.
500 data sets (K = 10, T = 10^4, fixed configuration, seed 20260927). For several
values of sigma and stopping rules it prints Pr(A(alpha) PD) with its Wilson interval,
the median lambda_min(hat A(alpha)) and lambda_min(A(alpha)) of the unit output, and
the largest difference between the outputs for sigma = 100 and sigma = 1 (scale
invariance: the initial transformation multiplies all iterates by sigma).
Also, with --typo, the illustration behind Remark 5: the step as printed in [5],
diag(lambda_i + (lambda_i - sigma)_+), versus the corrected diag(max(lambda_i, sigma)).
"""
import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
from model import *
from wilson import wilson


def pdc_v(As, sigma=1.0, stop=0.0, maxit=3000, printed_step=False):
    """PDC (Algorithm 3) stopping when lambda_min > stop*sigma. With printed_step=True
    the step of the pseudocode of [5] (Remark 5) is used instead of the corrected one."""
    Ast = np.stack(As); Am = np.stack([A.ravel() for A in As], 1); G = Am.T@Am
    a = np.array([np.trace(A) for A in As])
    lam = np.linalg.eigvalsh(np.tensordot(a, Ast, 1)); mu = np.abs(lam[lam != 0]).min()
    a = a*(1 if (lam > 0).sum() >= (lam < 0).sum() else -1)*sigma/mu
    for n in range(maxit):
        B = np.tensordot(a, Ast, 1)
        if not np.isfinite(B).all(): return None, -1                     # overflow
        lam, U = np.linalg.eigh(B)
        if lam[0] > stop*sigma: return a, n
        D = lam + np.where(lam > sigma, lam - sigma, 0) if printed_step else np.maximum(lam, sigma)
        a = np.linalg.solve(G, Am.T@(U@np.diag(D)@U.T).ravel())
    return None, maxit


if '--typo' in sys.argv:
    from exact_families import gen
    for p, K in [(5, 5), (20, 20)]:
        rng = np.random.default_rng(1)
        for printed in (True, False):
            ok = 0
            fams = [gen(p, K, np.random.default_rng(1000*t+p)) for t in range(100)]
            for As in fams:
                a, n = pdc_v(As, maxit=5000, printed_step=printed); ok += a is not None
            print(p, K, 'printed step' if printed else 'corrected step', 'terminated in %d/100' % ok)
    sys.exit()

md = make_model(); K = 10; A = pop_matrices(md, K); Ast = np.stack(A)
rng = np.random.default_rng(20260927)
data = [sample_matrices(simulate(md, 10000, rng), K) for _ in range(500)]
D1 = None
for sigma, stop, lab in [(1, 0, 'sigma=1, stop lambda_min>0'), (100, 0, 'sigma=100'), (0.01, 0, 'sigma=0.01'),
                         (1, 0.5, 'stop lambda_min>sigma/2'), (1, 0.999, 'run to lambda_min>0.999 sigma')]:
    pd = []; tl = []; el = []; dirs = []
    for Ah in data:
        a, n = pdc_v(Ah, sigma, stop)
        if a is None: pd.append(False); continue
        u = a/np.linalg.norm(a); dirs.append(u)
        t = np.linalg.eigvalsh(np.tensordot(u, Ast, 1))[0]; e = np.linalg.eigvalsh(np.tensordot(u, np.stack(Ah), 1))[0]
        pd.append(t > 0); tl.append(t); el.append(e)
    ci = wilson(sum(pd), len(pd))
    print(f'{lab:32s} Pr(PD) {ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]  median lmin(hatA) {np.median(el):.2e}  median lmin(A) {np.median(tl):.2e}')
    if sigma == 1 and stop == 0: D1 = np.array(dirs)
    if sigma == 100: print('   max |alpha(sigma=100) - alpha(sigma=1)| = %.1e' % np.abs(np.array(dirs)-D1).max())
