"""
make_tables.py -- prints every number of Tables 3-7 and the quantities quoted in
Sections 6 and 7 from the result files written by the run scripts:
  res_{K}_{T}.pkl    (sec6_runs.py)       -> Section 6: Theorem 1(a) check, capped runs, Table 3
  exact_{p}_{K}.pkl  (exact_families.py)  -> Table 4
  cmpR_*.pkl         (sec7_runs.py)       -> Table 5 and Sections 7.3-7.4
  sens_{d}.pkl       (robust_runs.py)     -> Tables 6-7 (Section 7.5)
Table 2 is printed by table2_lemma3.py. Proportions come with 95% Wilson intervals
(wilson.py); medians over trials exclude trials without a returned PD combination.
"""
import numpy as np, pickle, glob, warnings
warnings.filterwarnings("ignore")   # e.g. "mean of empty slice" when a method never ran
from collections import Counter
from wilson import wilson
from model import make_model, S_const, P

TS = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
pct = lambda k, n: '%.2f%% [%.2f, %.2f]' % tuple(100*x for x in wilson(k, n)) if n else '--'
ci3 = lambda k, n: '%.3f [%.3f, %.3f]' % wilson(k, n)

# ============================ Section 6 ============================
print('#################### SECTION 6 ####################')
viol = tot = 0
for K in [10, 6]:
    for T in TS:
        d = pickle.load(open(f'res_{K}_{T}.pkl', 'rb')); rho = d['rho']
        for r in d['rows']:
            if r['rho_hat_lo'] > 0:              # estimated problem feasible
                tot += 1
                # Theorem 1(a): |hat gamma - gamma| <= eps <= bar eps; hat gamma lies in the
                # certified bracket [lo, up], so the check is lo - bar eps <= gamma <= up + bar eps
                viol += not (r['rho_hat_lo'] - r['eps'] <= rho <= r['rho_hat_up'] + r['eps'])
print('Theorem 1(a) check: %d violations among %d feasible trials' % (viol, tot))
for K in [10, 6]:
    st = {T: Counter(r['status'] for r in pickle.load(open(f'res_{K}_{T}.pkl', 'rb'))['rows']) for T in TS}
    print('K=%d MNP status per T:' % K, {T: dict(c) for T, c in st.items()})
# Table 3: split-half screen applied to MNP-max; rows (lambda_hat, hat eps, truepd)
for K in [10, 6]:
    for c in [1, 2]:
        acc = fa = n = 0
        for T in TS:
            for r in pickle.load(open(f'res_{K}_{T}.pkl', 'rb'))['rows']:
                x = r['MNP-max']; n += 1                              # x = (truepd, tl, el, werr, ed, edh)
                if x[2] == x[2] and x[5] == x[5] and x[2] > c*x[5]:   # lambda_min(hat A) > c hat eps
                    acc += 1; fa += (not x[0])
        print('Table 3  K=%d c=%d  accepted %d  false %d  false/all %s  false/accepted %s' % (K, c, acc, fa, pct(fa, n), pct(fa, acc)))
for K in [10, 6]:
    for T in [500, 10000, 100000]:
        rows = pickle.load(open(f'res_{K}_{T}.pkl', 'rb'))['rows']
        ok = lambda x: x[2] == x[2] and x[4] == x[4] and x[2] > x[4]  # Theorem 1(b) with the true error
        print('certified test K=%d T=%d: %s' % (K, T, pct(sum(ok(r['MNP-max']) for r in rows), len(rows))))
md = make_model(); SX = S_const(md); gam = pickle.load(open('res_10_500.pkl', 'rb'))['rho']
print('Conservatism: bound (4) with delta=0.1, K=10: T >= %.2e;  median bar eps at T=1e5: %.4f'
      % (10 + 4*(P+1)*10*SX/(0.1*gam**2), np.median([r['eps'] for r in pickle.load(open('res_10_100000.pkl', 'rb'))['rows']])))

# ============================ Table 4 ============================
print('#################### TABLE 4 (exact matrices) ####################')
for (p, K) in [(5, 3), (5, 5), (10, 5), (10, 10), (20, 10), (20, 20)]:
    rows = pickle.load(open(f'exact_{p}_{K}.pkl', 'rb'))
    gu = np.array([r['gu'] for r in rows])                            # bar gamma = ||y_final||
    for m in ['PC', 'PDC', 'MNP']:
        v = np.array([r[m] for r in rows], float)                     # (iterations, time, terminated, lambda_min)
        c = wilson(int(v[:, 2].sum()), len(v))
        print('(%d,%d) %-9s mean it %9.1f  median %6.0f  within cap %3.0f [%.0f, %.0f]  CPU %8.2f ms  lmin/bar gamma %.3f'
              % (p, K, m if m != 'MNP' else 'MNP-first', v[:, 0].mean(), np.median(v[:, 0]), 100*c[0], 100*c[1], 100*c[2],
                 1e3*v[:, 1].mean(), np.nanmedian(v[:, 3]/gu)))
    reached = sum(r['status_max'] == 'converged' for r in rows); c = wilson(reached, len(rows))
    print('(%d,%d) MMRW      mean it %9.0f  reached tol %3.0f [%.0f, %.0f]  CPU %8.0f ms  lmin/bar gamma %.3f  status %s'
          % (p, K, np.mean([r['it_max'] for r in rows]), 100*c[0], 100*c[1], 100*c[2], 1e3*np.mean([r['t_max'] for r in rows]),
             np.nanmedian([r['m_max']/r['gu'] for r in rows]), dict(Counter(r['status_max'] for r in rows))))

# ============================ Table 5 and Sections 7.3-7.4 ============================
print('#################### TABLE 5, SECTIONS 7.3-7.4 ####################')
M4 = ['PC', 'PDC', 'MNP-first', 'MMRW']
def med(rows, m, k, cond=None):
    v = [x[m][k] for x in rows if x[m]['ok'] and (cond is None or cond(x[m]))]
    return (np.nanmedian(v) if v else np.nan), len(v)
for tag, vals in [('T', TS), ('snr', [-5, 0, 5, 10, 20, 40]), ('cond', [1, 2, 4, 8, 16])]:
    for v in vals:
        r = pickle.load(open(f'cmpR_{tag}_{v:g}.pkl', 'rb')); rows = r['rows']; n = len(rows)
        print('%s=%g  gamma=%.4f  SOBI PI %.3f  MMRW status %s' % (tag, v, r['gamma'], np.median([x['SOBI'] for x in rows]),
              dict(Counter(x['mmrw_status'] for x in rows))))
        for m in M4:
            ret = sum(x[m]['ok'] for x in rows); pd = sum(bool(x[m]['ok'] and x[m]['truepd']) for x in rows)
            a1 = sum(bool(x[m]['ok'] and x[m]['cert']) for x in rows); f1 = sum(bool(x[m]['ok'] and x[m]['cert'] and not x[m]['truepd']) for x in rows)
            a2 = sum(bool(x[m]['ok'] and x[m]['cert2']) for x in rows); f2 = sum(bool(x[m]['ok'] and x[m]['cert2'] and not x[m]['truepd']) for x in rows)
            tl, _ = med(rows, m, 'tl'); de, _ = med(rows, m, 'defect'); pi, _ = med(rows, m, 'pi')
            we, nwe = med(rows, m, 'werr', lambda d: d['truepd'])
            cpu = 1e3*np.nanmean([x[m]['time'] for x in rows])
            print('   %-9s returned %d/%d  Pr(PD) %s  lmin %.4f  defect %.2f  PI %.3f  werr|PD %.3f (n=%d)  screen c=1 %s (false %d)  c=2 %d (false %d)  CPU %.2f ms'
                  % (m, ret, n, ci3(pd, n), tl, de, pi, we, nwe, pct(a1, n), f1, a2, f2, cpu))

# ============================ Tables 6-7 ============================
print('#################### TABLES 6-7 (Section 7.5) ####################')
D = [pickle.load(open(f'sens_{d}.pkl', 'rb')) for d in range(100)]
feas = [d for d in D if d['gamma'] > 0]
print('draws %d, with a PD combination %d, cond %.1f-%.1f, gamma %.4f-%.3f'
      % (len(D), len(feas), min(d['cond'] for d in D), max(d['cond'] for d in D), min(d['gamma'] for d in feas), max(d['gamma'] for d in feas)))
rng = np.random.default_rng(2026)
def boot_median(x, B=20000):
    """95% percentile-bootstrap interval of the median over configurations."""
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x))); return np.quantile(np.median(x[idx], axis=1), [.025, .975])
for T in [2000, 10000]:
    print('=== T = %d' % T)
    per = {m: {} for m in M4}
    for m in M4:        # per-configuration summaries
        per[m]['pd'] = np.array([np.mean([r[m]['ok'] and r[m]['truepd'] for r in d['res'][T]]) for d in feas])
        for k in ['pi', 'defect', 'derr']:
            per[m][k] = np.array([np.median([r[m][k] for r in d['res'][T] if r[m]['ok']]) for d in feas])
    for m in M4:        # Table 6
        n = sum(len(d['res'][T]) for d in feas)
        a1 = sum(r[m]['ok'] and r[m]['pass1'] for d in feas for r in d['res'][T])
        f1 = sum(r[m]['ok'] and r[m]['pass1'] and not r[m]['truepd'] for d in feas for r in d['res'][T])
        best = sum(per[m]['pd'][i] >= max(per[k]['pd'][i] for k in M4) - 1e-12 for i in range(len(feas)))
        strict = sum(per[m]['pd'][i] > max(per[k]['pd'][i] for k in M4 if k != m) + 1e-12 for i in range(len(feas)))
        print('Table 6  %-9s Pr(PD) %.2f [%.2f, %.2f]  PI %.3f  defect %.2f  best %d / %d  screen %d (%.1f%%)  false/accepted %d/%d %s'
              % (m, np.median(per[m]['pd']), np.quantile(per[m]['pd'], .25), np.quantile(per[m]['pd'], .75),
                 np.median(per[m]['pi']), np.median(per[m]['defect']), best, strict, a1, 100*a1/n, f1, a1, pct(f1, a1)))
    g = np.array([d['gamma'] for d in feas])
    dpd = per['PC']['pd'] - per['MMRW']['pd']          # Delta_PD, positive favours PC
    dpi = per['MMRW']['pi'] - per['PC']['pi']          # Delta_PI
    ddir = per['MMRW']['derr'] - per['PC']['derr']     # Delta_dir
    for lab, i in [('all', np.ones(len(g), bool)), ('gamma<0.01', g < 0.01), ('0.01<=gamma<0.05', (g >= 0.01) & (g < 0.05)), ('gamma>=0.05', g >= 0.05)]:
        x = dpd[i]
        print('Table 7  %-18s n=%3d  dPD %+.3f IQR [%+.3f, %+.3f] CI %s  +/-/0 %d/%d/%d  dPI %+.4f CI %s  ddir %+.2f CI %s'
              % (lab, i.sum(), np.median(x), np.quantile(x, .25), np.quantile(x, .75), np.round(boot_median(x), 3),
                 (x > 0).sum(), (x < 0).sum(), (x == 0).sum(), np.median(dpi[i]), np.round(boot_median(dpi[i]), 4),
                 np.median(ddir[i]), np.round(boot_median(ddir[i]), 2)))

# ------------- Section 7.5: checks of the exploratory margin strata -------------
# The cut-offs 0.01 and 0.05 were chosen while analysing the first 40 configurations
# (d = 0..39). Two checks: (i) the strata on the 60 configurations drawn afterwards
# (d = 40..99); (ii) a bin-free Spearman correlation between gamma and Delta_PD,
# with a 95% bootstrap interval over configurations (5000 resamples).
from scipy.stats import spearmanr
rng2 = np.random.default_rng(2027)
def boot2(x, B=20000):
    x = np.asarray(x); idx = rng2.integers(0, len(x), (B, len(x))); return np.quantile(np.median(x[idx], 1), [.025, .975])
for T in [2000, 10000]:
    g = np.array([d['gamma'] for d in D])
    pdm = lambda m: np.array([np.mean([r[m]['ok'] and r[m]['truepd'] for r in d['res'][T]]) for d in D])
    dpd = pdm('PC') - pdm('MMRW')
    rho = spearmanr(g, dpd).correlation
    bs = []
    for b in range(5000):
        i = rng2.integers(0, len(D), len(D)); bs.append(spearmanr(g[i], dpd[i]).correlation)
    print('T=%d  Spearman(gamma, Delta_PD) = %.2f  CI %s' % (T, rho, np.round(np.quantile(bs, [.025, .975]), 2)))
    hold = np.arange(len(D)) >= 40
    for lab, sel in [('gamma<0.01', g < 0.01), ('0.01<=gamma<0.05', (g >= 0.01) & (g < 0.05)), ('gamma>=0.05', g >= 0.05)]:
        x = dpd[sel & hold]
        print('   configurations 40-99, %-17s n=%2d  Delta_PD median %+.3f  CI %s  +/-/0 %d/%d/%d'
              % (lab, len(x), np.median(x), np.round(boot2(x), 3), (x > 0).sum(), (x < 0).sum(), (x == 0).sum()))
