"""
make_figures.py -- Figures 1-5 of the manuscript, from the result files of the run scripts.

  Figure 1 (Section 6.3)  fig_cert.pdf        res_{K}_{T}.pkl   (sec6_runs.py)
  Figure 2 (Section 7.2)  fig_exact_bounds.pdf exact_{p}_{K}.pkl (exact_families.py)
  Figure 3 (Section 7.3)  fig_cmp_T.pdf       cmpR_T_*.pkl      (sec7_runs.py T ...)
  Figure 4 (Section 7.4)  fig_cmp_snr_cond.pdf cmpR_snr_*, cmpR_cond_* (sec7_runs.py)
  Figure 5 (Section 7.5)  fig_sens.pdf        sens_{d}.pkl      (robust_runs.py)

Usage:  python make_figures.py [RESULTS_DIR]   (default: the current folder)
The figures are written to the current folder.

Plotting conventions (identical in all figures): each method has its own colour, used in every
figure and for no other purpose: MMRW magenta, MNP-first blue, PC orange, PDC green, SOBI grey.
Other curves: in Figure 1, which shows only MMRW, the acceptance tests are black, red and blue;
red and black for the two sample sizes of Figure 5. Data points are small dots of the same colour as their line.
Probabilities are shown with 95% Wilson intervals (wilson.py); points of different
methods are shifted slightly along the x axis so that the error bars do not overlap.
Text and mathematics use the same serif typeface (STIX, shipped with Matplotlib), at
10 pt for figures that are printed about 25% smaller than their design width.
"""
import os, sys, numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator
from wilson import wilson
RES = sys.argv[1] if len(sys.argv) > 1 else '.'               # folder with the .pkl result files
pk = lambda name: pickle.load(open(os.path.join(RES, name), 'rb'))
plt.rcParams.update({'font.size': 10, 'font.family': 'STIXGeneral', 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'legend.frameon': False, 'lines.linewidth': 1.1,
    'lines.markersize': 3.0, 'axes.titlesize': 10, 'axes.titleweight': 'normal', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
COL = {'MMRW': '#D400D4', 'MNP-first': '#0072B2', 'PC': '#FF8C00', 'PDC': '#1B7F3B', 'SOBI': '#8C8C8C'}
MK  = {'MMRW': 'o', 'MNP-first': 'o', 'PC': 'o', 'PDC': 'o', 'SOBI': 'o'}   # small dots, coloured like their line
LS  = {'MMRW': '-', 'MNP-first': '--', 'PC': '-.', 'PDC': ':', 'SOBI': '-'}
LAB = {'MMRW': 'MMRW (proposed)', 'MNP-first': 'MNP-first', 'PC': 'PC', 'PDC': 'PDC', 'SOBI': 'SOBI, classical whitening'}
M4 = ['MMRW', 'MNP-first', 'PC', 'PDC']
def grid(ax): ax.grid(True, axis='y', lw=0.3, alpha=0.5); ax.grid(True, axis='x', which='major', lw=0.3, alpha=0.3)
def line(ax, x, y, m, **kw):
    ax.plot(x, y, color=COL[m], ls=LS[m], marker=MK[m], mec=COL[m], mfc=COL[m], mew=0, label=LAB[m], **kw)
def ebar(ax, x, cis, m, shift=1.0, logx=True):
    """Proportions with Wilson intervals; cis = list of (p_hat, lower, upper)."""
    x = np.array(x, float); x = x*shift if logx else x+shift
    p = np.array([c[0] for c in cis]); lo = np.clip(p-np.array([c[1] for c in cis]), 0, None); hi = np.clip(np.array([c[2] for c in cis])-p, 0, None)
    ax.errorbar(x, p, yerr=[lo, hi], color=COL[m], ls=LS[m], marker=MK[m], mec=COL[m], mfc=COL[m], mew=0, elinewidth=0.6, capsize=0, label=LAB[m])
def panel(ax, s): ax.set_title(s, loc='left')
def declog(ax, ticks):
    """Log-scaled y axis labelled with plain decimals (0.02, 0.05, 0.1, ...)."""
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ('%g' % v)))
def pdci(rows, m):
    """Pr(A(alpha) PD for the true matrices) over all trials, with its Wilson interval;
    a trial without a returned combination counts as not PD."""
    return wilson(sum(bool(x[m]['ok'] and x[m]['truepd']) for x in rows), len(rows))
def stat(rows, m, k):
    """Median over the trials with a returned combination (Section 7.1, trial sets)."""
    return np.nanmedian([x[m].get(k, np.nan) if x[m]['ok'] else np.nan for x in rows])
load = lambda tag, v: pk(f'cmpR_{tag}_{v:g}.pkl')
shifts = {'MMRW': 1/1.06, 'MNP-first': 1/1.02, 'PC': 1.02, 'PDC': 1.06}
# ---------------- Figure 3: effect of the sample size (Section 7.3) ----------------
# (a) Pr(A(alpha) PD); (b) median lambda_min(A(alpha)) of the unit output, with the
# population margin gamma of (1) as a dashed line; (c) median whitening defect
# ||hat Q B hat Q - I||_2 (left side of (5)); (d) median Amari index (Section 7.1).
Ts = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
R = {T: load('T', T) for T in Ts}; gam = R[Ts[0]]['gamma']
fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.9)); ax = ax.ravel()
for m in M4:
    ebar(ax[0], Ts, [pdci(R[T]['rows'], m) for T in Ts], m, shifts[m])
    line(ax[1], Ts, [stat(R[T]['rows'], m, 'tl') for T in Ts], m)
    line(ax[2], Ts, [stat(R[T]['rows'], m, 'defect') for T in Ts], m)
    line(ax[3], Ts, [stat(R[T]['rows'], m, 'pi') for T in Ts], m)
line(ax[3], Ts, [np.median([x['SOBI'] for x in R[T]['rows']]) for T in Ts], 'SOBI')
ax[1].axhline(gam, color='0.5', lw=0.7, ls=(0, (4, 3))); ax[1].annotate('$\\gamma$', (7.5e4, gam), xytext=(0, 3), textcoords='offset points', color='0.4')
for a in ax: a.set_xscale('log'); a.set_xlabel('sample size $T$'); grid(a)
ax[2].set_yscale('log'); ax[3].set_yscale('log')
declog(ax[2], [0.1, 0.2, 0.5, 1, 2, 5, 10]); declog(ax[3], [0.005, 0.01, 0.02, 0.05, 0.1])
panel(ax[0], '(a) probability that $A(\\hat\\alpha)$ is PD'); panel(ax[1], '(b) median $\\lambda_{\\min}(A(\\hat\\alpha))$, $\\|\\hat\\alpha\\|=1$')
panel(ax[2], '(c) median whitening defect $\\|\\hat Q B\\hat Q-I\\|_2$'); panel(ax[3], '(d) median Amari index')
ax[0].set_ylim(0, 1.03)
h, l = ax[3].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('fig_cmp_T.pdf'); fig.savefig('fig_cmp_T.png', dpi=80)
# ---------------- Figure 4: effect of the SNR and of cond(M) (Section 7.4) ----------------
# gamma does not depend on the noise (Proposition 1) but decreases with cond(M).
S = [-5, 0, 5, 10, 20, 40]; C = [1, 2, 4, 8, 16]
RS = {v: load('snr', v) for v in S}; RC = {v: load('cond', v) for v in C}
sh_lin = {'MMRW': -0.9, 'MNP-first': -0.3, 'PC': 0.3, 'PDC': 0.9}
fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.9)); ax = ax.ravel()
for m in M4:
    ebar(ax[0], S, [pdci(RS[v]['rows'], m) for v in S], m, sh_lin[m], logx=False)
    line(ax[1], S, [stat(RS[v]['rows'], m, 'pi') for v in S], m)
    ebar(ax[2], C, [pdci(RC[v]['rows'], m) for v in C], m, shifts[m])
    line(ax[3], C, [stat(RC[v]['rows'], m, 'pi') for v in C], m)
line(ax[1], S, [np.median([x['SOBI'] for x in RS[v]['rows']]) for v in S], 'SOBI')
line(ax[3], C, [np.median([x['SOBI'] for x in RC[v]['rows']]) for v in C], 'SOBI')
for a in ax[:2]: a.set_xlabel('SNR (dB)')
for a in ax[2:]: a.set_xscale('log'); a.set_xticks(C); a.set_xticklabels(C); a.minorticks_off(); a.set_xlabel('cond$(M)$')
ax[1].set_yscale('log'); ax[3].set_yscale('log')
declog(ax[1], [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]); declog(ax[3], [0.02, 0.03, 0.05, 0.1, 0.15])
for a in (ax[0], ax[2]): a.set_ylim(0, 1.03)
for a in ax: grid(a)
panel(ax[0], '(a) probability that $A(\\hat\\alpha)$ is PD'); panel(ax[1], '(b) median Amari index')
panel(ax[2], '(c) probability that $A(\\hat\\alpha)$ is PD'); panel(ax[3], '(d) median Amari index')
h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('fig_cmp_snr_cond.pdf'); fig.savefig('fig_cmp_snr_cond.png', dpi=80)
# ---------------- Figure 2: exact matrices (Section 7.2) ----------------
# Iterations to the first PD iterate against kappa^2 = (R_+/gamma_-)^2, where
# R_+ = (sum_k ||A_k||_2^2)^{1/2} >= R and gamma_- <= gamma (certified): the lines
# 3 kappa^2 and 2 kappa^2 are valid upper bounds for PC (Proposition A1(i)) and
# MNP-first (Proposition 3(ii)).
# Runs that did not terminate within the cap are censored: their iteration count is only
# known to exceed the cap, so they are drawn as open triangles on the cap line.
# (No run terminated at iteration 0, so every terminated run is visible on the log axis.)
CAP = 20000
fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.5))
for m, key in [('PC', 'PC'), ('PDC', 'PDC'), ('MNP-first', 'MNP')]:
    X = []; Y = []; Xc = []
    for (p, K) in [(5, 3), (5, 5), (10, 5), (10, 10), (20, 10), (20, 20)]:
        rows = pk(f'exact_{p}_{K}.pkl')
        for r in rows:
            k2 = (r['Rp']/r['gl'])**2
            if r[key][2]: X.append(k2); Y.append(max(r[key][0], 0.8))
            else: Xc.append(k2)
    ax.scatter(X, Y, s=3, marker='o', color=COL[m], linewidths=0, alpha=0.55, label=LAB[m])
    if Xc: ax.scatter(Xc, [CAP]*len(Xc), s=14, marker='^', facecolors='none', edgecolors=COL[m], linewidths=0.6,
                      label=f'{LAB[m]}, not terminated ({len(Xc)})')
xs = np.logspace(0.5, 10.5, 50)
ax.plot(xs, 3*xs, color='k', lw=0.8, label='bound $3\\kappa^2$ (PC)'); ax.plot(xs, 2*xs, color='k', lw=0.8, ls='--', label='bound $2\\kappa^2$ (MNP-first)')
ax.axhline(CAP, color='0.5', lw=0.6, ls=':'); ax.text(8e10, 13000, 'iteration cap', color='0.4', fontsize=8.5, ha='right', va='top')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.6, 3e7)
ax.set_xlabel('$\\kappa^2=(R_+/\\gamma_-)^2$'); ax.set_ylabel('iterations to the first PD iterate'); grid(ax)
ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(1.01, 1.0), markerscale=2, handlelength=1.8, borderaxespad=0)
fig.tight_layout(); fig.savefig('fig_exact_bounds.pdf'); fig.savefig('fig_exact_bounds.png', dpi=80)
# ---------------- Figure 1: acceptance tests (Section 6.3, 200 trials per point) ----------------
# 'test with true error' = Theorem 1(b): lambda_min(hat A(alpha)) > eps(alpha);
# 'split-half screen'    = Section 5: lambda_min(hat A(alpha)) > c hat eps(alpha);
# 'false acceptance'     = accepted by the screen (c = 1) although A(alpha) is not PD.
# Row layout of res_*.pkl: x = (truepd, lambda_min(A), lambda_min(hat A), werr, eps, hat eps).
Te = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.0))
spec = [('oracle', 'test with true error $\\varepsilon(\\hat\\alpha)$', '#000000', '-', 'o', 1/1.05), ('c1', 'split-half screen, $c=1$', '#E41A1C', '--', 'o', 1.0),
        ('c2', 'split-half screen, $c=2$', '#1F4FFF', '-.', 'o', 1.05), ('fa', 'false acceptance (screen, $c=1$)', '#E41A1C', ':', 'o', 1.0)]
for j, K in enumerate([10, 6]):
    vals = {k: [] for k, *_ in spec}
    for T in Te:
        rows = pk(f'res_{K}_{T}.pkl')['rows']; x = np.array([r['MNP-max'] for r in rows], float); n = len(x)
        el = np.nan_to_num(x[:, 2], nan=-1); ed = np.nan_to_num(x[:, 4], nan=np.inf); edh = np.nan_to_num(x[:, 5], nan=np.inf); bad = x[:, 0] < 0.5
        for k, acc in [('oracle', el > ed), ('c1', el > edh), ('c2', el > 2*edh), ('fa', (el > edh) & bad)]:
            vals[k].append(wilson(int(acc.sum()), n))
    for k, lab, c, ls, mk, sh in spec:
        p = np.array([v[0] for v in vals[k]]); lo = np.clip(p-np.array([v[1] for v in vals[k]]), 0, None); hi = np.clip(np.array([v[2] for v in vals[k]])-p, 0, None)
        ax[j].errorbar(np.array(Te)*sh, p, yerr=[lo, hi], color=c, ls=ls, marker=mk, ms=2.8, mfc=c, mec=c, mew=0, elinewidth=0.6, capsize=0, label=lab)
    ax[j].set_xscale('log'); ax[j].set_xlabel('sample size $T$'); grid(ax[j])
    # panel (a) on [0, 1]; panel (b) on its own, much smaller scale (all rates are at most a few %)
    if K == 10: ax[j].set_ylim(-0.02, 1.03)
    else: ax[j].set_ylim(-0.002, 0.05); ax[j].set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    panel(ax[j], f'({"ab"[j]}) $K={K}$, $\\gamma={[0.0185, 0.0025][j]}$')
ax[0].set_ylabel('fraction of all trials')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
fig.tight_layout(rect=(0, 0, 1, 0.82)); fig.savefig('fig_cert.pdf'); fig.savefig('fig_cert.png', dpi=80)
# ---------------- Figure 5: paired comparison of PC and MMRW (Section 7.5) ----------------
# Per configuration: (a) Delta_PD = Pr(PD) of PC minus that of MMRW; (b) Delta_PI =
# median Amari index of MMRW minus that of PC; (c) Delta_dir = median ||alpha - alpha*||
# of MMRW minus that of PC, with alpha* the population maximal-margin direction
# (Proposition 2(ii)). Positive values favour PC. Horizontal segments: medians in the
# strata gamma < 0.01, 0.01 <= gamma < 0.05, gamma >= 0.05 (Table 7).
D = [pk(f'sens_{d}.pkl') for d in range(100)]
g = np.array([d['gamma'] for d in D])
def per(T, m, k):
    if k == 'pd': return np.array([np.mean([r[m]['ok'] and r[m]['truepd'] for r in d['res'][T]]) for d in D])
    return np.array([np.median([r[m][k] for r in d['res'][T] if r[m]['ok']]) for d in D])
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10})   # larger labels for Figure 5
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.3))
spec5 = [(2000, '#E41A1C', 'o', '$T=2000$'), (10000, '#000000', 'o', '$T=10^4$')]   # colours not used for any method
strata = [(1e-4, 0.01), (0.01, 0.05), (0.05, 0.6)]
for T, c, mk, lab in spec5:
    dd = [per(T, 'PC', 'pd') - per(T, 'MMRW', 'pd'), per(T, 'MMRW', 'pi') - per(T, 'PC', 'pi'), per(T, 'MMRW', 'derr') - per(T, 'PC', 'derr')]
    for a, y in zip(ax, dd):
        a.scatter(g, y, s=9, marker=mk, color=c, linewidths=0, alpha=0.85, label=lab)
        for lo, hi in strata:
            i = (g >= lo) & (g < hi); m = np.median(y[i])
            a.plot([lo*1.05, hi/1.05], [m, m], color=c, lw=1.8, alpha=0.9, solid_capstyle='butt')
for a in ax: a.axhline(0, color='0.55', lw=0.7); a.set_xscale('log'); a.set_xlabel('population margin $\\gamma$'); a.set_xticks([1e-4, 1e-3, 1e-2, 1e-1]); grid(a)
for x in (0.01, 0.05):
    for a in ax: a.axvline(x, color='0.6', lw=0.5, ls=':')
ax[0].set_title(r'(a) $\Delta_{\rm PD}$', fontsize=10); ax[1].set_title(r'(b) $\Delta_{\rm PI}$', fontsize=10)
ax[2].set_title(r'(c) $\Delta_{\rm dir}$', fontsize=10)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.0), markerscale=2.5)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig('fig_sens.pdf'); fig.savefig('fig_sens.png', dpi=90)
print('done')
