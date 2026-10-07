"""
model.py -- the second-order BSS model of Section 2 and the estimators of Section 4.

Manuscript cross-references
---------------------------
  X(t) = M S(t) + N(t) ................................ Assumption 1
  A_k = sym Gamma_X(tau_k) = M D_k M^t ................ equation (2)
  hat Gamma_X(tau), hat A_k ........................... Section 4 (first display)
  S_X = sum_h ||Gamma_X(h)||_F^2 ...................... Assumption 2, Lemma 3
  SNR = 10 log10( tr(M M^t) / tr V ) dB ............... Section 6.1, 'Construction of the model'
  Q = B^{-1/2} (whitening matrix) ..................... Theorem 3
The fixed configuration of Sections 6 and 7.3-7.4 uses p = 5 unit-variance AR(2)
sources with pole radius 0.85 and frequencies WS, and a mixing matrix with
prescribed condition number (default 4). Lags are tau_k = k, k = 1..K.
"""
import numpy as np
from scipy.signal import lfilter
from core import *

P = 5                                     # number of sensors = number of sources
WS = [0.35, 0.95, 1.55, 2.15, 2.75]       # AR(2) frequencies omega_i (Section 6.1)
RAD = 0.85                                # AR(2) pole radius r (Section 6.1)
SNR_DB = 5.0                              # default SNR in dB


def make_model(seed=1, snr_db=None, cond=4.0):
    """Fixed configuration of Sections 6 and 7.3-7.4 (drawn once, seed 1).
    M = Q1 diag(1, ..., 1/cond) Q2 with orthogonal Q1, Q2 (QR factors of Gaussian
    matrices), so cond(M) = cond; V = c_V W W^t with c_V chosen so that
    tr V = tr(M M^t) 10^(-SNR/10); sources normalized to unit variance."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((P, P)))
    Q2, _ = np.linalg.qr(rng.standard_normal((P, P)))
    M = Q1 @ np.diag(np.linspace(1.0, 1.0/cond, P)) @ Q2    # singular values 1 ... 1/cond
    W = rng.standard_normal((P, P))
    V = W @ W.T; V /= np.trace(V)/P                         # spatially correlated noise
    sig_pow = np.trace(M @ M.T)/P                           # signal power per sensor
    V *= sig_pow * 10**(-(SNR_DB if snr_db is None else snr_db)/10)   # SNR definition
    a_list = [ar2_coeffs(RAD, w) for w in WS]
    H = 400                                                 # lags kept for S_X
    ac = np.array([acov(a, H) for a in a_list])             # rho_i(h), h = 0..H
    scale = 1/np.sqrt(ac[:, 0])                             # unit-variance sources
    ac = ac*scale[:, None]**2
    return dict(M=M, V=V, a_list=a_list, scale=scale, ac=ac, H=H)


def pop_matrices(md, K):
    """Population matrices A_k = M D_k M^t, D_k = diag(rho_1(k), ..., rho_p(k)),
    k = 1..K: equation (2). The noise does not appear (Assumption 1(iii))."""
    M = md['M']; ac = md['ac']
    return [sym(M @ np.diag(ac[:, k]) @ M.T) for k in range(1, K+1)]


def S_const(md):
    """S_X = sum_{h in Z} ||Gamma_X(h)||_F^2 (Assumption 2), with
    Gamma_X(0) = M Gamma_S(0) M^t + V and Gamma_X(h) = M Gamma_S(h) M^t for h != 0;
    h and -h contribute equally, hence the factor 2."""
    M = md['M']; ac = md['ac']; H = md['H']
    tot = np.linalg.norm(M @ np.diag(ac[:, 0]) @ M.T + md['V'], 'fro')**2
    for h in range(1, H+1):
        tot += 2*np.linalg.norm(M @ np.diag(ac[:, h]) @ M.T, 'fro')**2
    return tot


def simulate(md, T, rng, burn=600):
    """One data set X(1..T) of the model X = M S + N (Assumption 1): the sources are
    AR(2) filters of Gaussian white noise (burn-in 600 samples discarded), the noise
    is Gaussian, temporally white, with covariance V = L L^t."""
    n = P
    S = np.empty((n, T))
    for i, a in enumerate(md['a_list']):
        e = rng.standard_normal(T+burn)
        S[i] = lfilter([1.0], a, e)[burn:]*md['scale'][i]
    L = np.linalg.cholesky(md['V'])
    N = L @ rng.standard_normal((n, T))
    return md['M'] @ S + N


def sample_matrices(X, K):
    """Estimates hat A_k = sym hat Gamma_X(k), with the unbiased estimator
    hat Gamma_X(tau) = (T - tau)^{-1} sum_{t=1}^{T-tau} X(t+tau) X(t)^t (Section 4).
    Applied to half of the sample, it gives hat A_k^(1), hat A_k^(2) (Section 5)."""
    T = X.shape[1]
    return [sym(X[:, tau:] @ X[:, :T-tau].T/(T-tau)) for tau in range(1, K+1)]


def isqrt(B):
    """Whitening matrix Q = B^{-1/2} of a PD matrix B (Theorem 3)."""
    w, U = np.linalg.eigh(B)
    return U @ np.diag(w**-0.5) @ U.T
