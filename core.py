"""
core.py -- the three algorithms compared in the paper and two linear-algebra helpers.

Manuscript cross-references
---------------------------
  A(alpha) = sum_k alpha_k A_k ......................... notation of Section 1.1
  lambda_min, Rayleigh quotient (fact F1) .............. Section 1.4
  g(u) = (u^t A_1 u, ..., u^t A_K u) ................... Section 3, identity (3)
  K = conv{ g(u) : ||u|| = 1 } ......................... Section 3, Lemma 1
  mnp()  = Algorithms 1 and 4 (minimum-norm point, MNP-first / MNP-max / MMRW)
  pc()   = Algorithm 2 (Tong et al. [4])
  pdc()  = Algorithm 3 (PDC of [5], with the step of Remark 5)

Every function takes the list As = [A_1, ..., A_K] of symmetric p x p matrices.
In the experiments these are either the population matrices A_k of (2) or the
estimates hat A_k of Section 4.
"""
import numpy as np
from scipy.signal import lfilter


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sym(A):
    """Symmetric part sym(G) = (G + G^t)/2 (notation of Section 1.4)."""
    return 0.5*(A + A.T)


def ar2_coeffs(r, w):
    """Denominator coefficients (1, -2 r cos w, r^2) of the AR(2) recursion
    s(t) = 2 r cos(w) s(t-1) - r^2 s(t-2) + e(t) used in Section 6.1
    (poles r e^{+-i w})."""
    return np.array([1.0, -2*r*np.cos(w), r*r])


def psi_weights(a, L=4000):
    """First L coefficients psi_j of the MA(infinity) representation
    s(t) = sum_j psi_j e(t-j) of the AR process with coefficients a
    (impulse response of the filter 1/a(z))."""
    imp = np.zeros(L); imp[0] = 1.0
    return lfilter([1.0], a, imp)


def acov(a, H, L=4000):
    """Autocovariances rho(h) = sum_j psi_j psi_{j+h}, h = 0..H, of the AR process with
    unit innovation variance, evaluated numerically from the MA(inf) representation
    truncated after L = 4000 terms (Section 6.1); for pole radii <= 0.9 the omitted tail
    is below machine precision."""
    psi = psi_weights(a, L)
    return np.array([np.dot(psi[:L-h], psi[h:]) for h in range(H+1)])


def lmin(B):
    """Smallest eigenvalue lambda_min(B) and a unit eigenvector u.
    By fact (F1), u attains min_{||u||=1} u^t B u."""
    w, V = np.linalg.eigh(B)          # eigenvalues in ascending order
    return w[0], V[:, 0]


# ----------------------------------------------------------------------------
# Algorithm 1 / 4: minimum-norm-point iteration (von Neumann - Gilbert)
# ----------------------------------------------------------------------------
LAST_STATUS = None
"""Status of the last call of mnp(), as in Algorithm 4:
   'converged'     certified tolerance reached (MNP-max output),
   'not-converged' cap reached, final iterate PD (returned, flagged),
   'reverted'      cap reached, final iterate not PD, last PD iterate returned,
   'no-pd'         no PD iterate within the cap,
   'infeasible'    y = 0 reached: exact certificate that no PD combination exists."""


def mnp(As, maxit=5000, tol=None, stop_first=False, y0=None):
    """Minimum-norm-point iteration on the set K of Lemma 1 (Algorithms 1 and 4).

    Returns (y, lower, upper, n, first):
      y      the returned point of K; the returned direction is y/||y||;
      lower  certified lower bound lambda_min(A(y))/||y|| on the margin
             (Proposition 3(i)); it belongs to the returned y;
      upper  certified upper bound ||y_final|| on the margin (Proposition 2(iii));
      n      number of iterations;
      first  (y_first, n_first): the first PD iterate (MNP-first), or None.
    With tol = eta, the loop stops when upper - lower <= eta * upper (Algorithm 1,
    line 4). With stop_first=True it stops at the first PD iterate (MNP-first).
    The status is stored in the module variable LAST_STATUS.
    """
    global LAST_STATUS
    K = len(As); p = As[0].shape[0]
    Astack = np.stack(As)                                   # shape (K, p, p)
    # g(u) = (u^t A_k u)_k, the vector of Rayleigh quotients, identity (3)
    g = lambda u: np.einsum('i,kij,j->k', u, Astack, u)
    # y_0 = (trace A_k / p)_k = average of g(e_j) over the standard basis,
    # hence y_0 lies in K (Appendix C, proof of (i))
    y = np.array([np.trace(A)/p for A in As]) if y0 is None else y0.copy()
    first = None        # MNP-first output
    last_pos = None     # last PD iterate and its lower bound (Algorithm 4, y_pos)
    for n in range(maxit):
        ny = np.linalg.norm(y)
        if ny == 0:
            # y = 0 in K: exact certificate of infeasibility (Proposition 3(iv))
            LAST_STATUS = 'infeasible'; return y, np.nan, 0.0, n, first
        B = np.tensordot(y, Astack, 1)                      # A(y_n) = sum_k y_k A_k
        lam, u = lmin(B)                                    # lambda_n and u_n
        if lam > 0:
            # certified bounds lambda_n/||y_n|| <= gamma_hat <= ||y_n|| (Prop. 3(i))
            last_pos = (y.copy(), lam/ny)
            if first is None:
                first = (y.copy(), n)
                if stop_first:
                    LAST_STATUS = 'converged'; return y, lam/ny, ny, n, first
            if tol is not None and ny - lam/ny <= tol*ny:   # stopping rule
                LAST_STATUS = 'converged'; return y, lam/ny, ny, n, first
        # Frank-Wolfe step: s_n = g(u_n) minimizes <y_n, .> over K (Lemma 1(b)),
        # omega_n = <y_n, y_n - s_n> = ||y_n||^2 - lambda_n is the duality gap,
        # and t_n = min(1, omega_n / ||y_n - s_n||^2) minimizes ||y + t(s - y)||
        # on the segment [y_n, s_n] (exact line search, Appendix C).
        s = g(u); d = y - s; dd = d@d
        t = min(1.0, max(0.0, (y@d)/dd))                    # y@d = omega_n
        y = y - t*d                                         # y_{n+1} = (1-t) y_n + t s_n
    # ---- iteration cap reached (Algorithm 4, final branch) ----
    ny = np.linalg.norm(y)
    B = np.tensordot(y, Astack, 1); lam, _ = lmin(B)
    if lam > 0:
        LAST_STATUS = 'not-converged'; return y, lam/ny, ny, maxit, first
    if last_pos is not None:
        # final iterate not PD: return the last PD iterate with its own lower bound;
        # ||y_final|| remains a valid certified upper bound on the margin
        LAST_STATUS = 'reverted'; return last_pos[0], last_pos[1], ny, maxit, first
    LAST_STATUS = 'no-pd'; return y, lam/ny, ny, maxit, first


# ----------------------------------------------------------------------------
# Algorithm 2: PC, the perceptron-type method of Tong et al. [4]
# ----------------------------------------------------------------------------
def pc(As, maxit=100000, alpha0=None, rng=None):
    """alpha_{n+1} = alpha_n + g(u_n)/||g(u_n)||, where u_n is a unit eigenvector
    for lambda_min(A(alpha_n)); stop at the first PD combination.
    Proposition 4(i): at most 3 (R/gamma)^2 iterations when gamma > 0.
    Returns (alpha, n); n == maxit means 'not terminated within the cap'."""
    Astack = np.stack(As); K = len(As)
    a = np.array([np.trace(A) for A in As]) if alpha0 is None else alpha0.copy()
    a = a/np.linalg.norm(a)                                 # unit starting vector
    for n in range(maxit):
        lam, u = lmin(np.tensordot(a, Astack, 1))           # lambda_min(A(alpha_n)), u_n
        if lam > 0: return a, n
        gu = np.einsum('i,kij,j->k', u, Astack, u)          # g(u_n), identity (3)
        a = a + gu/np.linalg.norm(gu)                       # d_n = g(u_n)/||g(u_n)||
    return a, maxit


# ----------------------------------------------------------------------------
# Algorithm 3: PDC of [5] with the corrected step of Remark 5
# ----------------------------------------------------------------------------
def pdc(As, maxit=100000, alpha0=None, sigma=1.0):
    """Alternating projections B_{n+1} = P_E( P_S(B_n) ), where
    S = {C : lambda_min(C) >= sigma} and E = span{A_k} (Proposition 4(ii)).
    P_S(B) = U diag(max(lambda_i, sigma)) U^t   (Remark 5, corrected step);
    P_E(C) = sum_k alpha_k A_k with alpha = G^{-1} (<A_k, C>)_k, G the Gram matrix.
    Returns (alpha, n); n == maxit means 'not terminated within the cap'."""
    Astack = np.stack(As); K = len(As); p = As[0].shape[0]
    Amat = np.stack([A.ravel() for A in As], 1)             # p^2 x K, columns vec(A_k)
    G = Amat.T @ Amat                                       # Gram matrix <A_k, A_l>
    proj = lambda C: np.linalg.solve(G, Amat.T @ C.ravel()) # coefficients of P_E(C)
    a = np.array([np.trace(A) for A in As]) if alpha0 is None else alpha0.copy()
    if not np.any(a):                                       # Algorithm 3 requires alpha_0 != 0
        raise ValueError('pdc: the starting vector alpha_0 must be nonzero')
    B = np.tensordot(a, Astack, 1)
    lam, U = np.linalg.eigh(B)
    # initial transformation of [5]: scale by e*sigma/mu, with mu the smallest
    # nonzero |lambda_i| and e the sign of the majority of the eigenvalues
    nz = np.abs(lam[np.abs(lam) > 0])
    mu = nz.min()
    eps = 1.0 if (lam > 0).sum() >= (lam < 0).sum() else -1.0
    a = a*eps*sigma/mu
    for n in range(maxit):
        B = np.tensordot(a, Astack, 1)                      # B_n = A(alpha_n)
        lam, U = np.linalg.eigh(B)
        if lam[0] > 0: return a, n                          # first PD iterate
        D = np.maximum(lam, sigma)                          # corrected step (Remark 5)
        a = proj(U @ np.diag(D) @ U.T)                      # alpha_{n+1}: B_{n+1} = P_E(P_S(B_n))
    return a, maxit
