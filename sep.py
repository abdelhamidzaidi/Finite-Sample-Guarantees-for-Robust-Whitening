"""
sep.py -- source separation after whitening, and its quality measure (Section 7.1).

  joint_diag(): Jacobi-angle joint diagonalization of Cardoso and Souloumiac [11],
                applied to the whitened matrices Q hat A_k Q, k = 1..K, as in SOBI.
  amari():      Amari performance index PI(G) of the global matrix G = W M,
                formula of Section 7.1 (0 = perfect separation, 1 = worst).
"""
import numpy as np


def joint_diag(As, tol=1e-10, maxsweep=100):
    """Orthogonal V such that V^t A_k V is as diagonal as possible for all k.
    Each Jacobi rotation acts on the pair (i, j) with the angle that maximizes the
    sum over k of the squared diagonal entries (closed form of [11]).
    The separating matrix is then W = V^t Q."""
    A = np.array(As, dtype=float).copy(); K, p, _ = A.shape
    V = np.eye(p)
    for sweep in range(maxsweep):
        rot = False
        for i in range(p-1):
            for j in range(i+1, p):
                g0 = A[:, i, i] - A[:, j, j]; g1 = A[:, i, j] + A[:, j, i]
                ton = g0@g0 - g1@g1; toff = 2*(g0@g1)
                theta = 0.5*np.arctan2(toff, ton + np.sqrt(ton*ton + toff*toff))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) > tol:
                    rot = True
                    G = np.array([[c, -s], [s, c]]); idx = [i, j]
                    V[:, idx] = V[:, idx] @ G
                    A[:, idx, :] = np.einsum('ab,kbc->kac', G.T, A[:, idx, :])
                    A[:, :, idx] = A[:, :, idx] @ G
        if not rot: break
    return V


def amari(G):
    """PI(G) = [ sum_i ( sum_j |g_ij| / max_l |g_il| - 1 )
               + sum_j ( sum_i |g_ij| / max_l |g_lj| - 1 ) ] / (2 p (p-1))
    (Section 7.1): the average ratio of a non-dominant entry to the dominant one."""
    Ga = np.abs(G); p = G.shape[0]
    r = (Ga/Ga.max(1, keepdims=True)).sum(1) - 1      # rows
    c = (Ga/Ga.max(0, keepdims=True)).sum(0) - 1      # columns
    return (r.sum() + c.sum())/(2*p*(p-1))
