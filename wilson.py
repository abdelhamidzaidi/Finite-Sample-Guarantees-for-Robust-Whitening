"""wilson.py -- 95% Wilson score interval for a proportion (Section 7.1):
   ( p + z^2/(2n) +- z sqrt( p(1-p)/n + z^2/(4n^2) ) ) / (1 + z^2/n),  p = k/n, z = 1.96.
Used for every estimated probability in Tables 3-7 and Figures 1, 3, 4."""
import numpy as np


def wilson(k, n, z=1.959964):
    """Return (p_hat, lower, upper) for k successes in n trials."""
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0.0, c-h), min(1.0, c+h)
