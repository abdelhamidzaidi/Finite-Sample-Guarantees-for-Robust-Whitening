"""
compare.py -- value-by-value comparison of a fresh rerun with the stored results.

Usage:  python compare.py STORED_DIR FRESH_DIR

Every .pkl file in STORED_DIR is loaded together with its counterpart in FRESH_DIR and
walked recursively. Computing times are skipped (keys 'time', 't_max', and the second
entry of the per-method tuples (iterations, time, terminated, lambda_min) of the exact
families). For every other leaf the script records whether the two values are
  identical      (bit-for-bit, NaN equal to NaN),
  close          (relative difference <= 1e-9, a floating-point rounding difference),
  different      (anything else, including a changed boolean, count or status).
"""
import sys, os, glob, pickle, math
import numpy as np

STORED, FRESH = sys.argv[1], sys.argv[2]
TIME_KEYS = {'time', 't_max'}
stats = {'identical': 0, 'close': 0, 'different': 0}
worst_rel = 0.0
examples = []


def leaf_equal(a, b):
    """Return 'identical', 'close' or 'different' for two scalar leaves."""
    global worst_rel
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)) or isinstance(a, str):
        return 'identical' if a == b else 'different'
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return 'identical' if a == b else 'different'
    if (math.isnan(fa) and math.isnan(fb)) or fa == fb:
        return 'identical'
    if math.isnan(fa) or math.isnan(fb) or math.isinf(fa) or math.isinf(fb):
        return 'different'
    rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)
    worst_rel = max(worst_rel, rel)
    return 'close' if rel <= 1e-9 else 'different'


def walk(a, b, path, fname):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            stats['different'] += 1; examples.append((fname, path, 'keys differ')); return
        for k in a:
            if k in TIME_KEYS: continue
            walk(a[k], b[k], path + [k], fname)
        return
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            stats['different'] += 1; examples.append((fname, path, 'length differs')); return
        # exact-family tuples (iterations, time, terminated, lambda_min): skip the time
        skip_time = fname.startswith('exact_') and len(path) >= 1 and path[-1] in ('MNP', 'PC', 'PDC')
        for i, (x, y) in enumerate(zip(a, b)):
            if skip_time and i == 1: continue
            walk(x, y, path + [i], fname)
        return
    if isinstance(a, np.ndarray):
        a2, b2 = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
        if a2.shape != b2.shape:
            stats['different'] += 1; examples.append((fname, path, 'shape differs')); return
        for i, (x, y) in enumerate(zip(a2, b2)):
            r = leaf_equal(x, y); stats[r] += 1
            if r == 'different' and len(examples) < 30: examples.append((fname, path + [i], (x, y)))
        return
    r = leaf_equal(a, b); stats[r] += 1
    if r == 'different' and len(examples) < 30: examples.append((fname, path, (a, b)))


files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(STORED, '*.pkl')))
missing = [f for f in files if not os.path.exists(os.path.join(FRESH, f))]
per_file_diff = {}
for f in files:
    if f in missing: continue
    before = stats['different']
    walk(pickle.load(open(os.path.join(STORED, f), 'rb')), pickle.load(open(os.path.join(FRESH, f), 'rb')), [], f)
    per_file_diff[f] = stats['different'] - before
print('stored result files: %d, missing from fresh run: %d %s' % (len(files), len(missing), missing[:5]))
print('values compared: %d' % sum(stats.values()))
print('  identical: %d   close (rel <= 1e-9): %d   different: %d' % (stats['identical'], stats['close'], stats['different']))
print('  largest relative difference among close values: %.2e' % worst_rel)
print('files with differences: %s' % {k: v for k, v in per_file_diff.items() if v})
for e in examples[:30]: print('   ', e)
