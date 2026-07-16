"""Compare two screenshots produced by render_screenshot.py.

Usage: python agent_helpers/compare_screenshots.py <baseline.png> <candidate.png> [mean_tol] [frac_tol]

Exits 0 if images match within tolerance, 1 otherwise.
mean_tol: max allowed mean absolute difference in 8-bit counts (default 1.0)
frac_tol: max allowed fraction of pixels differing by more than 8 counts (default 0.005)
"""
import sys

import numpy as np
import PIL.Image


def main():
    a = np.asarray(PIL.Image.open(sys.argv[1]), dtype=float)
    b = np.asarray(PIL.Image.open(sys.argv[2]), dtype=float)
    mean_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    frac_tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005

    if a.shape != b.shape:
        print("FAIL: shape mismatch %s vs %s" % (a.shape, b.shape))
        sys.exit(1)

    diff = np.abs(a - b)
    mean_diff = diff.mean()
    frac_big = (diff > 8).any(axis=-1).mean()
    print("mean abs diff: %.4f (tol %.4f)" % (mean_diff, mean_tol))
    print("fraction of pixels off by >8: %.5f (tol %.5f)" % (frac_big, frac_tol))

    if mean_diff <= mean_tol and frac_big <= frac_tol:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()
