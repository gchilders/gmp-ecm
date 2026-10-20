#!/usr/bin/env python3
"""Independent reference for GPU ECM stage 1 (param 3), plain Python ints.

usage: gpu_ref_ladder.py EXPR SIGMA B1 [SAVEFILE [INDEX]]
       gpu_ref_ladder.py --all EXPR SIGMA0 B1 SAVEFILE LOGFILE [MAXCURVES]

Prints x = X/Z mod N of [s]P for the param-3 curve with this sigma, where
s = product of maximal prime powers <= B1 (compute_s in batch.c).  With a
save file, compares against line INDEX (default 0) of it.
"""
import re
import sys

sys.set_int_max_str_digits(0)


def primes(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def stage1(N, sigma, B1):
    s = 1
    for p in primes(B1):
        q = p
        while q * p <= B1:
            q *= p
        s *= q
    d = sigma * pow(2 ** 32, -1, N) % N
    # P = (2:1), 2P = (9 : 64d + 8); ladder over the bits of s after the top
    xa, za, xb, zb = 2, 1, 9, (64 * d + 8) % N
    for bit in bin(s)[3:]:
        if bit == "1":
            xa, za, xb, zb = xb, zb, xa, za
        # (xa:za) <- 2(xa:za), (xb:zb) <- (xa:za) + (xb:zb), difference (2:1)
        t = (zb + xb) * (za - xa)            # CB
        u = (zb - xb) * (za + xa)            # DA
        AA = (za + xa) ** 2 % N
        BB = (za - xa) ** 2 % N
        K = AA - BB
        nxa = AA * BB % N
        nza = K * (BB + d * K) % N
        nxb = (u + t) ** 2 % N
        nzb = 2 * (u - t) ** 2 % N
        xa, za, xb, zb = nxa, nza, nxb, nzb
        if bit == "1":
            xa, za, xb, zb = xb, zb, xa, za
    return xa * pow(za, -1, N) % N


def stage1_xz(N, sigma, B1):
    """(x, z) variant: returns None for z when the curve finds a factor."""
    try:
        return stage1(N, sigma, B1)
    except ValueError:
        return None


def check_all(expr, sigma0, B1, savefile, logfile, maxcurves=None):
    """Every curve of a GPU run: residue must match mod N (or mod the
    cofactor saved after a factor was found), or the GPU must have reported
    a factor for exactly the curves whose reference Z is not invertible."""
    from math import gcd
    N = eval(expr.replace("^", "**").replace("/", "//"))
    lines = open(savefile).read().splitlines()[:maxcurves]
    log = open(logfile).read()
    bad = 0
    for i, line in enumerate(lines):
        got = int(re.search(r"X=0x([0-9a-f]+)", line).group(1), 16)
        x = stage1_xz(N, sigma0 + i, B1)
        reported = re.search(r"found in Step 1 with curve %d " % i, log)
        if x is None:
            ok = reported is not None
        else:
            g = gcd(x - got, N)
            ok = not reported and (got == x or (g > 1 and (N // g).bit_length() < 200))
        if not ok:
            bad += 1
            print("curve %d: MISMATCH" % i)
    print("%s: %d curves, %d mismatches" % (expr, len(lines), bad))
    return bad == 0


def main():
    if sys.argv[1] == "--all":
        maxc = int(sys.argv[7]) if len(sys.argv) > 7 else None
        sys.exit(0 if check_all(sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])),
                                sys.argv[5], sys.argv[6], maxc) else 1)
    expr, sigma, B1 = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3]))
    N = eval(expr.replace("^", "**").replace("/", "//"))
    x = stage1(N, sigma, B1)
    print("ref X=0x%x" % x if len(sys.argv) <= 4 else "", end="")
    if len(sys.argv) > 4:
        idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        lines = open(sys.argv[4]).read().splitlines()
        got = int(re.search(r"X=0x([0-9a-f]+)", lines[idx]).group(1), 16)
        # once a factor f is found, saves are reduced mod the cofactor N/f
        from math import gcd
        g = gcd(x - got, N)
        if got == x:
            print("MATCH")
        elif g > 1 and (N // g).bit_length() < 200:
            print("MATCH mod cofactor (N/cofactor has %d bits)"
                  % (N // g).bit_length())
        else:
            print("DIFFER (ref %x... save %x...)"
                  % (x >> (x.bit_length() - 48), got >> (got.bit_length() - 48)))
    else:
        print()


if __name__ == "__main__":
    main()
