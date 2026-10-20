#!/usr/bin/env python3
"""Independent check of GPU stage 2: which curves must find a known factor p.

usage: gpu_s2_check.py P SIGMA0 CURVES B1 B2 [ECM_LOG [D]]

Works entirely mod the (small) prime P: stage 1 is the param-3 ladder of
gpu_ref_ladder.py; curve i "finds p in stage 1" if its point is O mod P,
and "finds p in stage 2" if r Q = O mod P for some prime B1 < r <= B2.
With D (the GPU's baby-step modulus, printed with -v), stage 2 is modelled
exactly: each prime p = kD +- j puts the pair (k, j) in, and a pair detects
both kD + j and kD - j, so r runs over all of those.
With ECM_LOG (output of `ecm -gpu ...` on a multiple of P) it checks that
the curves reporting a factor divisible by P in step 2 are exactly those.
"""
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from gpu_ref_ladder import primes  # noqa: E402


def ladder_xz(x, z, d, k, p):
    """[k](x:z) on the param-3 curve mod p (a24 = d), Montgomery ladder."""
    def dbl(X, Z):
        A = (X + Z) ** 2 % p
        B = (X - Z) ** 2 % p
        C = A - B
        return A * B % p, C * (B + d * C) % p

    def add(X1, Z1, X2, Z2, Xd, Zd):
        U = (X1 - Z1) * (X2 + Z2)
        V = (X1 + Z1) * (X2 - Z2)
        return Zd * (U + V) ** 2 % p, Xd * (U - V) ** 2 % p

    R0, R1 = (x, z), dbl(x, z)
    for bit in bin(k)[3:]:
        if bit == "1":
            R0, R1 = add(*R0, *R1, x, z), dbl(*R1)
        else:
            R1, R0 = add(*R0, *R1, x, z), dbl(*R0)
    return R0


def main():
    P, sigma0, curves, B1, B2 = (int(float(a)) for a in sys.argv[1:6])
    ps = primes(B2)
    s = 1
    for q in ps:
        if q > B1:
            break
        qq = q
        while qq * q <= B1:
            qq *= q
        s *= qq
    stage2_primes = [r for r in ps if r > B1]
    if len(sys.argv) > 7:
        D = int(sys.argv[7])
        covered = set()
        for r in stage2_primes:
            k = (r + D // 2) // D
            j = abs(r - k * D)
            covered.update((k * D - j, k * D + j))
        stage2_primes = sorted(covered)
    found1, found2 = [], []
    for i in range(curves):
        d = (sigma0 + i) * pow(2 ** 32, -1, P) % P
        # stage 1 from P=(2:1), 2P=(9:64d+8): [s](2:1)
        X, Z = ladder_xz(2, 1, d, s, P)
        if Z % P == 0:
            found1.append(i)
            continue
        x = X * pow(Z, -1, P) % P
        if any(ladder_xz(x, 1, d, r, P)[1] % P == 0 for r in stage2_primes):
            found2.append(i)
    print("stage 1 finds p on curves:", found1)
    print("stage 2 finds p on curves:", found2)
    if len(sys.argv) > 6:
        log = open(sys.argv[6]).read()
        got = sorted(int(c) for f, c in re.findall(
            r"factor (\d+) found in Step 2 with curve (\d+)", log)
            if int(f) % P == 0)
        print("GPU reported step 2 on:   ", got)
        ok = got == found2 and not found1
        print("MATCH" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
