/* gpu_ntt.cuh: NTT-based Montgomery multiplication for large inputs.

Copyright 2026 Greg Childers

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/* Numbers are W 32-bit limbs, W = L/2, in per-curve global memory; one
 * thread block works on one curve.  A product of two W-limb numbers is a
 * cyclic convolution of length L (no wrap), computed modulo three primes
 * p < 2^30 with 2^23 | p - 1 and recombined by CRT (Garner).  Their
 * product (2^89.35) exceeds every coefficient, < W 2^64 <= 2^77 for
 * L <= 2^14, so the recombination is exact.
 *
 * Transforms run in shared memory (L words plus one pad word per 32, see
 * NTT_PAD): forward decimation in frequency (natural order in, bit-reversed
 * out), inverse decimation in time (bit-reversed in, natural out), so no
 * permutation is ever done.  Each pass does up to three radix-2 stages in
 * registers on 8 elements per thread.  Butterflies are lazy (values in
 * [0, 2p), 4p < 2^32) with Shoup multiplication by twiddles; pointwise
 * products use 32-bit Montgomery reduction, whose 2^-32 is folded into the
 * final 1/L scaling.
 *
 * Montgomery multiplication mod N with R = 2^(32W):
 *   T = a b;  q = (T mod R) N' mod R  (N' = -1/N mod R);  r = (T + q N)/R.
 * Transforms of N and N' are precomputed.  With R >= 16N the bounds are
 * those of the schoolbook kernels (see gpu_stage1.cu): r < 2N whenever
 * a b < 16 N^2.
 *
 * All block-wide operations end with __syncthreads(), and all threads of
 * the block must call them. */

#ifndef _GPU_NTT_CUH
#define _GPU_NTT_CUH 1

#include <stdint.h>

#define NTT_NPRIMES 3
#define NTT_MIN_LG 10
#define NTT_MAX_LG 14
#define NTT_MAX_THREADS 1024

/* Shared-memory index of element i: a pad word after every 32 keeps the
 * strided accesses of the transforms and the carry runs free of bank
 * conflicts.  Buffers of n elements take NTT_PADDED (n) words. */
#define NTT_PAD(i) ((i) + ((i) >> 5))
#define NTT_PADDED(n) ((n) + ((n) >> 5))

/* The three primes; 2^23 divides p - 1 for each. */
#define NTT_P0 998244353u   /* 119 * 2^23 + 1 */
#define NTT_P1 897581057u   /* 107 * 2^23 + 1 */
#define NTT_P2 880803841u   /* 105 * 2^23 + 1 */

struct ntt_prime {
  uint32_t p;
  uint32_t pinv;          /* -1/p mod 2^32 (Montgomery) */
  uint32_t one_sh;        /* floor(2^32 / p): Shoup constant of 1 */
  uint32_t scale, scale_sh;  /* 2^32 / L mod p and its Shoup constant */
  const uint2 *w;         /* forward twiddles (w^k, Shoup), k < L/2 */
  const uint2 *wi;        /* inverse twiddles */
  const uint32_t *Nhat;   /* transforms of N and N' (bit-reversed order) */
  const uint32_t *Nphat;
};

struct ntt_ctx {
  int lg, L, W;
  ntt_prime pr[NTT_NPRIMES];
  /* Garner */
  uint32_t c01, c01_sh;   /* 1/p0 mod p1 */
  uint32_t c02, c02_sh;   /* 1/p0 mod p2 */
  uint32_t c12, c12_sh;   /* 1/p1 mod p2 */
  uint64_t p01;           /* p0 p1 */
  const uint32_t *N, *N2; /* N and 2N, W limbs */
  uint32_t np0;           /* -1/N mod 2^32 */
};

/* Per-curve scratch, in words: see ntt_scratch_words. */
struct ntt_scratch {
  uint32_t *ahat;         /* L: transform of the first operand */
  uint32_t *res[NTT_NPRIMES];  /* L each: inverse transforms */
  uint32_t *T, *Q, *U;    /* L each */
  uint32_t *mw;           /* W + 1: mul_ui */
};

#ifdef __CUDACC__
__host__ __device__
#endif
static inline size_t
ntt_scratch_words (int L)
{
  return (size_t) L * (1 + NTT_NPRIMES + 3) + L / 2 + 64;
}

#ifdef __CUDACC__

#define NTT_DEV __device__ __forceinline__

/* Per-phase timing (gpu_dev/ntt_phase_bench.cu): with NTT_PROFILE defined,
 * thread 0 charges the clock64 time since the previous mark to phase ID.
 * Marks follow a __syncthreads, so a phase covers all threads' work. */
enum { NTT_PH_FWD, NTT_PH_MID, NTT_PH_INV, NTT_PH_CRT, NTT_PH_FINAL,
       NTT_PH_MW, NTT_NPROF };
#ifdef NTT_PROFILE
#define NTT_PROF(S, id) do { if (threadIdx.x == 0) {                       \
      long long t_ = clock64 (); (S).prof[id] += t_ - (S).prof_t;           \
      (S).prof_t = t_; } } while (0)
#else
#define NTT_PROF(S, id) do { } while (0)
#endif

struct ntt_shared {
  uint32_t scan[3 * (NTT_MAX_THREADS / 32)];
  uint64_t xchg[NTT_MAX_THREADS];
  uint32_t word;
#ifdef NTT_PROFILE
  unsigned long long prof[NTT_NPROF];
  long long prof_t;
#endif
};

/* ---- arithmetic modulo one prime ------------------------------------- */

/* a * w mod p in [0, 2p), for any a < 2^32 and w < p; wsh = floor(w 2^32/p) */
NTT_DEV uint32_t
ntt_shoup (uint32_t a, uint32_t w, uint32_t wsh, uint32_t p)
{
  uint32_t q = __umulhi (a, wsh);
  return a * w - q * p;
}

NTT_DEV uint32_t
ntt_red (uint32_t x, uint32_t m)    /* x < 2m -> [0, m) */
{
  return x >= m ? x - m : x;
}

/* a b 2^-32 mod p in [0, 2p), for a, b < 2p */
NTT_DEV uint32_t
ntt_mont (uint32_t a, uint32_t b, uint32_t p, uint32_t pinv)
{
  uint64_t t = (uint64_t) a * b;
  uint32_t m = (uint32_t) t * pinv;
  return (uint32_t) ((t + (uint64_t) m * p) >> 32);
}

/* ---- transforms in shared memory ------------------------------------- */

/* One forward pass: the R radix-2 DIF stages s, s-1, ..., s-R+1 (half
 * sizes 2^s down to q = 2^(s-R+1)), each thread taking groups of 2^R
 * elements q apart through registers.  (Two groups at a time would give
 * more independent work, but spills under the 64-register cap.)  With
 * LOAD, the input comes from the first n limbs of src (global memory, zero
 * above) instead of x: the first pass does the load. */
template <int R, bool LOAD>
NTT_DEV void
ntt_fwd_pass (uint32_t *x, const ntt_ctx &c, const ntt_prime &P, int s,
              const uint32_t *src, int n)
{
  const uint32_t p = P.p, p2 = 2 * p;
  const int lq = s - R + 1, q = 1 << lq;
  for (int g = threadIdx.x; g < (c.L >> R); g += blockDim.x)
    {
      const int k = g & (q - 1);
      const int base = ((g >> lq) << (s + 1)) + k;
      uint32_t v[1 << R];
#pragma unroll
      for (int t = 0; t < (1 << R); t++)
        {
          const int i = base + t * q;
          if (LOAD)
            v[t] = i < n ? ntt_shoup (src[i], 1, P.one_sh, p) : 0;
          else
            v[t] = x[NTT_PAD (i)];
        }
#pragma unroll
      for (int r = 0; r < R; r++)
        {
          /* stage s - r: pairs (t, t + hs); the lower element's index in
             its half block is k + q (t mod hs) */
          const int hs = 1 << (R - 1 - r), sh = c.lg - 1 - (s - r);
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            if (!(t & hs))
              {
                const uint2 w = __ldg (&P.w[(k + (t & (hs - 1)) * q) << sh]);
                const uint32_t a = v[t], b = v[t + hs];
                v[t] = ntt_red (a + b, p2);
                v[t + hs] = ntt_shoup (a - b + p2, w.x, w.y, p);
              }
        }
#pragma unroll
      for (int t = 0; t < (1 << R); t++)
        x[NTT_PAD (base + t * q)] = v[t];
    }
  __syncthreads ();
}

/* One inverse pass: the R radix-2 DIT stages s0, ..., s0+R-1 (half sizes
 * q = 2^s0 up).  With STORE (the last pass) the results go, scaled by
 * 2^32/L and canonical, to res in global memory instead of x. */
template <int R, bool STORE>
NTT_DEV void
ntt_inv_pass (uint32_t *x, const ntt_ctx &c, const ntt_prime &P, int s0,
              uint32_t *res)
{
  const uint32_t p = P.p, p2 = 2 * p;
  const int q = 1 << s0;
  for (int g = threadIdx.x; g < (c.L >> R); g += blockDim.x)
    {
      const int k = g & (q - 1);
      const int base = ((g >> s0) << (s0 + R)) + k;
      uint32_t v[1 << R];
#pragma unroll
      for (int t = 0; t < (1 << R); t++)
        v[t] = x[NTT_PAD (base + t * q)];
#pragma unroll
      for (int r = 0; r < R; r++)
        {
          const int hs = 1 << r, sh = c.lg - 1 - (s0 + r);
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            if (!(t & hs))
              {
                const uint2 w = __ldg (&P.wi[(k + (t & (hs - 1)) * q) << sh]);
                const uint32_t a = ntt_red (v[t], p2);
                const uint32_t b = ntt_shoup (v[t + hs], w.x, w.y, p);
                v[t] = ntt_red (a + b, p2);
                v[t + hs] = ntt_red (a - b + p2, p2);
              }
        }
#pragma unroll
      for (int t = 0; t < (1 << R); t++)
        if (STORE)
          res[base + t * q] = ntt_red (ntt_shoup (v[t], P.scale, P.scale_sh, p), p);
        else
          x[NTT_PAD (base + t * q)] = v[t];
    }
  __syncthreads ();
}

/* x <- the forward transform of a (n limbs, zero padded to L) except for
 * its last three stages, which ntt_mid does.  The passes are aligned so
 * that the one on contiguous groups of 8 (conflict-free only thanks to the
 * padding) is ntt_mid's, and the odd stages go where the strides are
 * large. */
NTT_DEV void
ntt_forward_head (uint32_t *x, const ntt_ctx &c, const ntt_prime &P,
                  const uint32_t *a, int n)
{
  int s = c.lg - 1;
  switch (c.lg % 3)
    {
    case 1: ntt_fwd_pass<1, true> (x, c, P, s, a, n); s -= 1; break;
    case 2: ntt_fwd_pass<2, true> (x, c, P, s, a, n); s -= 2; break;
    default: ntt_fwd_pass<3, true> (x, c, P, s, a, n); s -= 3; break;
    }
  for (; s > 2; s -= 3)
    ntt_fwd_pass<3, false> (x, c, P, s, NULL, 0);
}

enum { NTT_MID_SQUARE, NTT_MID_MUL, NTT_MID_SAVE };

/* On contiguous groups of 8: the last three forward stages, then
 *   SQUARE: square pointwise;  MUL: multiply by the transform B (global);
 *   SAVE: write the transform to B and stop,
 * then (SQUARE, MUL) the first three inverse stages. */
template <int MODE>
NTT_DEV void
ntt_mid (uint32_t *x, const ntt_ctx &c, const ntt_prime &P, uint32_t *B)
{
  const uint32_t p = P.p, p2 = 2 * p;
  for (int g = threadIdx.x; g < (c.L >> 3); g += blockDim.x)
    {
      const int base = g << 3;
      uint32_t v[8];
#pragma unroll
      for (int t = 0; t < 8; t++)
        v[t] = x[NTT_PAD (base + t)];
#pragma unroll
      for (int r = 0; r < 3; r++)       /* stages 2, 1, 0 */
        {
          const int hs = 4 >> r, sh = c.lg - 3 + r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = __ldg (&P.w[(t & (hs - 1)) << sh]);
                const uint32_t a = v[t], b = v[t + hs];
                v[t] = ntt_red (a + b, p2);
                v[t + hs] = ntt_shoup (a - b + p2, w.x, w.y, p);
              }
        }
      uint4 *B4 = (uint4 *) (B + base);
      if (MODE == NTT_MID_SAVE)
        {
          B4[0] = make_uint4 (v[0], v[1], v[2], v[3]);
          B4[1] = make_uint4 (v[4], v[5], v[6], v[7]);
          continue;
        }
      if (MODE == NTT_MID_SQUARE)
        {
#pragma unroll
          for (int t = 0; t < 8; t++)
            v[t] = ntt_mont (v[t], v[t], p, P.pinv);
        }
      else
        {
          const uint4 b0 = B4[0], b1 = B4[1];
          v[0] = ntt_mont (v[0], b0.x, p, P.pinv);
          v[1] = ntt_mont (v[1], b0.y, p, P.pinv);
          v[2] = ntt_mont (v[2], b0.z, p, P.pinv);
          v[3] = ntt_mont (v[3], b0.w, p, P.pinv);
          v[4] = ntt_mont (v[4], b1.x, p, P.pinv);
          v[5] = ntt_mont (v[5], b1.y, p, P.pinv);
          v[6] = ntt_mont (v[6], b1.z, p, P.pinv);
          v[7] = ntt_mont (v[7], b1.w, p, P.pinv);
        }
#pragma unroll
      for (int r = 0; r < 3; r++)       /* stages 0, 1, 2 */
        {
          const int hs = 1 << r, sh = c.lg - 1 - r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = __ldg (&P.wi[(t & (hs - 1)) << sh]);
                const uint32_t a = ntt_red (v[t], p2);
                const uint32_t b = ntt_shoup (v[t + hs], w.x, w.y, p);
                v[t] = ntt_red (a + b, p2);
                v[t + hs] = ntt_red (a - b + p2, p2);
              }
        }
#pragma unroll
      for (int t = 0; t < 8; t++)
        x[NTT_PAD (base + t)] = v[t];
    }
  __syncthreads ();
}

/* The inverse transform from stage 3 on, the last pass writing res. */
NTT_DEV void
ntt_inverse_tail (uint32_t *x, const ntt_ctx &c, const ntt_prime &P,
                  uint32_t *res)
{
  int left = c.lg - 3, s0 = 3;
  const int last = left % 3 ? left % 3 : 3;
  for (; left > last; left -= 3, s0 += 3)
    ntt_inv_pass<3, false> (x, c, P, s0, NULL);
  switch (last)
    {
    case 1: ntt_inv_pass<1, true> (x, c, P, s0, res); break;
    case 2: ntt_inv_pass<2, true> (x, c, P, s0, res); break;
    default: ntt_inv_pass<3, true> (x, c, P, s0, res); break;
    }
}

/* res = the cyclic convolution of a (na limbs) and b, mod P, canonical.
 * b is given either as limbs (nb of them; the transform of a then goes
 * through ahat), or already transformed (bhat), or is a itself (square). */
NTT_DEV void
ntt_conv_prime (uint32_t *sm, ntt_shared &S, const ntt_ctx &c,
                const ntt_prime &P, uint32_t *ahat, const uint32_t *a, int na,
                const uint32_t *b, int nb, const uint32_t *bhat,
                bool square, uint32_t *res)
{
  ntt_forward_head (sm, c, P, a, na);
  NTT_PROF (S, NTT_PH_FWD);
  if (square)
    ntt_mid<NTT_MID_SQUARE> (sm, c, P, NULL);
  else if (bhat != NULL)
    ntt_mid<NTT_MID_MUL> (sm, c, P, (uint32_t *) bhat);
  else
    {
      ntt_mid<NTT_MID_SAVE> (sm, c, P, ahat);
      NTT_PROF (S, NTT_PH_MID);
      ntt_forward_head (sm, c, P, b, nb);
      NTT_PROF (S, NTT_PH_FWD);
      ntt_mid<NTT_MID_MUL> (sm, c, P, ahat);
    }
  NTT_PROF (S, NTT_PH_MID);
  ntt_inverse_tail (sm, c, P, res);
  NTT_PROF (S, NTT_PH_INV);
}

/* ---- block-wide carries ---------------------------------------------- */


/* Each thread owns a run of limbs; GEN is its carry out (0/1) and PROP
 * whether an incoming carry would pass through.  Returns the carry into
 * this thread's run; *top gets the carry out of the whole block.  The same
 * serves borrows (PROP = run is all zeros). */
NTT_DEV uint32_t
ntt_block_resolve (ntt_shared &S, uint32_t gen, uint32_t prop, uint32_t *top)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  uint32_t G = __ballot_sync (0xffffffffu, gen);
  uint32_t P = __ballot_sync (0xffffffffu, prop);
  if (lane == 0)
    {
      S.scan[warp] = G;
      S.scan[nw + warp] = P;
    }
  __syncthreads ();
  if (threadIdx.x == 0)
    {
      /* C = ((G << 1) + P) ^ P over the nw-word numbers */
      uint64_t cy = 0;
      for (int w = 0; w < nw; w++)
        {
          uint32_t g = S.scan[w], pw = S.scan[nw + w];
          uint32_t gs = (g << 1) | (w > 0 ? S.scan[w - 1] >> 31 : 0);
          uint64_t sum = (uint64_t) gs + pw + cy;
          S.scan[2 * nw + w] = (uint32_t) sum ^ pw;
          cy = sum >> 32;
        }
      S.word = (uint32_t) cy | (S.scan[nw - 1] >> 31);
    }
  __syncthreads ();
  uint32_t cin = (S.scan[2 * nw + warp] >> lane) & 1;
  if (top != NULL)
    *top = S.word;
  __syncthreads ();
  return cin;
}

/* Add the 0/1 carry CIN to r[0..n) (local run), return carry out. */
NTT_DEV uint32_t
ntt_run_inc (uint32_t *r, int n, uint32_t cin)
{
  for (int i = 0; i < n && cin; i++)
    {
      r[i] += 1;
      cin = r[i] == 0;
    }
  return cin;
}

NTT_DEV uint32_t
ntt_run_dec (uint32_t *r, int n, uint32_t bin)
{
  for (int i = 0; i < n && bin; i++)
    {
      bin = r[i] == 0;
      r[i] -= 1;
    }
  return bin;
}

NTT_DEV int
ntt_all_ones (const uint32_t *r, int n)
{
  uint32_t a = 0xffffffffu;
  for (int i = 0; i < n; i++)
    a &= r[i];
  return a == 0xffffffffu;
}

NTT_DEV int
ntt_all_zeros (const uint32_t *r, int n)
{
  uint32_t a = 0;
  for (int i = 0; i < n; i++)
    a |= r[i];
  return a == 0;
}

/* out[0..n) = the low n limbs of sum_j c_j 2^(32j), c_j the CRT of
 * res[i][j]; n is L or W (carries only go up, so the low limbs need only
 * the low coefficients).  The whole sum must fit in L limbs.  Works on tiles of LT = L/4 coefficients: Garner reads
 * the residues coalesced and leaves each c_j (< 2^77) as three words in the
 * shared buffer SM (3 NTT_PADDED (LT) <= NTT_PADDED (L) words); then each
 * thread sums a run of LT/B >= 2 of them, the runs' carries are resolved
 * block-wide, and the tile goes out coalesced.  A tile's carry out goes
 * into the next tile's first run. */
NTT_DEV void
ntt_crt_carry (uint32_t *sm, ntt_shared &S, const ntt_ctx &c,
               uint32_t *const res[3], uint32_t *out, int n)
{
  const uint32_t p0 = c.pr[0].p, p1 = c.pr[1].p, p2 = c.pr[2].p;
  const int LT = c.L / 4, B = blockDim.x;
  const int run = LT / B, j0 = threadIdx.x * run;
  uint32_t *t0 = sm, *t1 = sm + NTT_PADDED (LT), *t2 = sm + 2 * NTT_PADDED (LT);
  uint64_t tile_in = 0;
  for (int base = 0; base < n; base += LT)
    {
      for (int j = threadIdx.x; j < LT; j += B)
        {
          const uint32_t r0 = res[0][base + j], r1 = res[1][base + j],
                         r2 = res[2][base + j];
          /* Garner: x = r0 + p0 v1 + p0 p1 v2 */
          uint32_t u = ntt_red (r0, p1);
          uint32_t d = r1 >= u ? r1 - u : r1 + p1 - u;
          uint32_t v1 = ntt_red (ntt_shoup (d, c.c01, c.c01_sh, p1), p1);
          uint32_t u2 = ntt_red (r0, p2);
          uint32_t d2 = r2 >= u2 ? r2 - u2 : r2 + p2 - u2;
          uint32_t t = ntt_red (ntt_shoup (d2, c.c02, c.c02_sh, p2), p2);
          uint32_t v1m = ntt_red (v1, p2);
          uint32_t e = t >= v1m ? t - v1m : t + p2 - v1m;
          uint32_t v2 = ntt_red (ntt_shoup (e, c.c12, c.c12_sh, p2), p2);
          uint64_t lo = (uint64_t) p0 * v1 + r0;
          uint64_t mlo = c.p01 * v2, mhi = __umul64hi (c.p01, (uint64_t) v2);
          uint64_t x01 = lo + mlo;
          t0[NTT_PAD (j)] = (uint32_t) x01;
          t1[NTT_PAD (j)] = (uint32_t) (x01 >> 32);
          t2[NTT_PAD (j)] = (uint32_t) mhi + (x01 < lo);
        }
      __syncthreads ();
      /* sum this thread's run; a run never crosses a pad word (run | 32) */
      uint32_t *r = t0 + NTT_PAD (j0);
      const uint32_t *h1 = t1 + NTT_PAD (j0), *h2 = t2 + NTT_PAD (j0);
      uint64_t a01 = 0;
      uint32_t a2 = 0;
      for (int i = 0; i < run; i++)
        {
          const uint64_t x01 = r[i] | ((uint64_t) h1[i] << 32);
          a01 += x01;
          a2 += h2[i] + (a01 < x01);
          r[i] = (uint32_t) a01;
          a01 = (a01 >> 32) | ((uint64_t) a2 << 32);
          a2 = 0;
        }
      /* the carry (< 2^64) goes to the next run up */
      S.xchg[threadIdx.x] = a01;
      __syncthreads ();
      const uint64_t in = threadIdx.x > 0 ? S.xchg[threadIdx.x - 1] : tile_in;
      const uint64_t last = S.xchg[B - 1];
      uint64_t sum = (uint64_t) r[0] + (uint32_t) in;
      r[0] = (uint32_t) sum;
      sum = (sum >> 32) + r[1] + (in >> 32);
      r[1] = (uint32_t) sum;
      uint32_t gen = ntt_run_inc (r + 2, run - 2, (uint32_t) (sum >> 32));
      uint32_t top;
      uint32_t cin = ntt_block_resolve (S, gen, ntt_all_ones (r, run), &top);
      ntt_run_inc (r, run, cin);
      tile_in = last + top;
      __syncthreads ();
      for (int j = threadIdx.x; j < LT; j += B)
        out[base + j] = t0[NTT_PAD (j)];
      __syncthreads ();
    }
  NTT_PROF (S, NTT_PH_CRT);
}

/* ---- multiword operations on W limbs (each thread a run of W/B) ------ */

/* r = a + b (no overflow out of W limbs) */
NTT_DEV void
ntt_mw_add (ntt_shared &S, const ntt_ctx &c, uint32_t *r, const uint32_t *a,
            const uint32_t *b)
{
  const int run = c.W / blockDim.x, j0 = threadIdx.x * run;
  uint64_t cy = 0;
  for (int i = j0; i < j0 + run; i++)
    {
      cy += (uint64_t) a[i] + b[i];
      r[i] = (uint32_t) cy;
      cy >>= 32;
    }
  uint32_t cin = ntt_block_resolve (S, (uint32_t) cy,
                                    ntt_all_ones (r + j0, run), NULL);
  ntt_run_inc (r + j0, run, cin);
  __syncthreads ();
  NTT_PROF (S, NTT_PH_MW);
}

/* r = a - b, plus 2N if negative */
NTT_DEV void
ntt_mw_sub (ntt_shared &S, const ntt_ctx &c, uint32_t *r, const uint32_t *a,
            const uint32_t *b)
{
  const int run = c.W / blockDim.x, j0 = threadIdx.x * run;
  uint32_t bw = 0;
  for (int i = j0; i < j0 + run; i++)
    {
      uint64_t d = (uint64_t) a[i] - b[i] - bw;
      r[i] = (uint32_t) d;
      bw = (uint32_t) (d >> 63);
    }
  uint32_t neg;
  uint32_t bin = ntt_block_resolve (S, bw, ntt_all_zeros (r + j0, run), &neg);
  ntt_run_dec (r + j0, run, bin);
  /* neg is uniform across the block */
  if (neg)
    {
      uint64_t cy = 0;
      for (int i = j0; i < j0 + run; i++)
        {
          cy += (uint64_t) r[i] + c.N2[i];
          r[i] = (uint32_t) cy;
          cy >>= 32;
        }
      uint32_t cin = ntt_block_resolve (S, (uint32_t) cy,
                                        ntt_all_ones (r + j0, run), NULL);
      ntt_run_inc (r + j0, run, cin);
    }
  __syncthreads ();
  NTT_PROF (S, NTT_PH_MW);
}

/* r = 2a */
NTT_DEV void
ntt_mw_dbl (const ntt_ctx &c, uint32_t *r, const uint32_t *a)
{
  const int run = c.W / blockDim.x, j0 = threadIdx.x * run;
  uint32_t v[64];       /* run <= 64 */
  uint32_t below = j0 > 0 ? a[j0 - 1] : 0;
  for (int i = 0; i < run; i++)
    v[i] = a[j0 + i];
  __syncthreads ();
  r[j0] = (v[0] << 1) | (below >> 31);
  for (int i = 1; i < run; i++)
    r[j0 + i] = (v[i] << 1) | (v[i - 1] >> 31);
  __syncthreads ();
}

/* r = a d / 2^32 mod N (lazy: r < a + N) */
NTT_DEV void
ntt_mw_mul_ui (ntt_shared &S, const ntt_ctx &c, ntt_scratch &X, uint32_t *r,
               const uint32_t *a, uint32_t d)
{
  const int W = c.W, run = W / blockDim.x, j0 = threadIdx.x * run;
  uint32_t *t = X.mw;      /* W + 1 limbs */
  if (threadIdx.x == 0)
    S.word = (a[0] * d) * c.np0;
  __syncthreads ();
  const uint32_t m = S.word;
  /* t = a d + m N, as runs with a carry word each */
  uint64_t cy = 0;
  for (int i = j0; i < j0 + run; i++)
    {
      uint64_t x = (uint64_t) a[i] * d, y = (uint64_t) m * c.N[i];
      uint64_t s = x + y;
      uint64_t hi = (s < x) ? (1ull << 32) : 0;
      uint64_t s2 = s + cy;
      hi += (s2 < s) ? (1ull << 32) : 0;
      t[i] = (uint32_t) s2;
      cy = (s2 >> 32) + hi;
    }
  if (threadIdx.x == blockDim.x - 1)
    t[W] = 0;
  S.xchg[threadIdx.x] = cy;
  __syncthreads ();
  /* add the carry from the run below; the top run's goes to t[W] */
  uint64_t in = threadIdx.x > 0 ? S.xchg[threadIdx.x - 1] : 0;
  uint64_t s = (uint64_t) t[j0] + (uint32_t) in;
  t[j0] = (uint32_t) s;
  s = (s >> 32) + t[j0 + 1] + (in >> 32);
  t[j0 + 1] = (uint32_t) s;
  uint32_t gen = ntt_run_inc (t + j0 + 2, run - 2, (uint32_t) (s >> 32));
  uint32_t top;
  uint32_t cin = ntt_block_resolve (S, gen, ntt_all_ones (t + j0, run), &top);
  ntt_run_inc (t + j0, run, cin);
  if (threadIdx.x == blockDim.x - 1)
    t[W] = (uint32_t) S.xchg[blockDim.x - 1] + top;
  __syncthreads ();
  /* r = t / 2^32 */
  for (int i = j0; i < j0 + run; i++)
    r[i] = t[i + 1];
  __syncthreads ();
  NTT_PROF (S, NTT_PH_MW);
}

/* r = a b / R mod N (lazy).  r may alias a or b. */
NTT_DEV void
ntt_mont_mul (uint32_t *sm, ntt_shared &S, const ntt_ctx &c, ntt_scratch &X,
              uint32_t *r, const uint32_t *a, const uint32_t *b)
{
  const int W = c.W;
  const bool square = a == b;
  /* T = a b */
  for (int i = 0; i < NTT_NPRIMES; i++)
    ntt_conv_prime (sm, S, c, c.pr[i], X.ahat, a, W, b, W, NULL, square,
                    X.res[i]);
  ntt_crt_carry (sm, S, c, X.res, X.T, c.L);
  /* q = (T mod R) N' mod R: the low W limbs of the next product */
  for (int i = 0; i < NTT_NPRIMES; i++)
    ntt_conv_prime (sm, S, c, c.pr[i], X.ahat, X.T, W, NULL, 0,
                    c.pr[i].Nphat, false, X.res[i]);
  ntt_crt_carry (sm, S, c, X.res, X.Q, W);
  /* U = q N */
  for (int i = 0; i < NTT_NPRIMES; i++)
    ntt_conv_prime (sm, S, c, c.pr[i], X.ahat, X.Q, W, NULL, 0,
                    c.pr[i].Nhat, false, X.res[i]);
  ntt_crt_carry (sm, S, c, X.res, X.U, c.L);
  /* r = (T + U) / R; T + U = 0 mod R, so the low half carries 1 into the
     high half exactly when T mod R is not 0 */
  const int run = W / blockDim.x, j0 = threadIdx.x * run;
  int nz = !ntt_all_zeros (X.T + j0, run);
  int lowcarry = __syncthreads_or (nz);
  uint64_t cy = 0;
  for (int i = j0; i < j0 + run; i++)
    {
      cy += (uint64_t) X.T[W + i] + X.U[W + i];
      if (i == 0)
        cy += lowcarry;
      r[i] = (uint32_t) cy;
      cy >>= 32;
    }
  uint32_t cin = ntt_block_resolve (S, (uint32_t) cy,
                                    ntt_all_ones (r + j0, run), NULL);
  ntt_run_inc (r + j0, run, cin);
  __syncthreads ();
  NTT_PROF (S, NTT_PH_FINAL);
}

#endif /* __CUDACC__ */

#endif /* _GPU_NTT_CUH */
