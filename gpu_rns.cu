/* gpu_rns.cu: polynomials over Z/N in residue number system form on the GPU
   (see gpu_rns.h and gpu_dev/poly_stage2_design.md).

  Copyright 2026 Greg Childers

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 3 of the License, or (at your
  option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along
  with this program; see the file COPYING.  If not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02111-1307, USA.
*/

/* Small primes p < 2^30, so that lazy values in [0, 2p) and Montgomery
 * products of two of them (< 2p) fit 32 bits, as in gpu_ntt.cuh.  Every
 * bound below needs p < 2^30 (or very nearly), none has room for 31-bit
 * primes:
 * - butterflies: a + b and a - b + 2p of a, b in [0, 2p) are below 4p;
 * - Montgomery products of a, b < 2p: a b + m p < 4p^2 + 2^32 p fits 64
 *   bits, and the result (a b + m p) / 2^32 < p (4p / 2^32 + 1) is below 2p,
 *   only for 4p <= 2^32;
 * - Shoup products: any a < 2^32 gives [0, 2p) (2p <= 2^32 is enough, but
 *   their results go into the two above);
 * - k_from_z_mm, k_to_z_mm (cv_mac): products of a 32-bit word and a
 *   residue are below 2^62, so four of them and a folded part below 2^32
 *   fit 64 bits; with p < 2^31 only two would.  A sum of G such groups is
 *   lo + 2^32 hi with lo < 2^32, hi < 2^32 G;
 * - k_from_z_mm's two Montgomery steps on such a sum give [0, 2p) for
 *   G < p - 2 (G = ceil (NL / 4)); k_to_z_mm keeps hi / 2^32 < G (G =
 *   ceil (P / 4)) in 16 bits, for P < 2^18;
 * - the CRT estimate: t_i floor (2^63 / p_i) < 2^63 for any p.
 * 31-bit primes would take canonical values in [0, p) through every
 * butterfly and product, and a fold every two products, for about 3% fewer
 * primes (67 instead of 69 at 1024 bits).
 *
 * Z/N -> RNS: residue of a coefficient (limbs a_j) mod p is
 * sum_j a_j (2^(32j) mod p), accumulated in 96 bits and reduced once; for
 * many coefficients at once a matrix product (k_from_z_mm, with
 * 2^(32j + 64) mod p, the sum then times 2^-64 by Montgomery steps).
 *
 * RNS -> Z/N (explicit CRT): with M = prod p_i and M_i = M / p_i, the
 * convolution value x < M / 2 is sum_i t_i M_i - alpha M where
 * t_i = y_i M_i^-1 mod p_i and alpha = round (sum_i t_i / p_i) (fixed point,
 * units of 2^-48).  The estimate E = sum_i floor (t_i floor (2^63 / p_i)
 * / 2^15) is at most the exact sum times 2^48, and below it by less than
 * P (2^-32 + 2^-48) in units of 1; the exact sum is alpha + x / M, so
 * floor (E / 2^48 + 1/2) is alpha exactly when 0 <= x < M / 2.  That is
 * all the primes must give: M > 2^(K+1) Bn^2 for inputs below Bn (a
 * convolution value of length up to 2^K is below 2^K Bn^2).  Mod N we
 * compute
 * S = sum_i t_i A_i + alpha B with A_i = M_i 2^64 mod N, B = -M 2^64 mod N,
 * then (S + m N) / 2^64 with m = -S / N mod 2^64: congruent to x, and below
 * N (1 + P 2^30 / 2^64) < 2N.  The sums are again a matrix product
 * (k_to_z_mm), the carries and m one pass per coefficient.
 *
 * Transforms of length 2^lg: up to 2^base (base = min (K, 14)) one block
 * per row in shared memory (radix 8 passes; short rows many to a block),
 * decimation in frequency forward (natural in, bit-reversed out), in time
 * inverse.  Longer: four-step, len = R1 R2 with R2 = 2^13: transforms of
 * length R1 down the columns (stride R2) with the twiddle w^(k1 n2) folded
 * in, transforms of length R2 along the rows.  The scaling of an inverse
 * transform is folded into its last pass.  Twiddles are Montgomery
 * multipliers (one word, times 2^32 mod p), each stage's contiguous; what
 * limits the in-block transforms on an A10 is mostly fetching them.
 *
 * Development switches (environment): GMPECM_RNS_RADIX2 (the radix-2
 * transforms, unfused), GMPECM_RNS_OLDCOLS (the four-step columns by
 * k_ntt_small), GMPECM_RNS_OLDFROMZ, GMPECM_RNS_OLDTOZ (the per-thread
 * conversions), GMPECM_RNS_PROF (time the kernels). */

#include "gpu_rns.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include "cudacommon.h"

/* GMPECM_RNS_PROF set: time each kind of kernel (synchronising), printed
   by gpu_rns_destroy */
enum { RP_FROM_Z, RP_TO_Z, RP_NTT, RP_MUL, RP_ZN, RP_COUNT };
static struct {
  int on = -1;
  double ms[RP_COUNT] = { 0 };
  long calls[RP_COUNT] = { 0 };
  double msl[RP_COUNT][32] = { { 0 } };   /* by log2 of the row length */
  int lg = 0;
  cudaEvent_t a, b;
} rns_prof;

static void
rp_begin ()
{
  if (rns_prof.on < 0)
    {
      rns_prof.on = getenv ("GMPECM_RNS_PROF") != NULL;
      if (rns_prof.on)
        {
          cudaEventCreate (&rns_prof.a);
          cudaEventCreate (&rns_prof.b);
        }
    }
  if (rns_prof.on)
    cudaEventRecord (rns_prof.a);
}

static void
rp_end (int cat)
{
  if (!rns_prof.on)
    return;
  float ms;
  cudaEventRecord (rns_prof.b);
  cudaEventSynchronize (rns_prof.b);
  cudaEventElapsedTime (&ms, rns_prof.a, rns_prof.b);
  rns_prof.ms[cat] += ms;
  rns_prof.calls[cat]++;
  rns_prof.msl[cat][rns_prof.lg & 31] += ms;
}

void
gpu_rns_prof_begin (void)
{
  rp_begin ();
}

void
gpu_rns_prof_end (void)
{
  rp_end (RP_ZN);
}

#define RNS_PLIM (1u << 30)

struct gpu_rns {
  int nl, K, P, base;
  std::vector<uint32_t> p;          /* host copy of the primes */
  uint32_t *d_p, *d_pinv;           /* primes, -1/p mod 2^32 */
  uint32_t *d_ws, *d_wsi;           /* [P][2^base]: entry 2^s + k, k < 2^s,
                                       the twiddle w_b^(+-k 2^(base-1-s)) of
                                       stage s, times 2^32 mod p */
  uint2 *d_tlo, *d_tloi;            /* [P][2^base]: w^j (four-step, K > base) */
  uint2 *d_thi, *d_thii;            /* [P][2^(K-base)]: w^(2^base j) */
  int split, ncol;                  /* four-step lengths 2^(split+c), c <=
                                       ncol: column pass by k_ntt_cols */
  uint32_t *d_tc, *d_tci;           /* [P][ncol][2^split]: entry j of table
                                       c - 1 w_(split+c)^(+-j) (w_l of order
                                       2^l) times 2^32 mod p */
  uint2 *d_sc;                      /* [P][K+1][2]: 2^-lg (2^32 if nmul) */
  uint32_t *d_pow;                  /* [NL][P]: 2^(32 j) mod p */
  uint32_t *d_pow64;                /* [NL][P]: 2^(32 j + 64) mod p */
  uint32_t *d_r64;                  /* [P]: 2^64 mod p */
  uint4 *d_rr;                      /* [P]: 2^32, 2^64 mod p, Shoup pairs */
  uint2 *d_minv;                    /* [P]: M_i^-1 mod p_i (Shoup) */
  unsigned long long *d_ip;         /* [P]: floor (2^63 / p_i) */
  uint32_t *d_A;                    /* [P][NL]: M_i 2^64 mod N */
  uint32_t *d_B;                    /* [NL]: -M 2^64 mod N */
  uint32_t *d_A4, *d_B4, *d_N4;     /* the same, rows of NL4 = NL rounded
                                       up to a multiple of 4 (zeros) */
  uint32_t *d_N;                    /* [NL] */
  unsigned long long ninv;          /* -1/N mod 2^64 */
  uint32_t *d_tmp;                  /* polymul scratch */
  size_t tmp_words;
  std::vector<uint32_t *> d_negmap; /* [K+1]: per length, see negmap () */
  std::vector<void *> dev;
};

/* ---- arithmetic mod p (p < 2^30) ------------------------------------- */

static __device__ __forceinline__ uint32_t
rns_shoup (uint32_t a, uint32_t w, uint32_t wsh, uint32_t p)
{
  return a * w - __umulhi (a, wsh) * p;    /* [0, 2p) for any a < 2^32 */
}

static __device__ __forceinline__ uint32_t
rns_red (uint32_t x, uint32_t m)
{
  return x >= m ? x - m : x;
}

/* a b / 2^32 mod p in [0, 2p), for a b < 4p^2 (a, b < 2p, or a < 4p and
   b < p): a b + m p < 4p^2 + 2^32 p fits 64 bits and the quotient is below
   p (4p / 2^32 + 1) < 2p */
static __device__ __forceinline__ uint32_t
rns_mont (uint32_t a, uint32_t b, uint32_t p, uint32_t pinv)
{
  uint64_t t = (uint64_t) a * b;
  uint32_t m = (uint32_t) t * pinv;
  return (uint32_t) ((t + (uint64_t) m * p) >> 32);
}

/* ---- conversions ------------------------------------------------------ */

#define RNS_CT 16   /* coefficients per thread in k_from_z */

/* One thread per (prime i, RNS_CT coefficients of a row), four
   coefficients at a time (one load of 2^(32j) mod p per four products).
   The 96-bit sum lo + 2^64 hi goes mod p by Shoup products with 2^32 and
   2^64 mod p (RR[i]: those and their Shoup factors).  Entry ONE_AT (>= C)
   is 1 (the leading coefficient of a monic polynomial), unless it is
   ~0. */
__global__ void
k_from_z (uint32_t *res, size_t rowlen, const uint32_t *a, size_t sa,
          size_t C, int B, int nl, int P, const uint32_t *pr,
          const uint32_t *pw, const uint4 *rr, size_t one_at)
{
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= P)
    return;
  const uint32_t p = pr[i], p2 = 2 * p;
  const uint32_t one_sh = (uint32_t) ((1ull << 32) / p);
  const uint4 r = rr[i];
  const size_t c0 = (size_t) blockIdx.x * RNS_CT;
  const size_t c1 = c0 + RNS_CT < rowlen ? c0 + RNS_CT : rowlen;
  for (int b = blockIdx.z; b < B; b += gridDim.z)
  {
  uint32_t *row = res + ((size_t) b * P + i) * rowlen;
  for (size_t c = c0; c < c1; c += 4)
    {
      uint64_t lo[4] = { 0, 0, 0, 0 };
      uint32_t hi[4] = { 0, 0, 0, 0 };
      const uint32_t *ac = a + ((size_t) b * sa + c) * nl;
      if (c + 4 <= C)
        for (int j = 0; j < nl; j++)
          {
            const uint64_t w = pw[(size_t) j * P + i];
#pragma unroll
            for (int q = 0; q < 4; q++)
              {
                const uint64_t t = ac[(size_t) q * nl + j] * w;
                lo[q] += t;
                hi[q] += lo[q] < t;
              }
          }
      else if (c < C)
        for (int j = 0; j < nl; j++)
          {
            const uint64_t w = pw[(size_t) j * P + i];
            for (int q = 0; q < 4 && c + q < C; q++)
              {
                const uint64_t t = ac[(size_t) q * nl + j] * w;
                lo[q] += t;
                hi[q] += lo[q] < t;
              }
          }
      for (int q = 0; q < 4 && c + q < c1; q++)
        {
          uint32_t v = c + q == one_at;
          if (c + q < C)
            {
              const uint32_t x = rns_red (rns_shoup ((uint32_t) (lo[q] >> 32),
                                                     r.x, r.y, p)
                                          + rns_shoup (hi[q], r.z, r.w, p),
                                          p2);
              const uint32_t y = rns_shoup ((uint32_t) lo[q], 1, one_sh, p);
              v = rns_red (rns_red (x + y, p2), p);
            }
          row[c + q] = v;
        }
    }
  }
}

/* The conversions' matrix products, [rows x K] times [K x columns] through
   shared memory CV_TK at a time: CV_TM rows by TN = 8 RN columns per block
   of 32 x 8 threads, thread (tx, ty) rows tx + 32 q (q < 2) and columns
   ty + 8 r (r < RN), so a warp reads 32 consecutive words of the row tile
   and one 16-byte word (broadcast) of the column tile, laid out
   [k][ty][r] (r padded to RNP, a multiple of 4).  Products of a 32-bit
   word and a residue are below 2^62: four fit a 64-bit sum, which is
   folded into a 64-bit high part after every four (and after the last of
   the tile's KN), so at the end lo < 2^32 and the sum is lo + 2^32 hi
   (see the top for hi).  CV_TK = 32: one tile of 36 is no faster at
   NL = 33, and larger ones are slower. */
#define CV_TK 32
#define CV_TM 64
#define CV_QM 2

#define CV_RNP(RN) (((RN) + 3) & ~3)

template<int RN>
static __device__ __forceinline__ void
cv_mac (const uint32_t (*As)[CV_TM + 1],
        const uint32_t (*Ws)[8][CV_RNP (RN)], int kn, int tx, int ty,
        uint64_t (&lo)[CV_QM][RN], uint64_t (&hi)[CV_QM][RN])
{
  for (int k0 = 0; k0 < kn; k0 += 4)
    {
#pragma unroll
      for (int k = k0; k < k0 + 4; k++)
        if (k == k0 || k < kn)
        {
          uint32_t av[CV_QM], wv[CV_RNP (RN)];
#pragma unroll
          for (int q = 0; q < CV_QM; q++)
            av[q] = As[k][tx + 32 * q];
#pragma unroll
          for (int r = 0; r < CV_RNP (RN); r += 4)
            {
              const uint4 w4 = *(const uint4 *) &Ws[k][ty][r];
              wv[r] = w4.x;
              wv[r + 1] = w4.y;
              wv[r + 2] = w4.z;
              wv[r + 3] = w4.w;
            }
#pragma unroll
          for (int q = 0; q < CV_QM; q++)
#pragma unroll
            for (int r = 0; r < RN; r++)
              lo[q][r] += (uint64_t) av[q] * wv[r];
        }
#pragma unroll
      for (int q = 0; q < CV_QM; q++)
#pragma unroll
        for (int r = 0; r < RN; r++)
          {
            hi[q][r] += lo[q][r] >> 32;
            lo[q][r] &= 0xffffffffu;
          }
    }
}

/* Z/N -> RNS as a matrix product: [coefficients x NL limbs] times
   [NL x P] (2^(32 j + 64) mod p_i), CV_TM coefficients by 8 RN primes
   per block (cv_mac), then V = lo + 2^32 hi (lo < 2^32, hi < 2^32 G with
   G = ceil (NL / 4) groups of four products, each folded value below 2^32)
   times 2^-64 mod p by two Montgomery steps: u = hi + (lo + m p) / 2^32
   < 2^32 G + p + 1, then (u_lo + m' p) / 2^32 + u_hi < p + 1 + G + 1, in
   [0, 2p) for G < p - 2.  TOTAL = C B < 2^32 coefficients, coefficient x = b C + c
   (b, c from one 32-bit division per block row, kept in shared memory).
   Only entries c < C; the rest of each row by k_from_z_pad. */
template<int RN>
__global__ void __launch_bounds__ (256, 2)
k_from_z_mm (uint32_t *res, size_t rowlen, const uint32_t *a, size_t sa,
             uint32_t C, uint32_t total, int nl, int P, const uint32_t *pr,
             const uint32_t *pinv, const uint32_t *pw)
{
  constexpr int TN = 8 * RN;
  __shared__ uint32_t As[CV_TK][CV_TM + 1];
  __shared__ __align__ (16) uint32_t Ws[CV_TK][8][CV_RNP (RN)];
  __shared__ size_t aoff[CV_TM], roff[CV_TM];
  __shared__ uint32_t cp[TN], cpv[TN];
  const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
  const uint32_t x0 = blockIdx.x * CV_TM;
  const int nv = (int) min ((uint32_t) CV_TM, total - x0);
  const int i0 = blockIdx.y * TN;
  if (tid < nv)
    {
      const uint32_t x = x0 + tid, b = x / C, c = x - b * C;
      aoff[tid] = ((size_t) b * sa + c) * nl;
      roff[tid] = (size_t) b * P * rowlen + c;
    }
  else if (tid >= CV_TM && tid < CV_TM + TN && i0 + tid - CV_TM < P)
    {
      const int n = tid - CV_TM, i = i0 + n;
      cp[n] = pr[i];
      cpv[n] = pinv[i];
    }
  uint64_t lo[CV_QM][RN], hi[CV_QM][RN];
#pragma unroll
  for (int q = 0; q < CV_QM; q++)
#pragma unroll
    for (int r = 0; r < RN; r++)
      lo[q][r] = hi[q][r] = 0;
  for (int j0 = 0; j0 < nl; j0 += CV_TK)
    {
      __syncthreads ();
      /* As[k][m] = limb j0 + k of coefficient x0 + m (a warp reads one
         coefficient's limbs), Ws[k][n] = 2^(32 (j0 + k) + 64) mod
         p_(i0 + n) */
#pragma unroll
      for (int u = 0; u < CV_TK * CV_TM / 256; u++)
        {
          const int e = tid + 256 * u, m = e / CV_TK, k = e % CV_TK;
          As[k][m] = m < nv && j0 + k < nl ? a[aoff[m] + j0 + k] : 0;
        }
#pragma unroll
      for (int u = 0; u < (CV_TK * TN + 255) / 256; u++)
        {
          const int e = tid + 256 * u, k = e / TN, n = e % TN;
          if (CV_TK * TN % 256 == 0 || e < CV_TK * TN)
            Ws[k][n % 8][n / 8] = j0 + k < nl && i0 + n < P
                                  ? pw[(size_t) (j0 + k) * P + i0 + n] : 0;
        }
      __syncthreads ();
      cv_mac<RN> (As, Ws, min (CV_TK, nl - j0), tx, ty, lo, hi);
    }
#pragma unroll
  for (int r = 0; r < RN; r++)
    {
      const int n = ty + 8 * r;
      if (i0 + n >= P)
        continue;
      const uint32_t p = cp[n], pv = cpv[n];
      uint32_t *row = res + (size_t) (i0 + n) * rowlen;
#pragma unroll
      for (int q = 0; q < CV_QM; q++)
        {
          const int m = tx + 32 * q;
          if (m >= nv)
            continue;
          const uint32_t l = (uint32_t) lo[q][r];
          const uint64_t u = hi[q][r]
                             + (((uint64_t) (l * pv) * p + l) >> 32);
          const uint32_t ul = (uint32_t) u;
          const uint32_t y = (uint32_t) (u >> 32)
                             + (uint32_t) (((uint64_t) (ul * pv) * p + ul)
                                           >> 32);
          row[roff[m]] = rns_red (y, p);
        }
    }
}

/* entries C .. ROWLEN-1 of the B P rows: 0, or 1 at ONE_AT */
__global__ void
k_from_z_pad (uint32_t *res, size_t rowlen, size_t C, size_t rows,
              size_t one_at)
{
  const size_t w = rowlen - C;
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
       x < rows * w; x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t c = C + x % w;
      res[(x / w) * rowlen + c] = c == one_at;
    }
}

/* one block per (coefficient, polynomial); shared: col_lo[NL] (u64),
   col_hi[NL], s[NL + 3], t[P] */
__global__ void
k_to_z (uint32_t *a, size_t sc, const uint32_t *res, size_t rowlen, int B,
        int nl, int P, const uint32_t *pr, const uint2 *minv,
        const unsigned long long *ip, const uint32_t *A, const uint32_t *Bm,
        const uint32_t *N, unsigned long long ninv, int sub1)
{
  extern __shared__ unsigned long long shq[];
  __shared__ unsigned long long asum;
  unsigned long long *clo = shq;
  uint32_t *chi = (uint32_t *) (clo + nl);
  uint32_t *s = chi + nl;
  uint32_t *t = s + nl + 3;
  const size_t c = blockIdx.x;
  for (int b = blockIdx.y; b < B; b += gridDim.y)
  {
  if (threadIdx.x == 0)
    asum = 0;
  __syncthreads ();
  unsigned long long part = 0;
  for (int i = threadIdx.x; i < P; i += blockDim.x)
    {
      const uint32_t p = pr[i];
      uint32_t y = res[((size_t) b * P + i) * rowlen + c];
      if (sub1 && c == 0)
        y = y == 0 ? p - 1 : y - 1;
      const uint32_t ti = rns_red (rns_shoup (y, minv[i].x, minv[i].y, p), p);
      t[i] = ti;
      part += ((unsigned long long) ti * ip[i]) >> 15;
    }
  atomicAdd (&asum, part);
  __syncthreads ();
  const uint32_t alpha = (uint32_t) ((asum + (1ull << 47)) >> 48);

  for (int w = threadIdx.x; w < nl; w += blockDim.x)
    {
      uint64_t lo = (uint64_t) alpha * Bm[w];
      uint32_t hi = 0;
      for (int i = 0; i < P; i++)
        {
          const uint64_t m = (uint64_t) t[i] * A[(size_t) i * nl + w];
          lo += m;
          hi += lo < m;
        }
      clo[w] = lo;
      chi[w] = hi;
    }
  __syncthreads ();

  if (threadIdx.x == 0)
    {
      /* S into s[0 .. nl+2] */
      uint64_t carry = 0;
      for (int w = 0; w < nl + 3; w++)
        {
          const uint64_t cl = w < nl ? clo[w] : 0;
          const uint32_t ch = w < nl ? chi[w] : 0;
          const uint64_t vl = cl + carry;
          const uint32_t vh = ch + (vl < carry);
          s[w] = (uint32_t) vl;
          carry = (vl >> 32) | ((uint64_t) vh << 32);
        }
      /* (S + m N) / 2^64 */
      const uint64_t m = ((uint64_t) s[0] | ((uint64_t) s[1] << 32)) * ninv;
      for (int h = 0; h < 2; h++)
        {
          const uint32_t mh = (uint32_t) (m >> (32 * h));
          uint64_t cy = 0;
          for (int j = 0; j < nl; j++)
            {
              const uint64_t v = (uint64_t) mh * N[j] + s[h + j] + cy;
              s[h + j] = (uint32_t) v;
              cy = v >> 32;
            }
          for (int j = h + nl; j < nl + 3 && cy; j++)
            {
              const uint64_t v = (uint64_t) s[j] + cy;
              s[j] = (uint32_t) v;
              cy = v >> 32;
            }
        }
      uint32_t *out = a + ((size_t) b * sc + c) * nl;
      for (int j = 0; j < nl; j++)
        out[j] = s[j + 2];
    }
  __syncthreads ();
  }
}

/* The same, one thread per (coefficient, polynomial) (blocks of TB
   threads; shared: t[P][TB]): the columns of S + m N are summed four limbs
   at a time (one 16-byte load of A per residue), low to high, with the
   carry folded in, so the digits come out in order.  m = -S / N mod 2^64
   only needs the two low digits of S: columns 0 and 1 are kept, m
   computed, then added in.  A4, B4, N4: rows of NL4 limbs.  SUB1: minus 1
   at coefficient 0 (see gpu_rns_conv_monic). */
__global__ void
k_to_z_t (uint32_t *a, size_t sc, const uint32_t *res, size_t rowlen,
          size_t C, int B, int nl, int P, const uint32_t *pr,
          const uint2 *minv, const unsigned long long *ip, const uint32_t *A4,
          const uint32_t *B4, const uint32_t *N4, unsigned long long ninv,
          int sub1)
{
  extern __shared__ uint32_t tsh[];
  const int T = blockDim.x, tid = threadIdx.x;
  const int nl4 = (nl + 3) & ~3;
  const size_t total = (size_t) B * C;
  const size_t x = (size_t) blockIdx.x * T + tid;
  if (x >= total)
    return;
  const size_t b = x / C, c = x % C;
  unsigned long long asum = 0;
  for (int i = 0; i < P; i++)
    {
      const uint32_t p = pr[i];
      uint32_t y = res[((size_t) b * P + i) * rowlen + c];
      if (sub1 && c == 0)
        y = y == 0 ? p - 1 : y - 1;
      const uint32_t ti = rns_red (rns_shoup (y, minv[i].x, minv[i].y, p), p);
      tsh[i * T + tid] = ti;
      asum += ((unsigned long long) ti * ip[i]) >> 15;
    }
  const uint32_t alpha = (uint32_t) ((asum + (1ull << 47)) >> 48);
  uint32_t *out = a + ((size_t) b * sc + c) * nl;

  uint64_t l0 = 0, carry = 0;
  uint32_t h0 = 0, mlo = 0, mhi = 0;
  for (int w0 = 0; w0 < nl + 2; w0 += 4)
    {
      uint64_t lo[4];
      uint32_t hi[4];
      if (w0 < nl4)
        {
          const uint4 bb = *(const uint4 *) (B4 + w0);
          lo[0] = (uint64_t) alpha * bb.x;
          lo[1] = (uint64_t) alpha * bb.y;
          lo[2] = (uint64_t) alpha * bb.z;
          lo[3] = (uint64_t) alpha * bb.w;
          hi[0] = hi[1] = hi[2] = hi[3] = 0;
          for (int i = 0; i < P; i++)
            {
              const uint64_t t = tsh[i * T + tid];
              const uint4 aa = *(const uint4 *) (A4 + (size_t) i * nl4 + w0);
              uint64_t m;
              m = t * aa.x; lo[0] += m; hi[0] += lo[0] < m;
              m = t * aa.y; lo[1] += m; hi[1] += lo[1] < m;
              m = t * aa.z; lo[2] += m; hi[2] += lo[2] < m;
              m = t * aa.w; lo[3] += m; hi[3] += lo[3] < m;
            }
        }
      else
        for (int q = 0; q < 4; q++)
          lo[q] = 0, hi[q] = 0;
#pragma unroll
      for (int q = 0; q < 4; q++)
        {
          const int w = w0 + q;
          if (w >= nl + 2)
            break;
          uint64_t v = lo[q];
          uint32_t vh = hi[q];
          if (w == 0)
            {
              l0 = v;
              h0 = vh;
              continue;
            }
          if (w == 1)
            {
              /* the two low digits of S, then m, then columns 0 and 1 of
                 S + m N (whose digits are 0) */
              const uint64_t cy = (l0 >> 32) | ((uint64_t) h0 << 32);
              const uint64_t slo = (uint32_t) l0
                                   | ((uint64_t) (uint32_t) (v + cy) << 32);
              const uint64_t m = slo * ninv;
              mlo = (uint32_t) m;
              mhi = (uint32_t) (m >> 32);
              uint64_t t = (uint64_t) mlo * N4[0];
              l0 += t;
              h0 += l0 < t;
              carry = (l0 >> 32) | ((uint64_t) h0 << 32);
              t = (uint64_t) mlo * N4[1];
              v += t;
              vh += v < t;
              t = (uint64_t) mhi * N4[0];
              v += t;
              vh += v < t;
              v += carry;
              vh += v < carry;
              carry = (v >> 32) | ((uint64_t) vh << 32);
              continue;
            }
          uint64_t t;
          if (w < nl4)
            {
              t = (uint64_t) mlo * N4[w];
              v += t;
              vh += v < t;
            }
          if (w - 1 < nl4)
            {
              t = (uint64_t) mhi * N4[w - 1];
              v += t;
              vh += v < t;
            }
          v += carry;
          vh += v < carry;
          out[w - 2] = (uint32_t) v;
          carry = (v >> 32) | ((uint64_t) vh << 32);
        }
    }
}

/* The same as a matrix product: [coefficients x P] (t) times [P x NL4]
   (A4), CV_TM coefficients by TN = 8 RN limb columns at a time (cv_mac),
   the t_i of CV_TK primes at a time made from the residues in the block
   (and alpha with the first columns).  After each block of columns, one
   thread per coefficient adds them, low to high, into its digits with the
   carry, alpha B, m N (m from columns 0 and 1) as k_to_z_t does; the
   digits go out through shared memory (coalesced).  TOTAL = C B < 2^32
   coefficients. */
template<int RN>
__global__ void __launch_bounds__ (256, RN <= 5 ? 3 : 2)
k_to_z_mm (uint32_t *a, size_t sc, const uint32_t *res, size_t rowlen,
           uint32_t C, uint32_t total, int nl, int P, const uint32_t *pr,
           const uint2 *minv, const unsigned long long *ip,
           const uint32_t *A4, const uint32_t *B4, const uint32_t *N4,
           unsigned long long ninv, int sub1)
{
  constexpr int TN = 8 * RN, RNP = CV_RNP (RN);
  /* the products' tiles Ts, Ws, then the column sums lo + 2^32 (Cm + 2^32
     Ch) (Ch < 2^16: P < 2^18 four-term groups, each below 2^32) */
  constexpr int MW = CV_TK * (CV_TM + 1) + CV_TK * 8 * RNP;
  constexpr int DW = 2 * CV_TM * (TN + 1) + (CV_TM * (TN + 1) + 1) / 2;
  __shared__ __align__ (16) uint32_t sh[MW > DW ? MW : DW];
  __shared__ size_t roff[CV_TM], ooff[CV_TM];
  __shared__ unsigned long long asum[4][CV_TM];
  __shared__ bool cz[CV_TM];
  uint32_t (*Ts)[CV_TM + 1] = (uint32_t (*)[CV_TM + 1]) sh;
  uint32_t (*Ws)[8][RNP] = (uint32_t (*)[8][RNP]) (sh + CV_TK * (CV_TM + 1));
  uint32_t (*Cl)[TN + 1] = (uint32_t (*)[TN + 1]) sh;
  uint32_t (*Cm)[TN + 1] = Cl + CV_TM;
  uint16_t (*Ch)[TN + 1] = (uint16_t (*)[TN + 1]) (Cm + CV_TM);
  const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
  const int nl4 = (nl + 3) & ~3;
  const uint32_t x0 = blockIdx.x * CV_TM;
  const int nv = (int) min ((uint32_t) CV_TM, total - x0);
  if (tid < nv)
    {
      const uint32_t x = x0 + tid, b = x / C, c = x - b * C;
      roff[tid] = (size_t) b * P * rowlen + c;
      ooff[tid] = ((size_t) b * sc + c) * nl;
      cz[tid] = sub1 && c == 0;
    }
  /* T tiles: thread does coefficient tm, primes tk + 4 u */
  const int tm = tid % CV_TM, tk = tid / CV_TM;
  unsigned long long apart = 0;
  /* the digit pass: thread tid < nv does coefficient tid */
  uint32_t alpha = 0, mlo = 0, mhi = 0;
  uint64_t carry = 0, l0 = 0, h0 = 0;
  for (int w0 = 0; w0 < nl + 2; w0 += TN)
    {
      uint64_t lo[CV_QM][RN], hi[CV_QM][RN];
#pragma unroll
      for (int q = 0; q < CV_QM; q++)
#pragma unroll
        for (int r = 0; r < RN; r++)
          lo[q][r] = hi[q][r] = 0;
      if (w0 < nl4)
        for (int i0 = 0; i0 < P; i0 += CV_TK)
          {
            __syncthreads ();
#pragma unroll
            for (int u = 0; u < CV_TK / 4; u++)
              {
                const int k = tk + 4 * u, i = i0 + k;
                uint32_t t = 0;
                if (tm < nv && i < P)
                  {
                    const uint32_t p = pr[i];
                    uint32_t y = res[roff[tm] + (size_t) i * rowlen];
                    if (cz[tm])
                      y = y == 0 ? p - 1 : y - 1;
                    const uint2 mi = minv[i];
                    t = rns_red (rns_shoup (y, mi.x, mi.y, p), p);
                    if (w0 == 0)
                      apart += ((unsigned long long) t * ip[i]) >> 15;
                  }
                Ts[k][tm] = t;
              }
#pragma unroll
            for (int u = 0; u < (CV_TK * TN + 255) / 256; u++)
              {
                const int e = tid + 256 * u, k = e / TN, n = e % TN;
                if (CV_TK * TN % 256 == 0 || e < CV_TK * TN)
                  Ws[k][n % 8][n / 8] = i0 + k < P && w0 + n < nl4
                                        ? A4[(size_t) (i0 + k) * nl4 + w0 + n]
                                        : 0;
              }
            __syncthreads ();
            cv_mac<RN> (Ts, Ws, min (CV_TK, P - i0), tx, ty, lo, hi);
          }
      __syncthreads ();
      if (w0 == 0)
        asum[tk][tm] = apart;
#pragma unroll
      for (int q = 0; q < CV_QM; q++)
#pragma unroll
        for (int r = 0; r < RN; r++)
          {
            Cl[tx + 32 * q][ty + 8 * r] = (uint32_t) lo[q][r];
            Cm[tx + 32 * q][ty + 8 * r] = (uint32_t) hi[q][r];
            Ch[tx + 32 * q][ty + 8 * r] = (uint16_t) (hi[q][r] >> 32);
          }
      __syncthreads ();
      if (tid < nv)
        {
          if (w0 == 0)
            alpha = (uint32_t) ((asum[0][tid] + asum[1][tid] + asum[2][tid]
                                 + asum[3][tid] + (1ull << 47)) >> 48);
#pragma unroll
          for (int n = 0; n < TN; n++)
            {
              const int w = w0 + n;
              if (w >= nl + 2)
                break;
              /* column w of S plus alpha B[w], as lo + 2^64 hi */
              uint64_t v = Cl[tid][n] | ((uint64_t) Cm[tid][n] << 32);
              uint64_t vh = Ch[tid][n];
              uint64_t t;
              if (w < nl4)
                {
                  t = (uint64_t) alpha * B4[w];
                  v += t;
                  vh += v < t;
                }
              if (w == 0)
                {
                  l0 = v;
                  h0 = vh;
                  continue;
                }
              if (w == 1)
                {
                  const uint64_t cy = (l0 >> 32) | (h0 << 32);
                  const uint64_t slo = (uint32_t) l0
                                       | ((uint64_t) (uint32_t) (v + cy)
                                          << 32);
                  const uint64_t m = slo * ninv;
                  mlo = (uint32_t) m;
                  mhi = (uint32_t) (m >> 32);
                  t = (uint64_t) mlo * N4[0];
                  l0 += t;
                  h0 += l0 < t;
                  carry = (l0 >> 32) | (h0 << 32);
                  t = (uint64_t) mlo * N4[1];
                  v += t;
                  vh += v < t;
                  t = (uint64_t) mhi * N4[0];
                  v += t;
                  vh += v < t;
                  v += carry;
                  vh += v < carry;
                  carry = (v >> 32) | (vh << 32);
                  continue;
                }
              if (w < nl4)
                {
                  t = (uint64_t) mlo * N4[w];
                  v += t;
                  vh += v < t;
                }
              if (w - 1 < nl4)
                {
                  t = (uint64_t) mhi * N4[w - 1];
                  v += t;
                  vh += v < t;
                }
              v += carry;
              vh += v < carry;
              Cl[tid][n] = (uint32_t) v;           /* digit w - 2 */
              carry = (v >> 32) | (vh << 32);
            }
        }
      __syncthreads ();
      /* digits w - 2 of the columns w0 <= w < w0 + TN */
      const int n0 = max (0, 2 - w0), n1 = min (TN, nl + 2 - w0);
      for (int m = ty; m < nv; m += 8)
        for (int n = n0 + tx; n < n1; n += 32)
          a[ooff[m] + w0 + n - 2] = Cl[m][n];
      __syncthreads ();
    }
}

/* ---- transforms ------------------------------------------------------- */

/* Transform sub-sequences of length 2^mlg (<= 2^base) of each row: number
   q = blockIdx.x has its elements at q * qstep + e * stride.  The twiddle of
   stage s (half size 2^s) for index k is w_b^(k 2^(base - 1 - s)), entry
   2^s + k of the prime's row of WTAB (Montgomery: times 2^32), so that a
   stage's twiddles are contiguous (consecutive threads load consecutive
   words, and short transforms only touch the start of the table). */
__global__ void
k_ntt_sub (uint32_t *res, size_t rowlen, int rows, int P, int mlg, size_t qstep,
           size_t stride, int base, const uint32_t *pr, const uint32_t *pinv,
           const uint32_t *wtab, int inv)
{
  extern __shared__ uint32_t x[];
  const int m = 1 << mlg;
  for (int row = blockIdx.y; row < rows; row += gridDim.y)
  {
  const int i = row % P;
  const uint32_t p = pr[i], p2 = 2 * p, pv = pinv[i];
  const uint32_t *w = wtab + ((size_t) i << base);
  uint32_t *v = res + (size_t) row * rowlen + (size_t) blockIdx.x * qstep;

  for (int e = threadIdx.x; e < m; e += blockDim.x)
    x[e] = v[(size_t) e * stride];
  __syncthreads ();
  if (!inv)
    for (int s = mlg - 1; s >= 0; s--)
      {
        const int h = 1 << s;
        for (int j = threadIdx.x; j < m / 2; j += blockDim.x)
          {
            const int k = j & (h - 1);
            const int i1 = ((j >> s) << (s + 1)) + k, i2 = i1 + h;
            const uint32_t ww = w[h | k];
            const uint32_t a = x[i1], bb = x[i2];
            x[i1] = rns_red (a + bb, p2);
            x[i2] = rns_mont (a - bb + p2, ww, p, pv);
          }
        __syncthreads ();
      }
  else
    for (int s = 0; s < mlg; s++)
      {
        const int h = 1 << s;
        for (int j = threadIdx.x; j < m / 2; j += blockDim.x)
          {
            const int k = j & (h - 1);
            const int i1 = ((j >> s) << (s + 1)) + k, i2 = i1 + h;
            const uint32_t ww = w[h | k];
            const uint32_t a = rns_red (x[i1], p2);
            const uint32_t t = rns_mont (x[i2], ww, p, pv);
            x[i1] = rns_red (a + t, p2);
            x[i2] = rns_red (a - t + p2, p2);
          }
        __syncthreads ();
      }
  for (int e = threadIdx.x; e < m; e += blockDim.x)
    v[(size_t) e * stride] = x[e];
  __syncthreads ();
  }
}

/* Work folded into a transform pass: the four-step twiddle (TLO != NULL;
   forward after the column transforms, inverse before them) and the final
   scaling of an inverse transform (SC != NULL; at the store). */
struct ntt_fuse {
  const uint2 *tlo, *thi;
  int lg, r1lg;
  const uint2 *sc;
  int nmul;
  int K;
};

/* entry (t1, n2) of a four-step row (t1 the column transform's index) times
   w^(+-bitrev (t1) n2) */
static __device__ __forceinline__ uint32_t
ntt_tw (uint32_t v, uint32_t t1, uint32_t n2, const ntt_fuse &f, int base,
        int i, uint32_t p)
{
  const uint32_t k1 = __brev (t1) >> (32 - f.r1lg);
  const uint64_t ex = ((uint64_t) k1 * n2) << (f.K - f.lg);
  const uint2 a = f.tlo[((size_t) i << base) + (ex & ((1u << base) - 1))];
  const uint2 b = f.thi[((size_t) i << (f.K - base)) + (ex >> base)];
  const uint32_t y = rns_shoup (v, a.x, a.y, p);
  return rns_shoup (y, b.x, b.y, p);
}

static __device__ __forceinline__ uint32_t
ntt_scale (uint32_t v, const ntt_fuse &f, int i, uint32_t p)
{
  const uint2 c = f.sc[((size_t) i * (f.K + 1) + f.lg) * 2 + (f.nmul ? 1 : 0)];
  return rns_red (rns_shoup (v, c.x, c.y, p), p);
}

/* The same transforms when they are short (2 <= m <= 32, and the columns
   of four-step transforms k_ntt_cols does not do): one thread per
   butterfly and SPB segments per block (one block per segment would leave
   most threads idle).  Segment g is (row g / NSUB, sub g % NSUB).
   Contiguous segments (STRIDE 1): a segment's threads are consecutive, so
   they load consecutive words.  Strided ones (the columns of the four-step
   transform, consecutive subs adjacent in memory): consecutive threads
   take consecutive segments instead, element e of segment ls at
   x[e SPB + ls]. */
__global__ void
k_ntt_small (uint32_t *res, size_t rowlen, int rows, int P, int mlg,
             size_t nsub, size_t qstep, size_t stride, int base,
             const uint32_t *pr, const uint32_t *pinv, const uint32_t *wtab,
             int inv, ntt_fuse f)
{
  extern __shared__ uint32_t xs[];
  const int m = 1 << mlg, half = m >> 1;
  const int spb = blockDim.x / half;
  const bool col = stride != 1;
  const int ls = col ? threadIdx.x % spb : threadIdx.x / half;
  const int j = col ? threadIdx.x / spb : threadIdx.x % half;
  const int es = col ? spb : 1;               /* x[e es + eo] */
  const int eo = col ? ls : ls * m;
  const size_t nseg = nsub * (size_t) rows;
  for (size_t g0 = (size_t) blockIdx.x * spb; g0 < nseg;
       g0 += (size_t) gridDim.x * spb)
    {
      const size_t g = g0 + ls;
      const bool ok = g < nseg;
      const size_t row = ok ? g / nsub : 0, sub = ok ? g % nsub : 0;
      const int i = (int) (row % P);
      const uint32_t p = pr[i], p2 = 2 * p, pv = pinv[i];
      const uint32_t *w = wtab + ((size_t) i << base);
      uint32_t *v = res + row * rowlen + sub * qstep;
      if (ok)
        {
          uint32_t y0 = v[(size_t) j * stride];
          uint32_t y1 = v[(size_t) (j + half) * stride];
          if (f.tlo != NULL && inv)
            {
              y0 = ntt_tw (y0, j, (uint32_t) sub, f, base, i, p);
              y1 = ntt_tw (y1, j + half, (uint32_t) sub, f, base, i, p);
            }
          xs[j * es + eo] = y0;
          xs[(j + half) * es + eo] = y1;
        }
      __syncthreads ();
      if (!inv)
        for (int s = mlg - 1; s >= 0; s--)
          {
            const int h = 1 << s, k = j & (h - 1);
            const int i1 = ((j >> s) << (s + 1)) + k, i2 = i1 + h;
            const uint32_t ww = w[h | k];
            const uint32_t a = xs[i1 * es + eo], bb = xs[i2 * es + eo];
            xs[i1 * es + eo] = rns_red (a + bb, p2);
            xs[i2 * es + eo] = rns_mont (a - bb + p2, ww, p, pv);
            __syncthreads ();
          }
      else
        for (int s = 0; s < mlg; s++)
          {
            const int h = 1 << s, k = j & (h - 1);
            const int i1 = ((j >> s) << (s + 1)) + k, i2 = i1 + h;
            const uint32_t ww = w[h | k];
            const uint32_t a = rns_red (xs[i1 * es + eo], p2);
            const uint32_t t = rns_mont (xs[i2 * es + eo], ww, p, pv);
            xs[i1 * es + eo] = rns_red (a + t, p2);
            xs[i2 * es + eo] = rns_red (a - t + p2, p2);
            __syncthreads ();
          }
      if (ok)
        {
          uint32_t y0 = xs[j * es + eo], y1 = xs[(j + half) * es + eo];
          if (f.tlo != NULL && !inv)
            {
              y0 = ntt_tw (y0, j, (uint32_t) sub, f, base, i, p);
              y1 = ntt_tw (y1, j + half, (uint32_t) sub, f, base, i, p);
            }
          if (f.sc != NULL)
            {
              y0 = ntt_scale (y0, f, i, p);
              y1 = ntt_scale (y1, f, i, p);
            }
          v[(size_t) j * stride] = y0;
          v[(size_t) (j + half) * stride] = y1;
        }
      __syncthreads ();
    }
}

/* The same transforms for 2^6 <= m <= 2^base, one segment per block (short
   ones several), R stages per pass (R = 3, the last pass of a transform
   fewer): each thread takes 2^R elements into registers, does the R
   radix-2 stages on them exactly as k_ntt_sub would (same butterflies,
   twiddles and lazy reductions), and writes them back, so a pass is one
   round trip through shared memory and one barrier.  Shared memory is not
   padded but swizzled (word e at e xor bits 5..9 of e: bank conflicts as
   with one word of padding per 32, and three rows of 2^13 fit an SM).
   Forward (decimation in frequency), stages s, s-1, ..., s-R+1: the
   elements b + t 2^(s-R+1), t < 2^R, for b with bits s-R+1 .. s clear.
   Inverse (in time), stages s .. s+R-1: the elements b + t 2^s.  The 2^r
   twiddles of a group at stage lo + r are W^bl times the 2^(r+1)-th roots
   of unity (W^bl from the table, the roots ZC: stage 2's twiddles), one
   load instead of 2^r. */
#define NTT_PADI(e) ((e) ^ (((e) >> 5) & 31))

template<int R>
__device__ __forceinline__ void
ntt_rad_pass (uint32_t *x, int m, int s, uint32_t p, uint32_t pv,
              const uint32_t * __restrict__ w, const uint32_t (&zc)[4],
              int inv)
{
  const int T = 1 << R, groups = m >> R;
  for (int gi = threadIdx.x; gi < groups; gi += blockDim.x)
    {
      /* b: the group's base index, its R bits at the pass's stages clear */
      const int lo = inv ? s : s - R + 1;
      const int b = ((gi >> lo) << (lo + R)) | (gi & ((1 << lo) - 1));
      const int st = 1 << lo, bl = b & (st - 1);
      uint32_t v[T];
#pragma unroll
      for (int t = 0; t < T; t++)
        v[t] = x[NTT_PADI (b + t * st)];
      const uint32_t p2 = 2 * p;
      /* at stage lo + r the twiddle index is (b + t st) mod 2^(lo + r)
         = bl + (t mod 2^r) st: 2^r distinct ones */
      if (!inv)
#pragma unroll
        for (int r = R - 1; r >= 0; r--)
          {
            const int sp = lo + r, h = 1 << r;
            uint32_t ww[1 << (R - 1)];
            ww[0] = __ldg (&w[(1 << sp) | bl]);
#pragma unroll
            for (int u = 1; u < h; u++)
              ww[u] = rns_red (rns_mont (ww[0], zc[u << (2 - r)], p, pv), p);
#pragma unroll
            for (int t = 0; t < T; t++)
              if (!(t & h))
                {
                  const uint32_t wt = ww[t & (h - 1)];
                  const uint32_t a = v[t], bb = v[t + h];
                  v[t] = rns_red (a + bb, p2);
                  v[t + h] = rns_mont (a - bb + p2, wt, p, pv);
                }
          }
      else
#pragma unroll
        for (int r = 0; r < R; r++)
          {
            const int sp = lo + r, h = 1 << r;
            uint32_t ww[1 << (R - 1)];
            ww[0] = __ldg (&w[(1 << sp) | bl]);
#pragma unroll
            for (int u = 1; u < h; u++)
              ww[u] = rns_red (rns_mont (ww[0], zc[u << (2 - r)], p, pv), p);
#pragma unroll
            for (int t = 0; t < T; t++)
              if (!(t & h))
                {
                  const uint32_t wt = ww[t & (h - 1)];
                  const uint32_t a = rns_red (v[t], p2);
                  const uint32_t tt = rns_mont (v[t + h], wt, p, pv);
                  v[t] = rns_red (a + tt, p2);
                  v[t + h] = rns_red (a - tt + p2, p2);
                }
          }
#pragma unroll
      for (int t = 0; t < T; t++)
        x[NTT_PADI (b + t * st)] = v[t];
    }
}

__global__ void
k_ntt_rad (uint32_t *res, size_t rowlen, int rows, int P, int mlg,
           size_t qstep, size_t stride, int base, const uint32_t *pr,
           const uint32_t *pinv, const uint32_t *wtab, int inv, ntt_fuse f)
{
  /* blockDim.y rows (segments) per block, each with blockDim.x threads and
     its own part of the shared memory */
  extern __shared__ uint32_t xsh[];
  const int m = 1 << mlg;
  uint32_t *x = xsh + threadIdx.y * m;
  for (int r0 = blockIdx.y * blockDim.y; r0 < rows;
       r0 += gridDim.y * blockDim.y)
    {
      const int row = r0 + threadIdx.y;
      const bool ok = row < rows;
      const int i = (ok ? row : 0) % P;
      const uint32_t p = pr[i], pv = pinv[i];
      const uint32_t *w = wtab + ((size_t) i << base);
      /* the 8th roots of unity (stage 2's twiddles) */
      const uint32_t zc[4] = { 0, w[5], w[6], w[7] };
      uint32_t *v = res + (size_t) (ok ? row : 0) * rowlen
                    + (size_t) blockIdx.x * qstep;
      if (ok)
        for (int e = threadIdx.x; e < m; e += blockDim.x)
          x[NTT_PADI (e)] = v[(size_t) e * stride];
      __syncthreads ();
      if (!inv)
        for (int s = mlg - 1; s >= 0; s -= 3)
          {
            const int r = s + 1 < 3 ? s + 1 : 3;
            if (r == 3)
              ntt_rad_pass<3> (x, m, s, p, pv, w, zc, 0);
            else if (r == 2)
              ntt_rad_pass<2> (x, m, s, p, pv, w, zc, 0);
            else
              ntt_rad_pass<1> (x, m, s, p, pv, w, zc, 0);
            __syncthreads ();
          }
      else
        for (int s = 0; s < mlg; s += 3)
          {
            const int r = mlg - s < 3 ? mlg - s : 3;
            if (r == 3)
              ntt_rad_pass<3> (x, m, s, p, pv, w, zc, 1);
            else if (r == 2)
              ntt_rad_pass<2> (x, m, s, p, pv, w, zc, 1);
            else
              ntt_rad_pass<1> (x, m, s, p, pv, w, zc, 1);
            __syncthreads ();
          }
      if (ok)
        {
          if (f.sc != NULL)
            for (int e = threadIdx.x; e < m; e += blockDim.x)
              v[(size_t) e * stride] = ntt_scale (x[NTT_PADI (e)], f, i, p);
          else
            for (int e = threadIdx.x; e < m; e += blockDim.x)
              v[(size_t) e * stride] = x[NTT_PADI (e)];
        }
      __syncthreads ();
    }
}

/* The column pass of a four-step transform of length 2^lg = R1 R2, R1 =
   2^C <= 8 (C = lg - SL), R2 = 2^SL, in registers: one thread per
   column n2 of a row, its R1 entries t1 R2 + n2 (consecutive threads,
   consecutive columns: the loads are coalesced), with the twiddle
   w_lg^(+-bitrev (t1) n2) of the four-step transform (forward after the
   column transform, inverse before it, as in k_ntt_small).  The twiddles
   a^k, a = w_lg^(+-n2) from TC, come from products in registers (canonical,
   so that they can multiply lazy values); those of the length R1
   transform are WTAB's (as in k_ntt_sub).  SC != NULL (inverse): the final
   scaling at the store. */
static __host__ __device__ constexpr int
ntt_brev (int t, int c)
{
  return c == 0 ? 0 : ((t & 1) << (c - 1)) | ntt_brev (t >> 1, c - 1);
}

template<int C>
__global__ void __launch_bounds__ (256)
k_ntt_cols (uint32_t *res, size_t rowlen, int rows, int P, int base, int sl,
            const uint32_t *pr, const uint32_t *pinv, const uint32_t *wtab,
            const uint32_t *tc, int ncol, int inv, ntt_fuse f)
{
  const int T = 1 << C;
  const size_t R2 = (size_t) 1 << sl, total = (size_t) rows << sl;
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x; x < total;
       x += (size_t) gridDim.x * blockDim.x)
    {
      const uint32_t row = (uint32_t) (x >> sl);
      const size_t n2 = x & (R2 - 1);
      const int i = (int) (row % (uint32_t) P);
      const uint32_t p = pr[i], p2 = 2 * p, pv = pinv[i];
      const uint32_t *w = wtab + ((size_t) i << base);
      uint32_t *v = res + row * rowlen + n2;
      uint32_t y[T], tw[T];
#pragma unroll
      for (int t = 0; t < T; t++)
        y[t] = v[(size_t) t << sl];
      /* tw[k] = a^k, k >= 1 */
      tw[1] = __ldg (&tc[(((size_t) i * ncol + C - 1) << sl) + n2]);
#pragma unroll
      for (int k = 2; k < T; k++)
        tw[k] = rns_red (rns_mont (tw[k - 1], tw[1], p, pv), p);
      if (inv)
#pragma unroll
        for (int t = 1; t < T; t++)
          y[t] = rns_mont (y[t], tw[ntt_brev (t, C)], p, pv);
      if (!inv)
#pragma unroll
        for (int r = C - 1; r >= 0; r--)
          {
            const int h = 1 << r;
#pragma unroll
            for (int t = 0; t < T; t++)
              if (!(t & h))
                {
                  const uint32_t a = y[t], bb = y[t + h];
                  y[t] = rns_red (a + bb, p2);
                  y[t + h] = rns_mont (a - bb + p2, __ldg (&w[h | (t & (h - 1))]),
                                       p, pv);
                }
          }
      else
#pragma unroll
        for (int r = 0; r < C; r++)
          {
            const int h = 1 << r;
#pragma unroll
            for (int t = 0; t < T; t++)
              if (!(t & h))
                {
                  const uint32_t a = rns_red (y[t], p2);
                  const uint32_t tt = rns_mont (y[t + h],
                                                __ldg (&w[h | (t & (h - 1))]),
                                                p, pv);
                  y[t] = rns_red (a + tt, p2);
                  y[t + h] = rns_red (a - tt + p2, p2);
                }
          }
      if (!inv)
#pragma unroll
        for (int t = 1; t < T; t++)
          y[t] = rns_mont (y[t], tw[ntt_brev (t, C)], p, pv);
      if (f.sc != NULL)
#pragma unroll
        for (int t = 0; t < T; t++)
          y[t] = ntt_scale (y[t], f, i, p);
#pragma unroll
      for (int t = 0; t < T; t++)
        v[(size_t) t << sl] = y[t];
    }
}

/* four-step twiddle: entry t1 R2 + n2 of a row of length 2^lg (after the
   column pass t1 holds frequency k1 = bitrev (t1)) times w^(+-k1 n2), w of
   order 2^lg = w_max^(2^(K - lg)) */
__global__ void
k_twiddle (uint32_t *res, size_t rowlen, int rows, int P, int lg, int r1lg,
           int base, int K, const uint32_t *pr, const uint2 *tlo,
           const uint2 *thi)
{
  for (int row = blockIdx.y; row < rows; row += gridDim.y)
  {
  const int i = row % P;
  const uint32_t p = pr[i];
  const size_t len = (size_t) 1 << lg;
  uint32_t *v = res + (size_t) row * rowlen;
  const uint2 *lo = tlo + ((size_t) i << base);
  const uint2 *hi = thi + ((size_t) i << (K - base));
  for (size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x; e < len;
       e += (size_t) gridDim.x * blockDim.x)
    {
      const uint32_t t1 = (uint32_t) (e >> base);
      const uint32_t n2 = (uint32_t) (e & ((1u << base) - 1));
      const uint32_t k1 = __brev (t1) >> (32 - r1lg);
      const uint64_t ex = ((uint64_t) k1 * n2) << (K - lg);   /* < 2^K */
      const uint2 a = lo[ex & ((1u << base) - 1)];
      const uint2 b = hi[ex >> base];
      uint32_t y = rns_shoup (v[e], a.x, a.y, p);
      v[e] = rns_shoup (y, b.x, b.y, p);
    }
  }
}

/* times 2^-lg (and 2^32 with NMUL), canonical */
/* Flat loops over all ROWS x LEN entries (rows of any length, short ones
   included, keep every thread busy); LEN a power of 2. */
__global__ void
k_scale (uint32_t *res, size_t rowlen, int rows, int P, int lg, int K,
         int nmul, const uint32_t *pr, const uint2 *sc)
{
  const size_t total = (size_t) rows << lg;
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x; x < total;
       x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t row = x >> lg, e = x & (((size_t) 1 << lg) - 1);
      const int i = (int) (row % P);
      const uint32_t p = pr[i];
      const uint2 f = sc[((size_t) i * (K + 1) + lg) * 2 + (nmul ? 1 : 0)];
      uint32_t *v = res + row * rowlen + e;
      *v = rns_red (rns_shoup (*v, f.x, f.y, p), p);
    }
}

/* B shared: polynomial j of A times polynomial j / SHARE of B (P rows
   each) */
__global__ void
k_mul (uint32_t *r, const uint32_t *a, const uint32_t *b, size_t rowlen,
       int rows, int P, int lg, int share, const uint32_t *pr,
       const uint32_t *pinv)
{
  const size_t total = (size_t) rows << lg;
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x; x < total;
       x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t row = x >> lg, e = x & (((size_t) 1 << lg) - 1);
      const int i = (int) (row % P);
      const size_t off = row * rowlen + e;
      const size_t boff = (row / P / share * P + i) * rowlen + e;
      r[off] = rns_mont (a[off], b[boff], pr[i], pinv[i]);
    }
}

/* ---- host ------------------------------------------------------------- */

static uint64_t
mulmod64 (uint64_t a, uint64_t b, uint64_t m)
{
  return (uint64_t) ((unsigned __int128) a * b % m);
}

static uint64_t
powmod64 (uint64_t b, uint64_t e, uint64_t m)
{
  uint64_t r = 1;
  b %= m;
  while (e)
    {
      if (e & 1)
        r = mulmod64 (r, b, m);
      b = mulmod64 (b, b, m);
      e >>= 1;
    }
  return r;
}

static bool
is_prime32 (uint32_t n)
{
  if (n < 2)
    return false;
  for (uint32_t q : { 2u, 3u, 5u, 7u, 11u, 13u })
    if (n % q == 0)
      return n == q;
  uint32_t d = n - 1;
  int s = 0;
  while ((d & 1) == 0)
    {
      d >>= 1;
      s++;
    }
  for (uint32_t a : { 2u, 3u, 5u, 7u })   /* exact below 3.2e9 */
    {
      uint64_t x = powmod64 (a, d, n);
      if (x == 1 || x == n - 1)
        continue;
      bool comp = true;
      for (int r = 1; r < s && comp; r++)
        {
          x = mulmod64 (x, x, n);
          if (x == n - 1)
            comp = false;
        }
      if (comp)
        return false;
    }
  return true;
}

/* a generator of the 2^K-th roots of unity mod p = c 2^K + 1 */
static uint32_t
root_of_unity (uint32_t p, int K)
{
  const uint32_t c = (p - 1) >> K;
  std::vector<uint32_t> f;
  uint32_t m = c;
  for (uint32_t q = 3; q * q <= m; q += 2)
    if (m % q == 0)
      {
        f.push_back (q);
        while (m % q == 0)
          m /= q;
      }
  if (m > 1 && m % 2)
    f.push_back (m);
  f.push_back (2);
  for (uint32_t g = 2;; g++)
    {
      bool ok = true;
      for (uint32_t q : f)
        ok = ok && powmod64 (g, (p - 1) / q, p) != 1;
      if (ok)
        return (uint32_t) powmod64 (g, c, p);
    }
}

static uint2
shoup_pair (uint32_t w, uint32_t p)
{
  return make_uint2 (w, (uint32_t) (((uint64_t) w << 32) / p));
}

template<class T>
static T *
upload (gpu_rns *R, const std::vector<T> &h)
{
  T *d;
  CUDA_CHECK (cudaMalloc ((void **) &d, h.size () * sizeof (T)));
  CUDA_CHECK (cudaMemcpy (d, h.data (), h.size () * sizeof (T),
                          cudaMemcpyHostToDevice));
  R->dev.push_back (d);
  return d;
}

static void
to_limbs (uint32_t *w, int n, const mpz_t a)
{
  size_t cnt = 0;
  memset (w, 0, n * sizeof (uint32_t));
  assert (mpz_sgn (a) >= 0 && mpz_sizeinbase (a, 2) <= 32 * (size_t) n);
  mpz_export (w, &cnt, -1, sizeof (uint32_t), 0, 0, a);
}

gpu_rns *
gpu_rns_create (const mpz_t N, int lg_want, int below_2n)
{
  gpu_rns *R = new gpu_rns;
  /* 4 spare bits: results below 2N, and sums of a few of them, fit */
  R->nl = (int) ((mpz_sizeinbase (N, 2) + 4 + 31) / 32);
  R->d_tmp = NULL;
  R->tmp_words = 0;

  /* the primes: the largest K <= lg_want with enough of them, M = prod p
     > 2^(K+1) Bn^2 with the inputs below Bn: a convolution value is then
     below 2^K Bn^2 < M / 2, what the CRT's alpha needs (see the top) */
  mpz_t Pm, T, Bn;
  mpz_inits (Pm, T, Bn, NULL);
  if (below_2n)
    mpz_mul_2exp (Bn, N, 1);
  else
    mpz_setbit (Bn, 32 * (size_t) R->nl);
  int K;
  for (K = lg_want; K >= 1; K--)
    {
      R->p.clear ();
      mpz_set_ui (Pm, 1);
      mpz_mul (T, Bn, Bn);
      mpz_mul_2exp (T, T, K + 1);
      bool enough = false;
      for (uint32_t c = (RNS_PLIM - 2) >> K; c >= 1 && !enough; c--)
        {
          const uint32_t p = (c << K) | 1;
          if (p >= RNS_PLIM || !is_prime32 (p))
            continue;
          R->p.push_back (p);
          mpz_mul_ui (Pm, Pm, p);
          enough = mpz_cmp (Pm, T) > 0;
        }
      if (enough)
        break;
    }
  assert (K >= 1);
  R->K = K;
  R->P = (int) R->p.size ();
  /* in-block transforms up to 2^14 (on an A10, 2^14 in a block is faster
     than four-step 2 x 2^13; the four-step rows stay 2^13) */
  R->base = K < 14 ? K : 14;
  const int P = R->P, nl = R->nl, base = R->base;

  /* transform tables and scales */
  std::vector<uint32_t> pinv (P);
  std::vector<uint32_t> ws ((size_t) P << base, 0), wsi (ws.size (), 0);
  std::vector<uint2> tlo ((size_t) P << base), tloi (tlo.size ());
  std::vector<uint2> thi ((size_t) P << (K - base)), thii (thi.size ());
  std::vector<uint2> sc ((size_t) P * (K + 1) * 2);
  const int sl = std::min (base, 13), ncol = std::min (K - sl, 3);
  R->split = sl;
  R->ncol = ncol;
  std::vector<uint32_t> tc (((size_t) P * ncol) << sl), tci (tc.size ());
  for (int i = 0; i < P; i++)
    {
      const uint32_t p = R->p[i];
      uint32_t inv = p;
      for (int r = 0; r < 5; r++)
        inv *= 2 - p * inv;
      pinv[i] = 0 - inv;
      const uint32_t w = root_of_unity (p, K);
      const uint32_t wi = (uint32_t) powmod64 (w, p - 2, p);
      const uint32_t wbase = (uint32_t) powmod64 (w, 1ull << (K - base), p);
      const uint32_t wbasei = (uint32_t) powmod64 (wi, 1ull << (K - base), p);
      const uint64_t r32 = (1ull << 32) % p;
      for (int s = 0; s < base; s++)
        {
          /* stage s: powers of w_b^(2^(base-1-s)), in Montgomery form */
          const uint64_t ws1 = powmod64 (wbase, 1ull << (base - 1 - s), p);
          const uint64_t wsi1 = powmod64 (wbasei, 1ull << (base - 1 - s), p);
          uint64_t x = r32, y = r32;
          for (size_t k = 0; k < ((size_t) 1 << s); k++)
            {
              ws[((size_t) i << base) + ((size_t) 1 << s) + k] = (uint32_t) x;
              wsi[((size_t) i << base) + ((size_t) 1 << s) + k] = (uint32_t) y;
              x = x * ws1 % p;
              y = y * wsi1 % p;
            }
        }
      uint64_t x = 1, y = 1;
      for (size_t j = 0; j < ((size_t) 1 << base); j++)
        {
          tlo[((size_t) i << base) + j] = shoup_pair ((uint32_t) x, p);
          tloi[((size_t) i << base) + j] = shoup_pair ((uint32_t) y, p);
          x = x * w % p;
          y = y * wi % p;
        }
      const uint64_t wh = powmod64 (w, 1ull << base, p);
      const uint64_t whi = powmod64 (wi, 1ull << base, p);
      x = y = 1;
      for (size_t j = 0; j < ((size_t) 1 << (K - base)); j++)
        {
          thi[((size_t) i << (K - base)) + j] = shoup_pair ((uint32_t) x, p);
          thii[((size_t) i << (K - base)) + j] = shoup_pair ((uint32_t) y, p);
          x = x * wh % p;
          y = y * whi % p;
        }
      /* k_ntt_cols: w_(sl+c)^j = w^(j 2^(K-sl-c)) */
      for (int c = 1; c <= ncol; c++)
        {
          const uint64_t a = powmod64 (w, 1ull << (K - sl - c), p);
          const uint64_t ai = powmod64 (wi, 1ull << (K - sl - c), p);
          x = y = r32;
          for (size_t j = 0; j < ((size_t) 1 << sl); j++)
            {
              tc[(((size_t) i * ncol + c - 1) << sl) + j] = (uint32_t) x;
              tci[(((size_t) i * ncol + c - 1) << sl) + j] = (uint32_t) y;
              x = x * a % p;
              y = y * ai % p;
            }
        }
      const uint64_t inv2 = (p + 1) / 2;
      for (int lg = 0; lg <= K; lg++)
        {
          const uint64_t f = powmod64 (inv2, lg, p);
          sc[((size_t) i * (K + 1) + lg) * 2] = shoup_pair ((uint32_t) f, p);
          sc[((size_t) i * (K + 1) + lg) * 2 + 1]
            = shoup_pair ((uint32_t) (f * r32 % p), p);
        }
    }

  /* conversion tables */
  std::vector<uint32_t> pw ((size_t) nl * P), pw64 ((size_t) nl * P), r64 (P),
                        A ((size_t) P * nl), Bm (nl), Nl (nl);
  std::vector<uint2> minv (P);
  std::vector<unsigned long long> ip (P);
  mpz_t Mi, t, R64;
  mpz_inits (Mi, t, R64, NULL);
  mpz_setbit (R64, 64);
  for (int i = 0; i < P; i++)
    {
      const uint32_t p = R->p[i];
      r64[i] = (uint32_t) mulmod64 ((1ull << 32) % p, (1ull << 32) % p, p);
      uint64_t x = 1;
      for (int j = 0; j < nl; j++)
        {
          pw[(size_t) j * P + i] = (uint32_t) x;
          pw64[(size_t) j * P + i] = (uint32_t) mulmod64 (x, r64[i], p);
          x = (x << 32) % p;
        }
      mpz_divexact_ui (Mi, Pm, p);
      const uint32_t mip = (uint32_t) mpz_fdiv_ui (Mi, p);
      minv[i] = shoup_pair ((uint32_t) powmod64 (mip, p - 2, p), p);
      ip[i] = (unsigned long long) ((((unsigned __int128) 1) << 63) / p);
      mpz_mul (t, Mi, R64);
      mpz_mod (t, t, N);
      to_limbs (&A[(size_t) i * nl], nl, t);
    }
  mpz_mul (t, Pm, R64);
  mpz_neg (t, t);
  mpz_mod (t, t, N);
  to_limbs (Bm.data (), nl, t);
  to_limbs (Nl.data (), nl, N);
  {
    uint64_t n0 = Nl[0] | (nl > 1 ? (uint64_t) Nl[1] << 32 : 0);
    uint64_t inv = n0;
    for (int r = 0; r < 6; r++)
      inv *= 2 - n0 * inv;
    R->ninv = 0 - inv;
  }
  mpz_clears (Mi, t, R64, Pm, T, Bn, NULL);

  R->d_p = upload (R, R->p);
  R->d_pinv = upload (R, pinv);
  R->d_ws = upload (R, ws);
  R->d_wsi = upload (R, wsi);
  R->d_tlo = upload (R, tlo);
  R->d_tloi = upload (R, tloi);
  R->d_thi = upload (R, thi);
  R->d_thii = upload (R, thii);
  R->d_tc = ncol > 0 ? upload (R, tc) : NULL;
  R->d_tci = ncol > 0 ? upload (R, tci) : NULL;
  R->d_sc = upload (R, sc);
  R->d_pow = upload (R, pw);
  R->d_pow64 = upload (R, pw64);
  R->d_r64 = upload (R, r64);
  {
    std::vector<uint4> rr (P);
    for (int i = 0; i < P; i++)
      {
        const uint32_t p = R->p[i];
        const uint2 a = shoup_pair ((uint32_t) ((1ull << 32) % p), p);
        const uint2 b = shoup_pair (r64[i], p);
        rr[i] = make_uint4 (a.x, a.y, b.x, b.y);
      }
    R->d_rr = upload (R, rr);
  }
  R->d_minv = upload (R, minv);
  R->d_ip = upload (R, ip);
  R->d_A = upload (R, A);
  R->d_B = upload (R, Bm);
  R->d_N = upload (R, Nl);
  {
    const int nl4 = (nl + 3) & ~3;
    std::vector<uint32_t> A4 ((size_t) R->P * nl4, 0), B4 (nl4, 0), N4 (nl4, 0);
    for (int i = 0; i < R->P; i++)
      for (int w = 0; w < nl; w++)
        A4[(size_t) i * nl4 + w] = A[(size_t) i * nl + w];
    for (int w = 0; w < nl; w++)
      {
        B4[w] = Bm[w];
        N4[w] = Nl[w];
      }
    R->d_A4 = upload (R, A4);
    R->d_B4 = upload (R, B4);
    R->d_N4 = upload (R, N4);
  }
  return R;
}

void
gpu_rns_destroy (gpu_rns *R)
{
  if (rns_prof.on > 0)
    {
      static const char *name[RP_COUNT] = { "from_z", "to_z", "ntt", "mul",
                                            "Z/N" };
      printf ("RNS kernels:");
      for (int i = 0; i < RP_COUNT; i++)
        printf (" %s %.1f ms (%ld calls)", name[i], rns_prof.ms[i],
                rns_prof.calls[i]);
      printf ("\n");
      for (int i = 0; i < RP_ZN; i++)
        {
          printf ("  %s by log2 length:", name[i]);
          for (int l = 0; l < 32; l++)
            if (rns_prof.msl[i][l] > 0)
              printf (" %d: %.1f", l, rns_prof.msl[i][l]);
          printf ("\n");
        }
    }
  for (void *d : R->dev)
    CUDA_CHECK (cudaFree (d));
  if (R->d_tmp != NULL)
    CUDA_CHECK (cudaFree (R->d_tmp));
  delete R;
}

int gpu_rns_primes (const gpu_rns *R) { return R->P; }
const uint32_t *gpu_rns_dev_N (const gpu_rns *R) { return R->d_N; }
int gpu_rns_lg (const gpu_rns *R) { return R->K; }
int gpu_rns_limbs (const gpu_rns *R) { return R->nl; }

/* grid y for a loop over N items */
static unsigned
gy (size_t n)
{
  return (unsigned) std::min (n, (size_t) 65535);
}

static unsigned
flat_blocks (size_t items)
{
  return (unsigned) std::max ((size_t) 1,
                              std::min ((size_t) 1 << 16, (items + 255) / 256));
}

/* RN (at most MAXRN) for the conversions' matrix products (cv_mac) with N
   columns: the fewest column slots ceil (N / 8 RN) 8 RN, the larger RN on
   a tie */
static int
cv_rn (int n, int maxrn)
{
  int best = maxrn;
  for (int rn = maxrn - 2; rn >= 3; rn -= 2)
    if ((n + 8 * rn - 1) / (8 * rn) * rn
        < (n + 8 * best - 1) / (8 * best) * best)
      best = rn;
  return best;
}

static void
from_z (gpu_rns *R, uint32_t *res, size_t rowlen, const uint32_t *a,
        size_t sa, size_t C, int B, size_t one_at)
{
  const int th = 64;
  dim3 grid ((unsigned) ((rowlen + RNS_CT - 1) / RNS_CT),
             (unsigned) ((R->P + th - 1) / th), gy (B));
  rns_prof.lg = 0;
  while (((size_t) 1 << rns_prof.lg) < rowlen)
    rns_prof.lg++;
  rp_begin ();
  if (getenv ("GMPECM_RNS_OLDFROMZ") == NULL)
    {
      const size_t total = C * (size_t) B;
      assert (total < ((size_t) 1 << 32));
      if (total > 0)
        {
          const int rn = cv_rn (R->P, 9);
          dim3 g2 ((unsigned) ((total + CV_TM - 1) / CV_TM),
                   (unsigned) ((R->P + 8 * rn - 1) / (8 * rn)));
#define FROM_Z_MM(RN) \
  k_from_z_mm<RN><<<g2, 256>>> (res, rowlen, a, sa, (uint32_t) C,           \
                                (uint32_t) total, R->nl, R->P, R->d_p,     \
                                R->d_pinv, R->d_pow64)
          if (rn == 3)
            FROM_Z_MM (3);
          else if (rn == 5)
            FROM_Z_MM (5);
          else if (rn == 7)
            FROM_Z_MM (7);
          else
            FROM_Z_MM (9);
#undef FROM_Z_MM
        }
      if (rowlen > C)
        k_from_z_pad<<<flat_blocks ((size_t) B * R->P * (rowlen - C)), 256>>> (
            res, rowlen, C, (size_t) B * R->P, one_at);
    }
  else
    k_from_z<<<grid, th>>> (res, rowlen, a, sa, C, B, R->nl, R->P, R->d_p,
                            R->d_pow, R->d_rr, one_at);
  CUDA_CHECK (cudaGetLastError ());
  rp_end (RP_FROM_Z);
}

void
gpu_rns_from_z (gpu_rns *R, uint32_t *res, size_t rowlen, const uint32_t *a,
                size_t sa, size_t C, int B)
{
  from_z (R, res, rowlen, a, sa, C, B, ~(size_t) 0);
}

/* to_z by k_to_z_mm */
static void
to_z_mm (gpu_rns *R, uint32_t *a, size_t sc, const uint32_t *res,
         size_t rowlen, size_t C, int B, int sub1)
{
  const size_t total = C * (size_t) B;
  assert (total < ((size_t) 1 << 32) && R->P < (1 << 18));
  if (total == 0)
    return;
  const int rn = cv_rn (std::min ((R->nl + 3) & ~3, R->nl + 2), 7);
  const unsigned blocks = (unsigned) ((total + CV_TM - 1) / CV_TM);
#define TO_Z_MM(RN) \
  k_to_z_mm<RN><<<blocks, 256>>> (a, sc, res, rowlen, (uint32_t) C,          \
                                  (uint32_t) total, R->nl, R->P, R->d_p,     \
                                  R->d_minv, R->d_ip, R->d_A4, R->d_B4,      \
                                  R->d_N4, R->ninv, sub1)
  if (rn == 3)
    TO_Z_MM (3);
  else if (rn == 5)
    TO_Z_MM (5);
  else
    TO_Z_MM (7);
#undef TO_Z_MM
}

static void
to_z (gpu_rns *R, uint32_t *a, size_t sc, const uint32_t *res, size_t rowlen,
      size_t C, int B, int sub1)
{
  const size_t smem = (size_t) R->nl * 8 + (size_t) R->nl * 4
                      + (size_t) (R->nl + 3) * 4 + (size_t) R->P * 4;
  static size_t smem_set = 0;
  if (smem > 48 * 1024 && smem > smem_set)
    {
      CUDA_CHECK (cudaFuncSetAttribute (k_to_z,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
      smem_set = smem;
    }
  if (getenv ("GMPECM_RNS_OLDTOZ") == NULL && R->nl >= 2)
    {
      rns_prof.lg = 0;
      while (((size_t) 1 << rns_prof.lg) < rowlen)
        rns_prof.lg++;
      rp_begin ();
      to_z_mm (R, a, sc, res, rowlen, C, B, sub1);
      CUDA_CHECK (cudaGetLastError ());
      rp_end (RP_TO_Z);
      return;
    }
  /* thread per coefficient, if P residues of 32 of them fit in shared
     memory */
  int tb = 128;
  while (tb > 32 && (size_t) tb * R->P * 4 > 48 * 1024)
    tb /= 2;
  const size_t tsmem = (size_t) tb * R->P * 4;
  static size_t tsmem_set = 0;
  if (tsmem <= 99 * 1024 && R->nl >= 2)
    {
      if (tsmem > 48 * 1024 && tsmem > tsmem_set)
        {
          CUDA_CHECK (cudaFuncSetAttribute (k_to_z_t,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      (int) tsmem));
          tsmem_set = tsmem;
        }
      const size_t total = C * (size_t) B;
      rns_prof.lg = 0;
      while (((size_t) 1 << rns_prof.lg) < rowlen)
        rns_prof.lg++;
      rp_begin ();
      k_to_z_t<<<(unsigned) ((total + tb - 1) / tb), tb, tsmem>>> (
          a, sc, res, rowlen, C, B, R->nl, R->P, R->d_p, R->d_minv, R->d_ip,
          R->d_A4, R->d_B4, R->d_N4, R->ninv, sub1);
      CUDA_CHECK (cudaGetLastError ());
      rp_end (RP_TO_Z);
      return;
    }
  dim3 grid ((unsigned) C, gy (B));
  rp_begin ();
  k_to_z<<<grid, 128, smem>>> (a, sc, res, rowlen, B, R->nl, R->P, R->d_p,
                               R->d_minv, R->d_ip, R->d_A, R->d_B, R->d_N,
                               R->ninv, sub1);
  CUDA_CHECK (cudaGetLastError ());
  rp_end (RP_TO_Z);
}

void
gpu_rns_to_z (gpu_rns *R, uint32_t *a, size_t sc, const uint32_t *res,
              size_t rowlen, size_t C, int B)
{
  to_z (R, a, sc, res, rowlen, C, B, 0);
}

static void
ntt_sub (gpu_rns *R, uint32_t *res, size_t rowlen, int rows, int mlg,
         size_t nsub, size_t qstep, size_t stride, int inv,
         ntt_fuse f = ntt_fuse ())
{
  /* the fused work must have a kernel that does it (see below) */
  assert ((f.tlo == NULL || mlg <= 7)
          && (f.sc == NULL || mlg >= 1));
  if (mlg == 0)
    return;                     /* length 1: nothing to do */
  const int half = 1 << (mlg - 1);
  if (mlg <= 7 && (f.tlo != NULL || mlg <= 5))
    {
      const int spb = 256 / half;
      const size_t nseg = nsub * (size_t) rows;
      const unsigned blocks = (unsigned) std::min ((size_t) 1 << 20,
                                                   (nseg + spb - 1) / spb);
      k_ntt_small<<<blocks, 256, (size_t) spb * 4 << mlg>>> (
          res, rowlen, rows, R->P, mlg, nsub, qstep, stride, R->base, R->d_p,
          R->d_pinv, inv ? R->d_wsi : R->d_ws, inv, f);
      CUDA_CHECK (cudaGetLastError ());
      return;
    }
  if (getenv ("GMPECM_RNS_RADIX2") == NULL)
    {
      const size_t m = (size_t) 1 << mlg;
      /* threads per row (measured on an A10): 256, 512 for 2^13; short
         rows several to a block, at least 128 threads */
      const int th = (int) std::min (mlg >= 14 ? (size_t) 1024 : mlg >= 13 ? (size_t) 512 : 256, m / 8);
      const int rpb = th >= 128 ? 1 : 128 / th;
      dim3 grid ((unsigned) nsub, gy ((rows + rpb - 1) / rpb));
      const size_t smem = (size_t) m * 4 * rpb;
      static size_t smem_set = 0;
      if (smem > 48 * 1024 && smem > smem_set)
        {
          CUDA_CHECK (cudaFuncSetAttribute (k_ntt_rad,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      (int) smem));
          smem_set = smem;
        }
      k_ntt_rad<<<grid, dim3 (th, rpb), smem>>> (
          res, rowlen, rows, R->P, mlg, qstep, stride, R->base, R->d_p,
          R->d_pinv, inv ? R->d_wsi : R->d_ws, inv, f);
      CUDA_CHECK (cudaGetLastError ());
      return;
    }
  assert (f.tlo == NULL && f.sc == NULL);
  const int th = half > 256 ? 256 : half;
  dim3 grid ((unsigned) nsub, gy (rows));
  {
    static size_t sub_set = 0;
    const size_t smem = (size_t) 4 << mlg;
    if (smem > 48 * 1024 && smem > sub_set)
      {
        CUDA_CHECK (cudaFuncSetAttribute (k_ntt_sub,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        sub_set = smem;
      }
  }
  k_ntt_sub<<<grid, th, (size_t) 4 << mlg>>> (res, rowlen, rows, R->P, mlg,
                                              qstep, stride, R->base, R->d_p,
                                              R->d_pinv,
                                              inv ? R->d_wsi : R->d_ws, inv);
  CUDA_CHECK (cudaGetLastError ());
}

/* the column pass of a four-step transform: by k_ntt_cols (C <= ncol,
   with the twiddle), else ntt_sub */
static void
ntt_cols (gpu_rns *R, uint32_t *res, size_t rowlen, int rows, int r1lg,
          size_t R2, int inv, const ntt_fuse &ft)
{
  if (ft.tlo == NULL || r1lg > R->ncol || R2 != ((size_t) 1 << R->split)
      || getenv ("GMPECM_RNS_OLDCOLS") != NULL)
    {
      ntt_sub (R, res, rowlen, rows, r1lg, R2, 1, R2, inv, ft);
      return;
    }
  const unsigned blocks = flat_blocks ((size_t) rows << R->split);
  const uint32_t *ws = inv ? R->d_wsi : R->d_ws;
  const uint32_t *tc = inv ? R->d_tci : R->d_tc;
#define NTT_COLS(C) \
  k_ntt_cols<C><<<blocks, 256>>> (res, rowlen, rows, R->P, R->base,      \
                                  R->split, R->d_p, R->d_pinv, ws, tc,     \
                                  R->ncol, inv, ft)
  if (r1lg == 1)
    NTT_COLS (1);
  else if (r1lg == 2)
    NTT_COLS (2);
  else
    NTT_COLS (3);
#undef NTT_COLS
  CUDA_CHECK (cudaGetLastError ());
}

static unsigned
gx (size_t len)
{
  return (unsigned) std::max ((size_t) 1, std::min ((size_t) 1024, len / 256));
}

void
gpu_rns_ntt (gpu_rns *R, uint32_t *res, size_t rowlen, int rows, int lg,
             int inv, int nmul)
{
  assert (lg <= R->K && rows % R->P == 0);
  rns_prof.lg = lg;
  rp_begin ();
  const int base = R->base;
  const size_t len = (size_t) 1 << lg;
  /* the scaling of an inverse transform in its last pass, the four-step
     twiddle in the column pass (the radix-2 fallback does neither) */
  const bool fuse = getenv ("GMPECM_RNS_RADIX2") == NULL;
  ntt_fuse fs = ntt_fuse ();
  fs.K = R->K;
  fs.lg = lg;
  fs.nmul = nmul;
  if (inv && fuse && lg >= 1)
    fs.sc = R->d_sc;
  if (lg <= base)
    ntt_sub (R, res, rowlen, rows, lg, 1, 0, 1, inv, fs);
  else
    {
      /* rows of 2^13 (2^14 rows, one block per SM, are slower) and the
         twiddle in the column pass, if the columns are short enough for
         k_ntt_small; else (and in the radix-2 fallback) rows of 2^base
         and a twiddle pass */
      const bool ftw = fuse && lg - std::min (base, 13) <= 7;
      const int split = ftw ? std::min (base, 13) : base;
      const int r1lg = lg - split;
      const size_t R1 = (size_t) 1 << r1lg, R2 = (size_t) 1 << split;
      ntt_fuse ft = fs;
      ft.r1lg = r1lg;
      if (ftw)
        {
          ft.tlo = inv ? R->d_tloi : R->d_tlo;
          ft.thi = inv ? R->d_thii : R->d_thi;
        }
      if (!inv)
        {
          ntt_cols (R, res, rowlen, rows, r1lg, R2, 0, ft);       /* columns */
          if (!ftw)
            {
              k_twiddle<<<dim3 (gx (len), gy (rows)), 256>>> (
                  res, rowlen, rows, R->P, lg, r1lg, base, R->K, R->d_p,
                  R->d_tlo, R->d_thi);
              CUDA_CHECK (cudaGetLastError ());
            }
          ntt_sub (R, res, rowlen, rows, split, R1, R2, 1, 0);      /* rows */
        }
      else
        {
          ntt_sub (R, res, rowlen, rows, split, R1, R2, 1, 1);
          if (!ftw)
            {
              k_twiddle<<<dim3 (gx (len), gy (rows)), 256>>> (
                  res, rowlen, rows, R->P, lg, r1lg, base, R->K, R->d_p,
                  R->d_tloi, R->d_thii);
              CUDA_CHECK (cudaGetLastError ());
            }
          ntt_cols (R, res, rowlen, rows, r1lg, R2, 1, ft);
        }
    }
  if (inv && fs.sc == NULL)
    {
      k_scale<<<flat_blocks ((size_t) rows << lg), 256>>> (
          res, rowlen, rows, R->P, lg, R->K, nmul, R->d_p, R->d_sc);
      CUDA_CHECK (cudaGetLastError ());
    }
  rp_end (RP_NTT);
}

static void
rns_mul (gpu_rns *R, uint32_t *r, const uint32_t *a, const uint32_t *b,
         size_t rowlen, int rows, size_t len, int share)
{
  int lg = 0;
  while (((size_t) 1 << lg) < len)
    lg++;
  assert (((size_t) 1 << lg) == len);
  rns_prof.lg = lg;
  rp_begin ();
  k_mul<<<flat_blocks ((size_t) rows << lg), 256>>> (r, a, b, rowlen, rows,
                                                     R->P, lg, share, R->d_p,
                                                     R->d_pinv);
  CUDA_CHECK (cudaGetLastError ());
  rp_end (RP_MUL);
}

void
gpu_rns_mul (gpu_rns *R, uint32_t *r, const uint32_t *a, const uint32_t *b,
             size_t rowlen, int rows, size_t len, int bbc)
{
  rns_mul (R, r, a, b, rowlen, rows, len, bbc ? rows / R->P : 1);
}

static uint32_t *
scratch (gpu_rns *R, size_t words)
{
  if (R->tmp_words < words)
    {
      if (R->d_tmp != NULL)
        CUDA_CHECK (cudaFree (R->d_tmp));
      CUDA_CHECK (cudaMalloc ((void **) &R->d_tmp, words * sizeof (uint32_t)));
      R->tmp_words = words;
    }
  return R->d_tmp;
}

void
gpu_rns_prep (gpu_rns *R, uint32_t *out, const uint32_t *b, size_t sb,
              size_t nb, int lg, int batch)
{
  const size_t len = (size_t) 1 << lg;
  gpu_rns_from_z (R, out, len, b, sb, nb, batch);
  gpu_rns_ntt (R, out, len, batch * R->P, lg, 0, 0);
}

void
gpu_rns_conv_prep (gpu_rns *R, uint32_t *c, size_t sc, size_t o0, size_t oc,
                   const uint32_t *a, size_t sa, size_t na,
                   const uint32_t *bprep, int share, int lg, int batch)
{
  assert (lg <= R->K && share >= 1);
  const size_t len = (size_t) 1 << lg;
  assert (na <= len && o0 + oc <= len);
  const int rows = batch * R->P;
  uint32_t *X = scratch (R, (size_t) rows * len);
  gpu_rns_from_z (R, X, len, a, sa, na, batch);
  gpu_rns_ntt (R, X, len, rows, lg, 0, 0);
  rns_mul (R, X, X, bprep, len, rows, len, share);
  gpu_rns_ntt (R, X, len, rows, lg, 1, 1);
  gpu_rns_to_z (R, c, sc, X + o0, len, oc, batch);
}

void
gpu_rns_conv (gpu_rns *R, uint32_t *c, size_t sc, size_t o0, size_t oc,
              const uint32_t *a, size_t sa, size_t na, const uint32_t *b,
              size_t sb, size_t nb, const uint32_t *bprep, int lg, int batch)
{
  assert (lg <= R->K);
  const size_t len = (size_t) 1 << lg;
  if (bprep != NULL)
    {
      gpu_rns_conv_prep (R, c, sc, o0, oc, a, sa, na, bprep,
                         sb == 0 ? batch : 1, lg, batch);
      return;
    }
  assert (na <= len && nb <= len && o0 + oc <= len);
  const int rows = batch * R->P;
  const int brows = sb == 0 ? R->P : rows;
  uint32_t *X = scratch (R, (size_t) (rows + brows) * len);
  uint32_t *Y = X + (size_t) rows * len;
  gpu_rns_from_z (R, X, len, a, sa, na, batch);
  gpu_rns_ntt (R, X, len, rows, lg, 0, 0);
  gpu_rns_from_z (R, Y, len, b, sb, nb, brows / R->P);
  gpu_rns_ntt (R, Y, len, brows, lg, 0, 0);
  rns_mul (R, X, X, Y, len, rows, len, brows != rows ? batch : 1);
  gpu_rns_ntt (R, X, len, rows, lg, 1, 1);
  gpu_rns_to_z (R, c, sc, X + o0, len, oc, batch);
}

void
gpu_rns_conv_monic (gpu_rns *R, uint32_t *c, size_t sc, const uint32_t *a,
                    size_t sa, const uint32_t *b, size_t sb, size_t d,
                    int batch, uint32_t *keepx, uint32_t *keepy)
{
  int lg = 0;
  while (((size_t) 1 << lg) < 2 * d)
    lg++;
  assert (((size_t) 1 << lg) == 2 * d && lg <= R->K);
  const size_t len = 2 * d;
  const int rows = batch * R->P;
  uint32_t *Z = scratch (R, (size_t) (keepx ? 1 : 2) * rows * len);
  uint32_t *X = keepx ? keepx : Z, *Y = keepx ? keepy : Z + (size_t) rows * len;
  /* x^d + A and x^d + B: their cyclic product of length 2d has the x^2d
     term at x^0 */
  from_z (R, X, len, a, sa, d, batch, d);
  from_z (R, Y, len, b, sb, d, batch, d);
  gpu_rns_ntt (R, X, len, rows, lg, 0, 0);
  gpu_rns_ntt (R, Y, len, rows, lg, 0, 0);
  rns_mul (R, Z, X, Y, len, rows, len, 1);
  gpu_rns_ntt (R, Z, len, rows, lg, 1, 1);
  to_z (R, c, sc, Z, len, len, batch, 1);
}

/* For transforms of length 2^LG: entry e of the result is the position of
   frequency -f, f the frequency at e, plus 2^31 if (-1)^f = -1.  Found
   once per length from the transform of the polynomial x, whatever order
   the transform leaves (entry e holds w^f). */
static const uint32_t *
negmap (gpu_rns *R, int lg)
{
  if ((int) R->d_negmap.size () <= lg)
    R->d_negmap.resize (lg + 1, NULL);
  if (R->d_negmap[lg] != NULL)
    return R->d_negmap[lg];
  const size_t len = (size_t) 1 << lg;
  std::vector<uint32_t> h ((size_t) R->P * len, 0);
  for (int i = 0; i < R->P; i++)
    h[(size_t) i * len + 1] = 1;
  uint32_t *d;
  CUDA_CHECK (cudaMalloc ((void **) &d, h.size () * 4));
  CUDA_CHECK (cudaMemcpy (d, h.data (), h.size () * 4, cudaMemcpyHostToDevice));
  gpu_rns_ntt (R, d, len, R->P, lg, 0, 0);
  CUDA_CHECK (cudaMemcpy (h.data (), d, len * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK (cudaFree (d));
  const uint64_t p = R->p[0];
  std::vector<std::pair<uint32_t, uint32_t> > pos (len);
  for (size_t e = 0; e < len; e++)
    pos[e] = std::make_pair ((uint32_t) (h[e] % p), (uint32_t) e);
  std::sort (pos.begin (), pos.end ());
  std::vector<uint32_t> map (len);
  for (size_t e = 0; e < len; e++)
    {
      const uint64_t w = h[e] % p, wi = powmod64 (w, p - 2, p);
      auto it = std::lower_bound (pos.begin (), pos.end (),
                                  std::make_pair ((uint32_t) wi, (uint32_t) 0));
      assert (it != pos.end () && it->first == wi);
      const bool neg = powmod64 (w, len / 2, p) != 1;
      map[e] = it->second | (neg ? 0x80000000u : 0);
    }
  R->d_negmap[lg] = upload (R, map);
  return R->d_negmap[lg];
}

/* r (2 NODES P rows) = B's transforms times the children's at -f, less the
   x^d term ((-1)^f): item 2j the right child (KY), 2j + 1 the left (KX) */
__global__ void
k_mul_corr (uint32_t *r, const uint32_t *bp, const uint32_t *kx,
            const uint32_t *ky, int lg, int P, size_t nodes,
            const uint32_t *map, const uint32_t *pr, const uint32_t *pinv)
{
  const size_t len = (size_t) 1 << lg, total = 2 * nodes * P << lg;
  for (size_t x = (size_t) blockIdx.x * blockDim.x + threadIdx.x; x < total;
       x += (size_t) gridDim.x * blockDim.x)
    {
      const size_t row = x >> lg, e = x & (len - 1);
      const size_t item = row / P, q = row % P, src = (item >> 1) * P + q;
      const uint32_t m = map[e], p = pr[q], p2 = 2 * p;
      const uint32_t *k = (item & 1) ? kx : ky;
      /* the child alone: x^d + a at -f, less (-1)^f */
      uint32_t c = k[src * len + (m & 0x7fffffffu)] + p2
                   - ((m >> 31) ? p - 1 : 1);
      c = rns_red (c, p2);
      r[x] = rns_mont (bp[src * len + e], c, p, pinv[q]);
    }
}

void
gpu_rns_corr_kept (gpu_rns *R, uint32_t *c, size_t sc, size_t oc,
                   const uint32_t *bprep, const uint32_t *keepx,
                   const uint32_t *keepy, size_t d, int nodes)
{
  int lg = 0;
  while (((size_t) 1 << lg) < 2 * d)
    lg++;
  assert (((size_t) 1 << lg) == 2 * d && lg <= R->K && oc <= d);
  const size_t len = 2 * d;
  const int rows = 2 * nodes * R->P;
  const uint32_t *map = negmap (R, lg);
  uint32_t *X = scratch (R, (size_t) rows * len);
  rns_prof.lg = lg;
  rp_begin ();
  k_mul_corr<<<flat_blocks ((size_t) rows << lg), 256>>> (
      X, bprep, keepx, keepy, lg, R->P, (size_t) nodes, map, R->d_p,
      R->d_pinv);
  CUDA_CHECK (cudaGetLastError ());
  rp_end (RP_MUL);
  gpu_rns_ntt (R, X, len, rows, lg, 1, 1);
  gpu_rns_to_z (R, c, sc, X, len, oc, 2 * nodes);
}

void
gpu_rns_reserve (gpu_rns *R, size_t words)
{
  scratch (R, words);
}

void
gpu_rns_polymul (gpu_rns *R, uint32_t *c, const uint32_t *a, size_t na,
                 const uint32_t *b, size_t nb, int batch)
{
  const size_t nc = na + nb - 1;
  int lg = 0;
  while (((size_t) 1 << lg) < nc)
    lg++;
  gpu_rns_conv (R, c, nc, 0, nc, a, na, na, b, nb, nb, NULL, lg, batch);
}
