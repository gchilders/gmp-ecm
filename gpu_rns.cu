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
 * products of two of them (< 2p) fit 32 bits, as in gpu_ntt.cuh.
 *
 * Z/N -> RNS: residue of a coefficient (limbs a_j) mod p is
 * sum_j a_j (2^(32j) mod p), accumulated in 96 bits and reduced once.
 *
 * RNS -> Z/N (explicit CRT): with M = prod p_i and M_i = M / p_i, the
 * convolution value x < M / 2 (by far) is sum_i t_i M_i - alpha M where
 * t_i = y_i M_i^-1 mod p_i and alpha = round (sum_i t_i / p_i) (fixed point,
 * units of 2^-48; the error is far below 1/2).  Mod N we compute
 * S = sum_i t_i A_i + alpha B with A_i = M_i 2^64 mod N, B = -M 2^64 mod N,
 * then (S + m N) / 2^64 with m = -S / N mod 2^64: congruent to x, and below
 * N (1 + P 2^30 / 2^64) < 2N.
 *
 * Transforms of length 2^lg: up to 2^base (base = min (K, 12)) one block
 * per row in shared memory, decimation in frequency forward (natural in,
 * bit-reversed out), in time inverse.  Longer: four-step, len = R1 R2 with
 * R2 = 2^base: transforms of length R1 down the columns (stride R2), a
 * twiddle w^(k1 n2), transforms of length R2 along the rows. */

#include "gpu_rns.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include "cudacommon.h"

#define RNS_BASE_LG 12
#define RNS_PLIM (1u << 30)

struct gpu_rns {
  int nl, K, P, base;
  std::vector<uint32_t> p;          /* host copy of the primes */
  uint32_t *d_p, *d_pinv;           /* primes, -1/p mod 2^32 */
  uint2 *d_wb, *d_wbi;              /* [P][2^(base-1)]: w_b^j, w_b^-j (Shoup) */
  uint2 *d_tlo, *d_tloi;            /* [P][2^base]: w^j (four-step, K > base) */
  uint2 *d_thi, *d_thii;            /* [P][2^(K-base)]: w^(2^base j) */
  uint2 *d_sc;                      /* [P][K+1][2]: 2^-lg (2^32 if nmul) */
  uint32_t *d_pow;                  /* [NL][P]: 2^(32 j) mod p */
  uint32_t *d_r64;                  /* [P]: 2^64 mod p */
  uint2 *d_minv;                    /* [P]: M_i^-1 mod p_i (Shoup) */
  unsigned long long *d_ip;         /* [P]: floor (2^63 / p_i) */
  uint32_t *d_A;                    /* [P][NL]: M_i 2^64 mod N */
  uint32_t *d_B;                    /* [NL]: -M 2^64 mod N */
  uint32_t *d_N;                    /* [NL] */
  unsigned long long ninv;          /* -1/N mod 2^64 */
  uint32_t *d_tmp;                  /* polymul scratch */
  size_t tmp_words;
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

/* a b / 2^32 mod p in [0, 2p), a, b < 2p */
static __device__ __forceinline__ uint32_t
rns_mont (uint32_t a, uint32_t b, uint32_t p, uint32_t pinv)
{
  uint64_t t = (uint64_t) a * b;
  uint32_t m = (uint32_t) t * pinv;
  return (uint32_t) ((t + (uint64_t) m * p) >> 32);
}

/* ---- conversions ------------------------------------------------------ */

#define RNS_CT 16   /* coefficients per thread in k_from_z */

__global__ void
k_from_z (uint32_t *res, size_t rowlen, const uint32_t *a, size_t sa,
          size_t C, int B, int nl, int P, const uint32_t *pr,
          const uint32_t *pw, const uint32_t *r64)
{
  const int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= P)
    return;
  const uint32_t p = pr[i];
  const size_t c0 = (size_t) blockIdx.x * RNS_CT;
  const size_t c1 = c0 + RNS_CT < rowlen ? c0 + RNS_CT : rowlen;
  for (int b = blockIdx.z; b < B; b += gridDim.z)
  {
  uint32_t *row = res + ((size_t) b * P + i) * rowlen;
  for (size_t c = c0; c < c1; c++)
    {
      uint32_t v = 0;
      if (c < C)
        {
          const uint32_t *ac = a + ((size_t) b * sa + c) * nl;
          uint64_t lo = 0;
          uint32_t hi = 0;
          for (int j = 0; j < nl; j++)
            {
              const uint64_t t = (uint64_t) ac[j] * pw[(size_t) j * P + i];
              lo += t;
              hi += lo < t;
            }
          v = (uint32_t) ((((uint64_t) hi * r64[i]) % p + lo % p) % p);
        }
      row[c] = v;
    }
  }
}

/* one block per (coefficient, polynomial); shared: col_lo[NL] (u64),
   col_hi[NL], s[NL + 3], t[P] */
__global__ void
k_to_z (uint32_t *a, size_t sc, const uint32_t *res, size_t rowlen, int B,
        int nl, int P, const uint32_t *pr, const uint2 *minv,
        const unsigned long long *ip, const uint32_t *A, const uint32_t *Bm,
        const uint32_t *N, unsigned long long ninv)
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
      const uint32_t y = res[((size_t) b * P + i) * rowlen + c];
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

/* ---- transforms ------------------------------------------------------- */

/* Transform sub-sequences of length 2^mlg (<= 2^base) of each row: number
   q = blockIdx.x has its elements at q * qstep + e * stride.  The twiddle of
   stage s (half size 2^s) for index k is w_b^(k 2^(base - 1 - s)). */
__global__ void
k_ntt_sub (uint32_t *res, size_t rowlen, int rows, int P, int mlg, size_t qstep,
           size_t stride, int base, const uint32_t *pr, const uint2 *wtab,
           int inv)
{
  extern __shared__ uint32_t x[];
  const int m = 1 << mlg;
  for (int row = blockIdx.y; row < rows; row += gridDim.y)
  {
  const int i = row % P;
  const uint32_t p = pr[i], p2 = 2 * p;
  const uint2 *w = wtab + ((size_t) i << (base - 1));
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
            const uint2 ww = w[k << (base - 1 - s)];
            const uint32_t a = x[i1], bb = x[i2];
            x[i1] = rns_red (a + bb, p2);
            x[i2] = rns_shoup (a - bb + p2, ww.x, ww.y, p);
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
            const uint2 ww = w[k << (base - 1 - s)];
            const uint32_t a = rns_red (x[i1], p2);
            const uint32_t t = rns_shoup (x[i2], ww.x, ww.y, p);
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
__global__ void
k_scale (uint32_t *res, size_t rowlen, int rows, int P, int lg, int K,
         int nmul, const uint32_t *pr, const uint2 *sc)
{
  for (int row = blockIdx.y; row < rows; row += gridDim.y)
  {
  const int i = row % P;
  const uint32_t p = pr[i];
  const uint2 f = sc[((size_t) i * (K + 1) + lg) * 2 + (nmul ? 1 : 0)];
  const size_t len = (size_t) 1 << lg;
  uint32_t *v = res + (size_t) row * rowlen;
  for (size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x; e < len;
       e += (size_t) gridDim.x * blockDim.x)
    v[e] = rns_red (rns_shoup (v[e], f.x, f.y, p), p);
  }
}

/* B broadcast (BBC): one polynomial's P rows for every polynomial of A */
__global__ void
k_mul (uint32_t *r, const uint32_t *a, const uint32_t *b, size_t rowlen,
       int rows, int P, size_t len, int bbc, const uint32_t *pr,
       const uint32_t *pinv)
{
  for (int row = blockIdx.y; row < rows; row += gridDim.y)
  {
  const int i = row % P;
  const uint32_t p = pr[i], pi = pinv[i];
  const size_t off = (size_t) row * rowlen;
  const size_t boff = (size_t) (bbc ? i : row) * rowlen;
  for (size_t e = (size_t) blockIdx.x * blockDim.x + threadIdx.x; e < len;
       e += (size_t) gridDim.x * blockDim.x)
    r[off + e] = rns_mont (a[off + e], b[boff + e], p, pi);
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
gpu_rns_create (const mpz_t N, int lg_want)
{
  gpu_rns *R = new gpu_rns;
  /* 4 spare bits: results below 2N, and sums of a few of them, fit */
  R->nl = (int) ((mpz_sizeinbase (N, 2) + 4 + 31) / 32);
  R->d_tmp = NULL;
  R->tmp_words = 0;

  /* the primes: the largest K <= lg_want with enough of them */
  mpz_t Pm, S, T, Bn;
  mpz_inits (Pm, S, T, Bn, NULL);
  mpz_setbit (Bn, 32 * (size_t) R->nl);
  int K;
  for (K = lg_want; K >= 1; K--)
    {
      R->p.clear ();
      mpz_set_ui (Pm, 1);
      mpz_set_ui (S, 0);
      bool enough = false;
      for (uint32_t c = (RNS_PLIM - 2) >> K; c >= 1 && !enough; c--)
        {
          const uint32_t p = (c << K) | 1;
          if (p >= RNS_PLIM || !is_prime32 (p))
            continue;
          R->p.push_back (p);
          mpz_mul_ui (Pm, Pm, p);
          mpz_add_ui (S, S, p);
          mpz_mul (T, Bn, S);
          mpz_mul (T, T, T);
          mpz_mul_2exp (T, T, K + 2);
          enough = mpz_cmp (Pm, T) > 0;
        }
      if (enough)
        break;
    }
  assert (K >= 1);
  R->K = K;
  R->P = (int) R->p.size ();
  R->base = K < RNS_BASE_LG ? K : RNS_BASE_LG;
  const int P = R->P, nl = R->nl, base = R->base;

  /* transform tables and scales */
  std::vector<uint32_t> pinv (P);
  std::vector<uint2> wb ((size_t) P << (base - 1)), wbi (wb.size ());
  std::vector<uint2> tlo ((size_t) P << base), tloi (tlo.size ());
  std::vector<uint2> thi ((size_t) P << (K - base)), thii (thi.size ());
  std::vector<uint2> sc ((size_t) P * (K + 1) * 2);
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
      uint64_t x = 1, y = 1;
      for (size_t j = 0; j < ((size_t) 1 << (base - 1)); j++)
        {
          wb[((size_t) i << (base - 1)) + j] = shoup_pair ((uint32_t) x, p);
          wbi[((size_t) i << (base - 1)) + j] = shoup_pair ((uint32_t) y, p);
          x = x * wbase % p;
          y = y * wbasei % p;
        }
      x = y = 1;
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
      const uint64_t inv2 = (p + 1) / 2;
      const uint64_t r32 = (1ull << 32) % p;
      for (int lg = 0; lg <= K; lg++)
        {
          const uint64_t f = powmod64 (inv2, lg, p);
          sc[((size_t) i * (K + 1) + lg) * 2] = shoup_pair ((uint32_t) f, p);
          sc[((size_t) i * (K + 1) + lg) * 2 + 1]
            = shoup_pair ((uint32_t) (f * r32 % p), p);
        }
    }

  /* conversion tables */
  std::vector<uint32_t> pw ((size_t) nl * P), r64 (P), A ((size_t) P * nl),
                        Bm (nl), Nl (nl);
  std::vector<uint2> minv (P);
  std::vector<unsigned long long> ip (P);
  mpz_t Mi, t, R64;
  mpz_inits (Mi, t, R64, NULL);
  mpz_setbit (R64, 64);
  for (int i = 0; i < P; i++)
    {
      const uint32_t p = R->p[i];
      uint64_t x = 1;
      for (int j = 0; j < nl; j++)
        {
          pw[(size_t) j * P + i] = (uint32_t) x;
          x = (x << 32) % p;
        }
      r64[i] = (uint32_t) mulmod64 ((1ull << 32) % p, (1ull << 32) % p, p);
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
  mpz_clears (Mi, t, R64, Pm, S, T, Bn, NULL);

  R->d_p = upload (R, R->p);
  R->d_pinv = upload (R, pinv);
  R->d_wb = upload (R, wb);
  R->d_wbi = upload (R, wbi);
  R->d_tlo = upload (R, tlo);
  R->d_tloi = upload (R, tloi);
  R->d_thi = upload (R, thi);
  R->d_thii = upload (R, thii);
  R->d_sc = upload (R, sc);
  R->d_pow = upload (R, pw);
  R->d_r64 = upload (R, r64);
  R->d_minv = upload (R, minv);
  R->d_ip = upload (R, ip);
  R->d_A = upload (R, A);
  R->d_B = upload (R, Bm);
  R->d_N = upload (R, Nl);
  return R;
}

void
gpu_rns_destroy (gpu_rns *R)
{
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

void
gpu_rns_from_z (gpu_rns *R, uint32_t *res, size_t rowlen, const uint32_t *a,
                size_t sa, size_t C, int B)
{
  const int th = 64;
  dim3 grid ((unsigned) ((rowlen + RNS_CT - 1) / RNS_CT),
             (unsigned) ((R->P + th - 1) / th), gy (B));
  k_from_z<<<grid, th>>> (res, rowlen, a, sa, C, B, R->nl, R->P, R->d_p,
                          R->d_pow, R->d_r64);
  CUDA_CHECK (cudaGetLastError ());
}

void
gpu_rns_to_z (gpu_rns *R, uint32_t *a, size_t sc, const uint32_t *res,
              size_t rowlen, size_t C, int B)
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
  dim3 grid ((unsigned) C, gy (B));
  k_to_z<<<grid, 128, smem>>> (a, sc, res, rowlen, B, R->nl, R->P, R->d_p,
                               R->d_minv, R->d_ip, R->d_A, R->d_B, R->d_N,
                               R->ninv);
  CUDA_CHECK (cudaGetLastError ());
}

static void
ntt_sub (gpu_rns *R, uint32_t *res, size_t rowlen, int rows, int mlg,
         size_t nsub, size_t qstep, size_t stride, int inv)
{
  const int half = mlg > 0 ? 1 << (mlg - 1) : 1;
  const int th = half < 32 ? 32 : (half > 256 ? 256 : half);
  dim3 grid ((unsigned) nsub, gy (rows));
  k_ntt_sub<<<grid, th, (size_t) 4 << mlg>>> (res, rowlen, rows, R->P, mlg,
                                              qstep, stride, R->base, R->d_p,
                                              inv ? R->d_wbi : R->d_wb, inv);
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
  const int base = R->base;
  const size_t len = (size_t) 1 << lg;
  if (lg <= base)
    ntt_sub (R, res, rowlen, rows, lg, 1, 0, 1, inv);
  else
    {
      const int r1lg = lg - base;
      const size_t R1 = (size_t) 1 << r1lg, R2 = (size_t) 1 << base;
      if (!inv)
        {
          ntt_sub (R, res, rowlen, rows, r1lg, R2, 1, R2, 0);       /* columns */
          k_twiddle<<<dim3 (gx (len), gy (rows)), 256>>> (
              res, rowlen, rows, R->P, lg, r1lg, base, R->K, R->d_p, R->d_tlo,
              R->d_thi);
          CUDA_CHECK (cudaGetLastError ());
          ntt_sub (R, res, rowlen, rows, base, R1, R2, 1, 0);       /* rows */
        }
      else
        {
          ntt_sub (R, res, rowlen, rows, base, R1, R2, 1, 1);
          k_twiddle<<<dim3 (gx (len), gy (rows)), 256>>> (
              res, rowlen, rows, R->P, lg, r1lg, base, R->K, R->d_p,
              R->d_tloi, R->d_thii);
          CUDA_CHECK (cudaGetLastError ());
          ntt_sub (R, res, rowlen, rows, r1lg, R2, 1, R2, 1);
        }
    }
  if (inv)
    {
      k_scale<<<dim3 (gx (len), gy (rows)), 256>>> (res, rowlen, rows, R->P,
                                                    lg, R->K, nmul, R->d_p,
                                                    R->d_sc);
      CUDA_CHECK (cudaGetLastError ());
    }
}

void
gpu_rns_mul (gpu_rns *R, uint32_t *r, const uint32_t *a, const uint32_t *b,
             size_t rowlen, int rows, size_t len, int bbc)
{
  k_mul<<<dim3 (gx (len), gy (rows)), 256>>> (r, a, b, rowlen, rows, R->P,
                                              len, bbc, R->d_p, R->d_pinv);
  CUDA_CHECK (cudaGetLastError ());
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
gpu_rns_prep (gpu_rns *R, uint32_t *out, const uint32_t *b, size_t nb, int lg)
{
  const size_t len = (size_t) 1 << lg;
  gpu_rns_from_z (R, out, len, b, nb, nb, 1);
  gpu_rns_ntt (R, out, len, R->P, lg, 0, 0);
}

void
gpu_rns_conv (gpu_rns *R, uint32_t *c, size_t sc, size_t o0, size_t oc,
              const uint32_t *a, size_t sa, size_t na, const uint32_t *b,
              size_t sb, size_t nb, const uint32_t *bprep, int lg, int batch)
{
  assert (lg <= R->K);
  const size_t len = (size_t) 1 << lg;
  assert (na <= len && nb <= len && o0 + oc <= len);
  const int rows = batch * R->P;
  const int brows = bprep != NULL || sb == 0 ? R->P : rows;
  uint32_t *X = scratch (R, (size_t) (rows + (bprep ? 0 : brows)) * len);
  uint32_t *Y = bprep ? (uint32_t *) bprep : X + (size_t) rows * len;
  gpu_rns_from_z (R, X, len, a, sa, na, batch);
  gpu_rns_ntt (R, X, len, rows, lg, 0, 0);
  if (bprep == NULL)
    {
      gpu_rns_from_z (R, Y, len, b, sb, nb, brows / R->P);
      gpu_rns_ntt (R, Y, len, brows, lg, 0, 0);
    }
  gpu_rns_mul (R, X, X, Y, len, rows, len, brows != rows);
  gpu_rns_ntt (R, X, len, rows, lg, 1, 1);
  gpu_rns_to_z (R, c, sc, X + o0, len, oc, batch);
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
