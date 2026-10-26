/* gpu_nmul.cuh: products mod a general N by number-theoretic transforms
   of the numbers themselves (prime95 style), for large N.

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

/* A number is D = 2^LG signed int32 digits of b bits, x = sum x_i B^i
 * (B = 2^b), in global memory, D words.  Digits are not normalised: they
 * stay below about 2^(b+1.1) in magnitude (sums of three numbers about
 * 2^(b+2.7)), so a convolution coefficient is below D 2^(2b+4.8) <=
 * 2^58, under p0 p1 / 2 = 2^58.6 for the two primes p < 2^30 of the
 * transforms (2^23 | p - 1), and Garner recovers it with its sign.
 * Hence b <= (53 - LG) / 2; and the one-step carry below needs
 * b >= LG + 4.  A thread group of D / 8 threads (one block) works on one
 * product at a time; thread TID owns digits TID + t D/8 (t < 8) of every
 * number ("column layout"), which is what the first forward and the last
 * inverse transform pass use, so digit vectors never meet another
 * thread's except inside the transforms (shared memory) and in the
 * normalisation.  Values in global memory are therefore read and written
 * by their owner only, and need no barriers.
 *
 * Montgomery products with R = B^D - 1 (gpu_dev/ntt_modmul_bench.cu):
 *   T_c = a b mod (B^D - 1) (cyclic),  T_n = a b mod (B^D + 1) (negacyclic),
 *   q = T_c N' mod (B^D - 1), N' = -1/N mod (B^D - 1)    (cyclic),
 *   r = (T_n + q N mod (B^D + 1)) / -2 mod (B^D + 1)     (negacyclic),
 * all transforms of length D with no zero padding; r = a b / R mod N with
 * the representative picked from the top digit.  With B^D >= 2^10 N and
 * |a|, |b| below 8N, r is in (-0.6 N, 2.6 N).  Needs gcd (N, B^D - 1) = 1.
 *
 * A "hat" is a number's four transforms (cyclic mod p0, p1, negacyclic mod
 * p0, p1; 4 D words, bit-reversed order), reusable in several products:
 * a product costs 8 transforms per prime with one hat given, 6 with both
 * (10 with neither).  Transforms run in shared memory (2 padded buffers,
 * nm_smem_bytes) as radix-8 passes in registers, both primes per pass: DIF
 * forward to bit-reversed order, DIT inverse back, Shoup twiddles, lazy
 * [0, 2p) values, Montgomery pointwise products with 2^-32 folded into the
 * scaling.
 *
 * All nm_* group operations must be called by all threads of the block
 * (blockDim.x = D / 8) and end with __syncthreads where shared memory is
 * reused. */

#ifndef _GPU_NMUL_CUH
#define _GPU_NMUL_CUH 1

#include <stdint.h>

#define NM_P0 998244353u    /* 119 * 2^23 + 1 */
#define NM_P1 897581057u    /* 107 * 2^23 + 1 */
#define NM_MIN_LG 8
#define NM_MAX_LG 12
#define NM_SPARE_BITS 10    /* B^D >= 2^10 N */
#define NM_PAD(i) ((i) + ((i) >> 5))
#define NM_PADDED(n) ((n) + ((n) >> 5))

/* digit bits allowed for transform length 2^LG */
#define NM_BMIN(lg) ((lg) + 4)
#define NM_BMAX(lg) ((53 - (lg)) / 2)

struct nm_prime {
  uint32_t p, pinv;          /* -1/p mod 2^32 */
  uint32_t sc, sc_sh;        /* 2^32 / D mod p and its Shoup constant */
  const uint2 *w, *wi;       /* w^k, w^-k (Shoup pairs), k < D/2 */
  const uint2 *psi;          /* psi^i, i < D (negacyclic input weights) */
  const uint2 *ipsi;         /* psi^-i 2^32 / D (negacyclic output) */
};

struct nm_ctx {
  nm_prime pr[2];
  uint32_t g, g_sh;          /* 1/p0 mod p1 */
  uint64_t p01;
  int lg, b, D;
  const uint32_t *Nph;       /* hat of N' (its cyclic parts are used) */
  const uint32_t *Nh;        /* hat of N (negacyclic parts used) */
  const uint32_t *hone;      /* hat of the integer 1: x / R */
  const uint32_t *hR2;       /* hat of R^2 mod N: x R */
  const int32_t *N;          /* N, canonical digits */
  const int32_t *one;        /* R mod N, canonical digits */
  int ntop;                  /* N's top nonzero digit */
  double nhi;                /* N / B^(ntop - 3), from its top 4 digits */
};

/* shared memory per block */
static inline size_t
nm_smem_bytes (int lg)
{
  return (size_t) 2 * NM_PADDED ((size_t) 1 << lg) * 4;
}

#ifdef __CUDACC__

#define NM_DEV __device__ __forceinline__

/* The kernels take their nm_ctx by value and use it by reference:
   __grid_constant__ (compute capability 7.0 and up) spares each thread a
   copy of it.  A build that also targets older GPUs (GPU_MIN_CC, the
   lowest architecture configure chose, below 70) goes without. */
#if defined (GPU_MIN_CC) && GPU_MIN_CC < 70
#define NM_GRID_CONST
#else
#define NM_GRID_CONST __grid_constant__
#endif

extern __shared__ uint32_t nm_sm[];

NM_DEV uint32_t nm_shoup (uint32_t a, uint32_t w, uint32_t wsh, uint32_t p)
{
  uint32_t q = __umulhi (a, wsh);
  return a * w - q * p;      /* [0, 2p) */
}
NM_DEV uint32_t nm_red (uint32_t x, uint32_t m) { return x >= m ? x - m : x; }
NM_DEV uint32_t nm_mont (uint32_t a, uint32_t b, uint32_t p, uint32_t pinv)
{
  uint64_t t = (uint64_t) a * b;
  uint32_t m = (uint32_t) t * pinv;
  return (uint32_t) ((t + (uint64_t) m * p) >> 32);
}

/* this thread's digits of number X (column layout) */
template<int LG>
NM_DEV void
nm_load (int32_t (&v)[8], const int32_t *x)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    v[t] = x[threadIdx.x + t * TPG];
}

template<int LG>
NM_DEV void
nm_store (int32_t *x, const int32_t (&v)[8])
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    x[threadIdx.x + t * TPG] = v[t];
}

/* elementwise: r = a + b, a - b, -a, a; on this thread's digits only */
template<int LG>
NM_DEV void
nm_add (int32_t *r, const int32_t *a, const int32_t *b)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    r[threadIdx.x + t * TPG] = a[threadIdx.x + t * TPG] + b[threadIdx.x + t * TPG];
}

template<int LG>
NM_DEV void
nm_sub (int32_t *r, const int32_t *a, const int32_t *b)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    r[threadIdx.x + t * TPG] = a[threadIdx.x + t * TPG] - b[threadIdx.x + t * TPG];
}

template<int LG>
NM_DEV void
nm_neg (int32_t *r, const int32_t *a)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    r[threadIdx.x + t * TPG] = -a[threadIdx.x + t * TPG];
}

template<int LG>
NM_DEV void
nm_copy (int32_t *r, const int32_t *a)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int t = 0; t < 8; t++)
    r[threadIdx.x + t * TPG] = a[threadIdx.x + t * TPG];
}

/* ---- transforms ------------------------------------------------------- */

/* forward DIF, first pass (stages LG-1 .. LG-3), from digits in registers */
template<int LG, bool NEG>
NM_DEV void
nm_first (const nm_ctx &c, const int32_t (&dg)[8])
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      uint32_t *x = nm_sm + j * NM_PADDED (D);
      const uint32_t p = P.p, p2 = 2 * p;
      uint32_t v[8];
#pragma unroll
      for (int t = 0; t < 8; t++)
        {
          const uint32_t xm = dg[t] < 0 ? (uint32_t) (dg[t] + (int32_t) p)
                                        : (uint32_t) dg[t];
          if (NEG)
            {
              const uint2 w = P.psi[tid + t * TPG];
              v[t] = nm_shoup (xm, w.x, w.y, p);
            }
          else
            v[t] = xm;
        }
#pragma unroll
      for (int r = 0; r < 3; r++)
        {
          const int hs = 4 >> r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = P.w[(tid + (t & (hs - 1)) * TPG) << r];
                const uint32_t a = v[t], b = v[t + hs];
                v[t] = nm_red (a + b, p2);
                v[t + hs] = nm_shoup (a - b + p2, w.x, w.y, p);
              }
        }
#pragma unroll
      for (int t = 0; t < 8; t++)
        x[NM_PAD (tid + t * TPG)] = v[t];
    }
  __syncthreads ();
}

/* forward DIF stages s .. s-R+1 on both primes */
template<int LG, int R>
NM_DEV void
nm_fpass (const nm_ctx &c, int s)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int lq = s - R + 1, q = 1 << lq;
  for (int g = threadIdx.x; g < (D >> R); g += TPG)
    {
      const int k = g & (q - 1);
      const int base = ((g >> lq) << (s + 1)) + k;
#pragma unroll
      for (int j = 0; j < 2; j++)
        {
          const nm_prime &P = c.pr[j];
          uint32_t *x = nm_sm + j * NM_PADDED (D);
          const uint32_t p = P.p, p2 = 2 * p;
          uint32_t v[1 << R];
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            v[t] = x[NM_PAD (base + t * q)];
#pragma unroll
          for (int r = 0; r < R; r++)
            {
              const int hs = 1 << (R - 1 - r), sh = LG - 1 - (s - r);
#pragma unroll
              for (int t = 0; t < (1 << R); t++)
                if (!(t & hs))
                  {
                    const uint2 w = P.w[(k + (t & (hs - 1)) * q) << sh];
                    const uint32_t a = v[t], b = v[t + hs];
                    v[t] = nm_red (a + b, p2);
                    v[t + hs] = nm_shoup (a - b + p2, w.x, w.y, p);
                  }
            }
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            x[NM_PAD (base + t * q)] = v[t];
        }
    }
  __syncthreads ();
}

/* inverse DIT stages s0 .. s0+R-1 on both primes */
template<int LG, int R>
NM_DEV void
nm_ipass (const nm_ctx &c, int s0)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int q = 1 << s0;
  for (int g = threadIdx.x; g < (D >> R); g += TPG)
    {
      const int k = g & (q - 1);
      const int base = ((g >> s0) << (s0 + R)) + k;
#pragma unroll
      for (int j = 0; j < 2; j++)
        {
          const nm_prime &P = c.pr[j];
          uint32_t *x = nm_sm + j * NM_PADDED (D);
          const uint32_t p = P.p, p2 = 2 * p;
          uint32_t v[1 << R];
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            v[t] = x[NM_PAD (base + t * q)];
#pragma unroll
          for (int r = 0; r < R; r++)
            {
              const int hs = 1 << r, sh = LG - 1 - (s0 + r);
#pragma unroll
              for (int t = 0; t < (1 << R); t++)
                if (!(t & hs))
                  {
                    const uint2 w = P.wi[(k + (t & (hs - 1)) * q) << sh];
                    const uint32_t a = nm_red (v[t], p2);
                    const uint32_t b = nm_shoup (v[t + hs], w.x, w.y, p);
                    v[t] = nm_red (a + b, p2);
                    v[t + hs] = nm_red (a - b + p2, p2);
                  }
            }
#pragma unroll
          for (int t = 0; t < (1 << R); t++)
            x[NM_PAD (base + t * q)] = v[t];
        }
    }
  __syncthreads ();
}

/* the forward transform of DG up to (not including) its last three stages */
template<int LG, bool NEG>
NM_DEV void
nm_fwd_head (const nm_ctx &c, const int32_t (&dg)[8])
{
  constexpr int M = LG - 6;      /* stages between the first pass and mid */
  nm_first<LG, NEG> (c, dg);
  int s = LG - 4;
  if constexpr (M % 3 == 1)
    { nm_fpass<LG, 1> (c, s); s -= 1; }
  if constexpr (M % 3 == 2)
    { nm_fpass<LG, 2> (c, s); s -= 2; }
  for (; s >= 5; s -= 3)
    nm_fpass<LG, 3> (c, s);
}

/* the mid pass, on contiguous groups of 8, both primes; H is a hat part
   (H, H + D for the two primes):
     SAVE: the last three forward stages, written to H; stop;
     MUL:  the last three forward stages, times H;
     HH:   A (another hat part) times H, no forward stages;
     ACC:  ACC (this thread's values, [prime][8]), no forward stages;
   then (not SAVE) the first three inverse stages into shared memory. */
enum { NM_SAVE, NM_MUL, NM_HH, NM_ACC };

template<int LG, int MODE>
NM_DEV void
nm_mid (const nm_ctx &c, uint32_t *H, const uint32_t *A,
        const uint32_t (*acc)[8])
{
  constexpr int D = 1 << LG;
  const int base = threadIdx.x << 3;
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      uint32_t *x = nm_sm + j * NM_PADDED (D);
      const uint32_t p = P.p, p2 = 2 * p;
      uint32_t v[8];
      if (MODE == NM_SAVE || MODE == NM_MUL)
        {
#pragma unroll
          for (int t = 0; t < 8; t++)
            v[t] = x[NM_PAD (base + t)];
#pragma unroll
          for (int r = 0; r < 3; r++)
            {
              const int hs = 4 >> r, sh = LG - 3 + r;
#pragma unroll
              for (int t = 0; t < 8; t++)
                if (!(t & hs))
                  {
                    const uint2 w = P.w[(t & (hs - 1)) << sh];
                    const uint32_t a = v[t], b = v[t + hs];
                    v[t] = nm_red (a + b, p2);
                    v[t + hs] = nm_shoup (a - b + p2, w.x, w.y, p);
                  }
            }
        }
      else if (MODE == NM_HH)
        {
          const uint4 *A4 = (const uint4 *) (A + j * D + base);
          const uint4 a0 = A4[0], a1 = A4[1];
          v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w;
          v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
        }
      else
        {
#pragma unroll
          for (int t = 0; t < 8; t++)
            v[t] = acc[j][t];
        }
      uint4 *B4 = (uint4 *) (H + j * D + base);
      if (MODE == NM_SAVE)
        {
          B4[0] = make_uint4 (v[0], v[1], v[2], v[3]);
          B4[1] = make_uint4 (v[4], v[5], v[6], v[7]);
          continue;
        }
      if (MODE != NM_ACC)
        {
          const uint4 b0 = B4[0], b1 = B4[1];
          v[0] = nm_mont (v[0], b0.x, p, P.pinv);
          v[1] = nm_mont (v[1], b0.y, p, P.pinv);
          v[2] = nm_mont (v[2], b0.z, p, P.pinv);
          v[3] = nm_mont (v[3], b0.w, p, P.pinv);
          v[4] = nm_mont (v[4], b1.x, p, P.pinv);
          v[5] = nm_mont (v[5], b1.y, p, P.pinv);
          v[6] = nm_mont (v[6], b1.z, p, P.pinv);
          v[7] = nm_mont (v[7], b1.w, p, P.pinv);
        }
#pragma unroll
      for (int r = 0; r < 3; r++)
        {
          const int hs = 1 << r, sh = LG - 1 - r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = P.wi[(t & (hs - 1)) << sh];
                const uint32_t a = nm_red (v[t], p2);
                const uint32_t b = nm_shoup (v[t + hs], w.x, w.y, p);
                v[t] = nm_red (a + b, p2);
                v[t + hs] = nm_red (a - b + p2, p2);
              }
        }
#pragma unroll
      for (int t = 0; t < 8; t++)
        x[NM_PAD (base + t)] = v[t];
    }
  __syncthreads ();
}

/* the rest of the inverse transform; the last pass (stages LG-3 .. LG-1)
   scaled (NEG: with the negacyclic weights), then the CRT: CO = this
   thread's signed coefficients */
template<int LG, bool NEG>
NM_DEV void
nm_inv_tail (const nm_ctx &c, int64_t (&co)[8])
{
  constexpr int M = LG - 6, D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x;
  int s0 = 3;
  for (int k = 0; k < M / 3; k++, s0 += 3)
    nm_ipass<LG, 3> (c, s0);
  if constexpr (M % 3 == 1)
    nm_ipass<LG, 1> (c, s0);
  if constexpr (M % 3 == 2)
    nm_ipass<LG, 2> (c, s0);
  uint32_t res[2][8];
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      const uint32_t *x = nm_sm + j * NM_PADDED (D);
      const uint32_t p = P.p, p2 = 2 * p;
      uint32_t *v = res[j];
#pragma unroll
      for (int t = 0; t < 8; t++)
        v[t] = x[NM_PAD (tid + t * TPG)];
#pragma unroll
      for (int r = 0; r < 3; r++)
        {
          const int hs = 1 << r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = P.wi[(tid + (t & (hs - 1)) * TPG) << (2 - r)];
                const uint32_t a = nm_red (v[t], p2);
                const uint32_t b = nm_shoup (v[t + hs], w.x, w.y, p);
                v[t] = nm_red (a + b, p2);
                v[t + hs] = nm_red (a - b + p2, p2);
              }
        }
#pragma unroll
      for (int t = 0; t < 8; t++)
        if (NEG)
          {
            const uint2 w = P.ipsi[tid + t * TPG];
            v[t] = nm_red (nm_shoup (v[t], w.x, w.y, p), p);
          }
        else
          v[t] = nm_red (nm_shoup (v[t], P.sc, P.sc_sh, p), p);
    }
  const uint32_t p0 = c.pr[0].p, p1 = c.pr[1].p;
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      uint32_t u = res[1][t] + 2 * p1 - res[0][t];
      u = nm_red (nm_shoup (u, c.g, c.g_sh, p1), p1);
      const uint64_t v = res[0][t] + (uint64_t) p0 * u;
      co[t] = v > c.p01 / 2 ? (int64_t) (v - c.p01) : (int64_t) v;
    }
  __syncthreads ();
}

/* coefficients -> digits, one carry step: c_i = lo + mid B + hi B^2,
   digit_i = lo_i + mid_(i-1) + hi_(i-2), wrapping with sign SGN (cyclic
   +1, negacyclic -1).  Shared memory as scratch. */
template<int LG>
NM_DEV void
nm_carry (const nm_ctx &c, const int64_t (&co)[8], int32_t (&dg)[8], int sgn)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x, b = c.b;
  int32_t *Mi = (int32_t *) nm_sm, *Hi = Mi + NM_PADDED (D);
  const int32_t mask = (1 << b) - 1;
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      dg[t] = (int32_t) (co[t] & mask);
      Mi[tid + t * TPG] = (int32_t) ((co[t] >> b) & mask);
      Hi[tid + t * TPG] = (int32_t) (co[t] >> (2 * b));
    }
  __syncthreads ();
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      const int i = tid + t * TPG;
      const int32_t m = i >= 1 ? Mi[i - 1] : sgn * Mi[D - 1];
      const int32_t h = i >= 2 ? Hi[i - 2] : sgn * Hi[D - 2 + i];
      dg[t] += m + h;
    }
  __syncthreads ();
}

/* cyclic (NEG = false) or negacyclic convolution of DG with hat H */
template<int LG, bool NEG>
NM_DEV void
nm_conv (const nm_ctx &c, const int32_t (&dg)[8], const uint32_t *H,
         int32_t (&out)[8])
{
  int64_t co[8];
  nm_fwd_head<LG, NEG> (c, dg);
  nm_mid<LG, NM_MUL> (c, (uint32_t *) H, NULL, NULL);
  nm_inv_tail<LG, NEG> (c, co);
  nm_carry<LG> (c, co, out, NEG ? -1 : 1);
}

/* x = REDC (T): from T mod (B^D - 1) and T mod (B^D + 1).  ADDN: digits
   added to T_n's coefficients before (q N's carry). */
template<int LG>
NM_DEV void
nm_redc (const nm_ctx &c, const int32_t (&Tc)[8], const int32_t (&Tn)[8],
         int32_t (&x)[8])
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x, b = c.b, B = 1 << b;
  int32_t q[8], s[8];
  int64_t co[8];
  nm_conv<LG, false> (c, Tc, c.Nph, q);
  nm_fwd_head<LG, true> (c, q);
  nm_mid<LG, NM_MUL> (c, (uint32_t *) c.Nh + 2 * D, NULL, NULL);
  nm_inv_tail<LG, true> (c, co);
#pragma unroll
  for (int t = 0; t < 8; t++)
    co[t] += Tn[t];
  nm_carry<LG> (c, co, s, -1);
  /* r = S / -2 mod B^D + 1: U = S (+ B^D + 1 if odd), H = U / 2, x = -H */
  int32_t *Si = (int32_t *) nm_sm;
#pragma unroll
  for (int t = 0; t < 8; t++)
    Si[tid + t * TPG] = s[t];
  __syncthreads ();
  const bool odd = Si[0] & 1;
  int32_t nx[8];
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      const int i = tid + t * TPG;
      nx[t] = i < D - 1 ? Si[i + 1] : 0;
    }
  if (odd)
    {
      if (tid == 0)
        s[0] += 1;
      if (tid == TPG - 1)
        s[7] += B;
    }
#pragma unroll
  for (int t = 0; t < 8; t++)
    x[t] = -((s[t] >> 1) + ((nx[t] & 1) << (b - 1)));
  /* + k (B^D + 1) to land in [-B^D / 8, B^D / 2): k from the top digit */
  int32_t *top = (int32_t *) (nm_sm + NM_PADDED (D));
  if (tid == TPG - 1)
    top[0] = x[7];
  __syncthreads ();
  const int32_t y = top[0];
  const int32_t kk = ((5 * B / 8) - y) >> b;
  __syncthreads ();
  if (tid == 0)
    x[0] += kk;
  if (tid == TPG - 1)
    x[7] += kk * B;
}

/* ---- group operations (one block) --------------------------------------- */

/* H = the hat of A */
template<int LG>
__device__ __noinline__ void
nm_hat (const nm_ctx &c, uint32_t *H, const int32_t *A)
{
  constexpr int D = 1 << LG;
  int32_t a[8];
  nm_load<LG> (a, A);
  nm_fwd_head<LG, false> (c, a);
  nm_mid<LG, NM_SAVE> (c, H, NULL, NULL);
  nm_fwd_head<LG, true> (c, a);
  nm_mid<LG, NM_SAVE> (c, H + 2 * D, NULL, NULL);
}

/* R = A B / R (B given by its hat); R may alias A */
template<int LG>
__device__ __noinline__ void
nm_mul (const nm_ctx &c, int32_t *R, const int32_t *A, const uint32_t *HB)
{
  constexpr int D = 1 << LG;
  int32_t a[8], Tc[8], Tn[8], x[8];
  nm_load<LG> (a, A);
  nm_conv<LG, false> (c, a, HB, Tc);
  nm_conv<LG, true> (c, a, HB + 2 * D, Tn);
  nm_redc<LG> (c, Tc, Tn, x);
  nm_store<LG> (R, x);
}

/* R = A B / R from both hats */
template<int LG>
__device__ __noinline__ void
nm_mulhh (const nm_ctx &c, int32_t *R, const uint32_t *HA, const uint32_t *HB)
{
  constexpr int D = 1 << LG;
  int32_t Tc[8], Tn[8], x[8];
  int64_t co[8];
  nm_mid<LG, NM_HH> (c, (uint32_t *) HB, HA, NULL);
  nm_inv_tail<LG, false> (c, co);
  nm_carry<LG> (c, co, Tc, 1);
  nm_mid<LG, NM_HH> (c, (uint32_t *) HB + 2 * D, HA + 2 * D, NULL);
  nm_inv_tail<LG, true> (c, co);
  nm_carry<LG> (c, co, Tn, -1);
  nm_redc<LG> (c, Tc, Tn, x);
  nm_store<LG> (R, x);
}

/* the hat parts of this thread (for sums of products in the transform
   domain): ACC[part][prime][8], part 0 cyclic, 1 negacyclic */
template<int LG>
NM_DEV void
nm_acc_clear (uint32_t (&acc)[2][2][8])
{
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int t = 0; t < 8; t++)
        acc[h][j][t] = 0;
}

/* ACC += HA HB pointwise (canonical mod p) */
template<int LG>
NM_DEV void
nm_acc_add (const nm_ctx &c, uint32_t (&acc)[2][2][8], const uint32_t *HA,
            const uint32_t *HB)
{
  constexpr int D = 1 << LG;
  const int base = threadIdx.x << 3;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int j = 0; j < 2; j++)
      {
        const nm_prime &P = c.pr[j];
        const uint4 *A4 = (const uint4 *) (HA + (2 * h + j) * D + base);
        const uint4 *B4 = (const uint4 *) (HB + (2 * h + j) * D + base);
        const uint4 a0 = A4[0], a1 = A4[1], b0 = B4[0], b1 = B4[1];
        const uint32_t av[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
        const uint32_t bv[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
#pragma unroll
        for (int t = 0; t < 8; t++)
          acc[h][j][t] = nm_red (acc[h][j][t]
                                 + nm_red (nm_mont (av[t], bv[t], P.p, P.pinv), P.p), P.p);
      }
}

/* R = REDC (sum of the products in ACC) */
template<int LG>
NM_DEV void
nm_acc_redc (const nm_ctx &c, int32_t *R, const uint32_t (&acc)[2][2][8])
{
  int32_t Tc[8], Tn[8], x[8];
  int64_t co[8];
  nm_mid<LG, NM_ACC> (c, NULL, NULL, acc[0]);
  nm_inv_tail<LG, false> (c, co);
  nm_carry<LG> (c, co, Tc, 1);
  nm_mid<LG, NM_ACC> (c, NULL, NULL, acc[1]);
  nm_inv_tail<LG, true> (c, co);
  nm_carry<LG> (c, co, Tn, -1);
  nm_redc<LG> (c, Tc, Tn, x);
  nm_store<LG> (R, x);
}

/* ---- conversions -------------------------------------------------------- */

/* X (canonical digits) = the plain number of NL limbs at IN */
template<int LG>
__device__ __noinline__ void
nm_from_limbs (const nm_ctx &c, int32_t *X, const uint32_t *in, uint32_t nl)
{
  constexpr int TPG = (1 << LG) / 8;
  const int b = c.b;
  const uint32_t mask = (1u << b) - 1;
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      const uint32_t i = threadIdx.x + t * TPG, bit = i * b;
      const uint32_t w = bit >> 5, s = bit & 31;
      const uint64_t lo = w < nl ? in[w] : 0, hi = w + 1 < nl ? in[w + 1] : 0;
      X[i] = (int32_t) ((uint32_t) (((hi << 32) | lo) >> s) & mask);
    }
}

/* Normalise the digits in shared memory (int32, natural order, D of them)
   to [0, B), carrying out of the top into the returned value (the number
   is sum d_i B^i + top B^D).  Thread TID takes digits 8 TID .. 8 TID + 7;
   rounds of carries between threads until none moves. */
template<int LG>
NM_DEV int32_t
nm_normalise (const nm_ctx &c)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x, b = c.b;
  int32_t *d = (int32_t *) nm_sm, *cx = (int32_t *) (nm_sm + NM_PADDED (D));
  const int32_t mask = (1 << b) - 1;
  int32_t v[8], top = 0;
  int64_t cr = 0;
#pragma unroll
  for (int k = 0; k < 8; k++)
    {
      const int64_t s = (int64_t) d[8 * tid + k] + cr;
      v[k] = (int32_t) (s & mask);
      cr = s >> b;
    }
  for (;;)
    {
      cx[tid] = (int32_t) cr;
      __syncthreads ();
      int32_t cin = tid ? cx[tid - 1] : 0;
      if (tid == TPG - 1)
        top += (int32_t) cr;
      __syncthreads ();
      cr = 0;
      if (cin != 0)
        {
          int32_t cc = cin;
#pragma unroll
          for (int k = 0; k < 8; k++)
            {
              const int32_t s = v[k] + cc;
              v[k] = s & mask;
              cc = s >> b;
            }
          cr = cc;
        }
      if (!__syncthreads_or (cr != 0))
        break;
    }
#pragma unroll
  for (int k = 0; k < 8; k++)
    d[8 * tid + k] = v[k];
  if (tid == TPG - 1)
    cx[0] = top;
  __syncthreads ();
  top = cx[0];
  __syncthreads ();
  return top;
}

/* shared digits += K N (canonical N; K may be negative), this thread's
   chunk */
template<int LG>
NM_DEV void
nm_addmul_N (const nm_ctx &c, int32_t k)
{
  int32_t *d = (int32_t *) nm_sm;
  const int tid = threadIdx.x;
#pragma unroll
  for (int t = 0; t < 8; t++)
    d[8 * tid + t] += k * c.N[8 * tid + t];
  __syncthreads ();
}

/* OUT (NL limbs) = X reduced to [0, N + small) (below 2N), X's digits in
   registers (column layout), X in (-4N, 12N) */
template<int LG>
__device__ __noinline__ void
nm_to_limbs (const nm_ctx &c, uint32_t *out, uint32_t nl,
             const int32_t (&x)[8])
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int tid = threadIdx.x, b = c.b;
  int32_t *d = (int32_t *) nm_sm;
#pragma unroll
  for (int t = 0; t < 8; t++)
    d[tid + t * TPG] = x[t];
  __syncthreads ();
  int32_t top = nm_normalise<LG> (c);
  if (top < 0)
    {
      nm_addmul_N<LG> (c, 4);
      top = nm_normalise<LG> (c);
    }
  /* quotient estimate from the top digits */
  int32_t *kx = (int32_t *) (nm_sm + NM_PADDED (D));
  if (tid == 0)
    {
      /* vh = X / B^(ntop - 3), digits below ignored (X is now in [0, 16N),
         so top = 0 and digits above ntop + 2 are zero) */
      double vh = 0;
      for (int i = min (D - 1, c.ntop + 3); i >= c.ntop - 3; i--)
        vh = vh * (double) (1 << b) + d[i];
      kx[0] = (int32_t) floor (vh / c.nhi);
    }
  __syncthreads ();
  const int32_t k = kx[0];
  __syncthreads ();
  if (k != 0)
    {
      nm_addmul_N<LG> (c, -k);
      top = nm_normalise<LG> (c);
    }
  if (top < 0)
    {
      nm_addmul_N<LG> (c, 1);
      top = nm_normalise<LG> (c);
    }
  /* pack: limb j has bits [32 j, 32 j + 32) */
  for (uint32_t j = tid; j < nl; j += TPG)
    {
      const uint32_t i0 = 32 * j / b;
      int pos = (int) (i0 * b) - (int) (32 * j);
      uint64_t v = 0;
      for (uint32_t i = i0; pos < 32 && i < (uint32_t) D; i++, pos += b)
        v |= pos < 0 ? (uint64_t) ((uint32_t) d[i] >> -pos)
                     : (uint64_t) (uint32_t) d[i] << pos;
      out[j] = (uint32_t) v;
    }
  __syncthreads ();
}

#endif /* __CUDACC__ */

/* ---- host: the tables for N --------------------------------------------- */

#ifdef __CUDACC__
#include <vector>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <gmp.h>

/* transform length and digit bits for an N of BITS bits, from LG up to
   LGMAX: false if none (b d >= bits + NM_SPARE_BITS, NM_BMIN <= b <=
   NM_BMAX); the smallest LG that fits */
static inline bool
nm_shape (size_t bits, int lgmin, int lgmax, int *lg, int *b)
{
  for (int l = std::max (lgmin, NM_MIN_LG); l <= std::min (lgmax, NM_MAX_LG); l++)
    {
      const size_t D = (size_t) 1 << l;
      int bb = (int) ((bits + NM_SPARE_BITS + D - 1) / D);
      bb = std::max (bb, NM_BMIN (l));
      if (bb <= NM_BMAX (l))
        {
          *lg = l;
          *b = bb;
          return true;
        }
    }
  return false;
}

static inline uint32_t
nm_powmod (uint64_t a, uint64_t e, uint32_t p)
{
  uint64_t r = 1;
  a %= p;
  while (e)
    {
      if (e & 1)
        r = r * a % p;
      a = a * a % p;
      e >>= 1;
    }
  return (uint32_t) r;
}

static inline uint2
nm_shoup_pair (uint32_t w, uint32_t p)
{
  return make_uint2 (w, (uint32_t) (((uint64_t) w << 32) / p));
}

/* a primitive 2^k-th root of unity mod p */
static inline uint32_t
nm_root2k (uint32_t p, int k)
{
  for (uint32_t g = 2;; g++)
    {
      uint32_t m = p - 1, f = 2, ok = 1;
      while (m > 1 && ok)
        {
          if (m % f == 0)
            {
              if (nm_powmod (g, (p - 1) / f, p) == 1)
                ok = 0;
              while (m % f == 0)
                m /= f;
            }
          f++;
        }
      if (ok)
        return nm_powmod (g, (p - 1) >> k, p);
    }
}

/* canonical digits of X >= 0 (below B^D) */
static inline void
nm_digits (std::vector<int32_t> &dg, const mpz_t x, int D, int b)
{
  dg.assign (D, 0);
  const size_t nb = mpz_sizeinbase (x, 2);
  for (int i = 0; i < D; i++)
    {
      uint32_t v = 0;
      for (int k = 0; k < b; k++)
        if ((size_t) (i * b + k) < nb && mpz_tstbit (x, (mp_bitcnt_t) i * b + k))
          v |= 1u << k;
      dg[i] = (int32_t) v;
    }
}

/* the value of signed digits */
static inline void
nm_value (mpz_t r, const int32_t *dg, int D, int b)
{
  mpz_set_ui (r, 0);
  for (int i = D - 1; i >= 0; i--)
    {
      mpz_mul_2exp (r, r, b);
      if (dg[i] >= 0)
        mpz_add_ui (r, r, (unsigned long) dg[i]);
      else
        mpz_sub_ui (r, r, (unsigned long) - (long) dg[i]);
    }
}

template<int LG>
__global__ void
nm_hat_kernel (const NM_GRID_CONST nm_ctx c, uint32_t *H, const int32_t *X,
               uint32_t count)
{
  for (uint32_t i = blockIdx.x; i < count; i += gridDim.x)
    nm_hat<LG> (c, H + (size_t) i * 4 * (1 << LG), X + (size_t) i * (1 << LG));
}

/* the hats of COUNT numbers (contiguous, D words each) */
static inline void
nm_hats (const nm_ctx &c, uint32_t *H, const int32_t *X, uint32_t count,
         cudaStream_t st)
{
  if (count == 0)
    return;
  const size_t sm = nm_smem_bytes (c.lg);
  const uint32_t TPG = (uint32_t) c.D / 8, nb = count < 1024 ? count : 1024;
  switch (c.lg)
    {
#define NM_HC(l) case l:                                                     \
      cudaFuncSetAttribute (nm_hat_kernel<l>,                                \
                            cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                            (int) sm);                                        \
      nm_hat_kernel<l><<<nb, TPG, sm, st>>> (c, H, X, count); break;
      NM_HC (8) NM_HC (9) NM_HC (10) NM_HC (11) NM_HC (12)
#undef NM_HC
    }
}

/* Device tables for N (gcd (N, B^D - 1) must be 1); ALLOC collects every
   device allocation, for nm_ctx_free. */
struct nm_host {
  nm_ctx c;
  std::vector<void *> alloc;
  mpz_t R, R2;               /* R = (B^D - 1) mod N, R^2 mod N */
};

template<class T>
static inline T *
nm_upload (nm_host &h, const std::vector<T> &v)
{
  T *d;
  CUDA_CHECK (cudaMalloc ((void **) &d, v.size () * sizeof (T)));
  CUDA_CHECK (cudaMemcpy (d, v.data (), v.size () * sizeof (T),
                          cudaMemcpyHostToDevice));
  h.alloc.push_back (d);
  return d;
}

/* false if gcd (N, B^D - 1) != 1 */
static inline bool
nm_ctx_create (nm_host &h, const mpz_t N, int lg, int b)
{
  nm_ctx &c = h.c;
  memset (&c, 0, sizeof c);
  c.lg = lg;
  c.b = b;
  c.D = 1 << lg;
  const int D = c.D;
  mpz_t Rr, t, Np;
  mpz_inits (Rr, t, Np, NULL);
  mpz_setbit (Rr, (mp_bitcnt_t) b * D);
  mpz_sub_ui (Rr, Rr, 1);
  mpz_gcd (t, Rr, N);
  if (mpz_cmp_ui (t, 1) != 0)
    {
      mpz_clears (Rr, t, Np, NULL);
      return false;
    }
  mpz_inits (h.R, h.R2, NULL);
  mpz_mod (h.R, Rr, N);
  mpz_mul (h.R2, h.R, h.R);
  mpz_mod (h.R2, h.R2, N);
  mpz_invert (Np, N, Rr);
  mpz_sub (Np, Rr, Np);                 /* -1/N mod R */

  const uint32_t ps[2] = { NM_P0, NM_P1 };
  for (int j = 0; j < 2; j++)
    {
      const uint32_t p = ps[j];
      nm_prime &P = c.pr[j];
      P.p = p;
      uint32_t inv = 1;
      for (int i = 0; i < 5; i++)
        inv *= 2 - p * inv;
      P.pinv = -inv;
      const uint32_t psi = nm_root2k (p, lg + 1);
      const uint32_t w = (uint32_t) ((uint64_t) psi * psi % p);
      const uint32_t wi = nm_powmod (w, p - 2, p), psii = nm_powmod (psi, p - 2, p);
      const uint32_t s32 = (uint32_t) ((1ull << 32) % p);
      const uint32_t sc = (uint32_t) ((uint64_t) s32 * nm_powmod (D, p - 2, p) % p);
      P.sc = sc;
      P.sc_sh = nm_shoup_pair (sc, p).y;
      std::vector<uint2> tw (D / 2), twi (D / 2), tp (D), tip (D);
      uint64_t a = 1, ai = 1;
      for (int k = 0; k < D / 2; k++)
        {
          tw[k] = nm_shoup_pair ((uint32_t) a, p);
          twi[k] = nm_shoup_pair ((uint32_t) ai, p);
          a = a * w % p;
          ai = ai * wi % p;
        }
      a = 1;
      ai = sc;
      for (int i = 0; i < D; i++)
        {
          tp[i] = nm_shoup_pair ((uint32_t) a, p);
          tip[i] = nm_shoup_pair ((uint32_t) ai, p);
          a = a * psi % p;
          ai = ai * psii % p;
        }
      P.w = nm_upload (h, tw);
      P.wi = nm_upload (h, twi);
      P.psi = nm_upload (h, tp);
      P.ipsi = nm_upload (h, tip);
    }
  c.p01 = (uint64_t) NM_P0 * NM_P1;
  c.g = nm_powmod (NM_P0, NM_P1 - 2, NM_P1);
  c.g_sh = nm_shoup_pair (c.g, NM_P1).y;

  /* numbers: N', N, 1, R^2, R (contiguous), then their hats */
  std::vector<int32_t> all ((size_t) 5 * D), dg;
  mpz_t one;
  mpz_init_set_ui (one, 1);
  const mpz_srcptr nums[5] = { Np, N, one, h.R2, h.R };
  for (int k = 0; k < 5; k++)
    {
      nm_digits (dg, nums[k], D, b);
      std::copy (dg.begin (), dg.end (), all.begin () + (size_t) k * D);
    }
  mpz_clear (one);
  int32_t *dn = nm_upload (h, all);
  uint32_t *hats;
  CUDA_CHECK (cudaMalloc ((void **) &hats, (size_t) 4 * 4 * D * 4));
  h.alloc.push_back (hats);
  c.N = dn + D;
  c.one = dn + 4 * D;
  nm_hats (c, hats, dn, 4, 0);
  CUDA_CHECK (cudaDeviceSynchronize ());
  CUDA_CHECK (cudaGetLastError ());
  c.Nph = hats;
  c.Nh = hats + 4 * D;
  c.hone = hats + 8 * D;
  c.hR2 = hats + 12 * D;
  /* N's top digits for quotient estimates */
  nm_digits (dg, N, D, b);
  c.ntop = D - 1;
  while (c.ntop > 0 && dg[c.ntop] == 0)
    c.ntop--;
  assert (c.ntop >= 3);
  double v = 0;
  for (int i = c.ntop; i >= c.ntop - 3; i--)
    v = v * (double) (1 << b) + dg[i];
  c.nhi = v;
  mpz_clears (Rr, t, Np, NULL);
  return true;
}

static inline void
nm_ctx_free (nm_host &h)
{
  for (void *p : h.alloc)
    CUDA_CHECK (cudaFree (p));
  h.alloc.clear ();
  mpz_clears (h.R, h.R2, NULL);
}

#endif /* __CUDACC__ */

#endif /* _GPU_NMUL_CUH */
