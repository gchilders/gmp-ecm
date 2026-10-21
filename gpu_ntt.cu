/* gpu_ntt.cu: GPU ECM stage 1 for large inputs, with NTT multiplication.

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

/* One thread block per curve runs the same Montgomery ladder as the
 * schoolbook kernel (gpu_stage1_kernel.cuh, same bounds), with the
 * multiplications of gpu_ntt.cuh.  Values live in per-curve global
 * memory; the block's shared memory holds one transform at a time. */

#ifndef __CUDACC__
#error "This file should only be compiled with nvcc"
#endif

#include "gpu_ntt.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include <cuda.h>

#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_ntt.cuh"

/* six working numbers per curve (the ladder's four and two temporaries;
   stage 2's temporaries) */
#define NTT_NBUF 6

static size_t
per_curve_words (int L)
{
  return ntt_scratch_words (L) + (size_t) NTT_NBUF * (L / 2);
}

#ifdef NTT_PROFILE
__device__ unsigned long long ntt_prof_g[NTT_NPROF];   /* summed over blocks */
#endif

static __device__ void
ntt_copy (const ntt_ctx &c, uint32_t *r, const uint32_t *a)
{
  for (int i = threadIdx.x; i < c.W; i += blockDim.x)
    r[i] = a[i];
  __syncthreads ();
}

/* This block's curve's scratch: the multiplication's (X) and, returned,
   NTT_NBUF working numbers. */
static __device__ uint32_t *
ntt_curve_scratch (const ntt_ctx &c, uint32_t *scratch, size_t per_curve,
                   ntt_scratch &X)
{
  const int L = c.L;
  uint32_t *base = scratch + (size_t) blockIdx.x * per_curve;
  X.ahat = base;
  for (int i = 0; i < NTT_NPRIMES; i++)
    X.res[i] = base + (size_t) L * (1 + i);
  X.T = base + (size_t) L * (1 + NTT_NPRIMES);
  X.Q = X.T + L;
  X.U = X.Q + L;
  X.mw = X.U + L;
  return base + ntt_scratch_words (L);
}

__global__ void __launch_bounds__ (NTT_MAX_THREADS)
ntt_stage1_kernel (const uint32_t * __restrict__ s_bits, uint64_t s_nbits,
                   uint64_t b_start, uint64_t b_count,
                   uint32_t * __restrict__ data,
                   uint32_t * __restrict__ scratch, size_t per_curve,
                   const ntt_ctx c, uint32_t sigma0,
                   const uint32_t * __restrict__ dlist,
                   const uint32_t * __restrict__ x0g)
{
  extern __shared__ uint32_t sm[];
  __shared__ ntt_shared S;
#ifdef NTT_PROFILE
  if (threadIdx.x == 0)
    {
      for (int i = 0; i < NTT_NPROF; i++)
        S.prof[i] = 0;
      S.prof_t = clock64 ();
    }
  __syncthreads ();
#endif

  const uint32_t inst = blockIdx.x;
  const int W = c.W;
  ntt_scratch X;
  uint32_t *buf = ntt_curve_scratch (c, scratch, per_curve, X);

  /* logical numbers -> buffers; the ladder rotates the pointers */
  uint32_t *aX = buf, *aZ = buf + W, *bX = buf + 2 * W, *bZ = buf + 3 * W;
  uint32_t *t1 = buf + 4 * W, *t2 = buf + 5 * W;
  uint32_t *dp = data + (size_t) inst * 4 * W;
  ntt_copy (c, aX, dp);
  ntt_copy (c, aZ, dp + W);
  ntt_copy (c, bX, dp + 2 * W);
  ntt_copy (c, bZ, dp + 3 * W);

  const uint32_t d = dlist != NULL ? dlist[inst] : sigma0 + inst;
  const uint32_t *x0 = x0g != NULL ? x0g + (size_t) inst * W : NULL;

  int swapped = 0;
  for (uint64_t b = b_start; b < b_start + b_count; b++)
    {
      uint64_t nth = s_nbits - 1 - b;
      int bit = (s_bits[nth >> 5] >> (nth & 31)) & 1;
      if (bit != swapped)
        {
          uint32_t *t;
          swapped = bit;
          t = aX; aX = bX; bX = t;
          t = aZ; aZ = bZ; bZ = t;
        }
      /* the ladder step of gpu_stage1_kernel.cuh */
      ntt_mw_add (S, c, t1, bZ, bX);
      ntt_mw_sub (S, c, bZ, bZ, bX);
      ntt_mw_add (S, c, bX, aZ, aX);
      ntt_mw_sub (S, c, aZ, aZ, aX);
      ntt_mont_mul (sm, S, c, X, aX, t1, aZ);     /* aX = CB */
      ntt_mont_mul (sm, S, c, X, t1, bZ, bX);     /* t1 = DA */
      ntt_mont_mul (sm, S, c, X, bZ, bX, bX);     /* bZ = AA */
      ntt_mont_mul (sm, S, c, X, bX, aZ, aZ);     /* bX = BB */
      ntt_mw_sub (S, c, aZ, bZ, bX);              /* aZ = K */
      ntt_mont_mul (sm, S, c, X, bZ, bZ, bX);     /* bZ = AA*BB = new aX */
      ntt_mw_mul_ui (S, c, X, t2, aZ, d);         /* t2 = dK */
      ntt_mw_add (S, c, bX, bX, t2);              /* bX = BB + dK */
      ntt_mont_mul (sm, S, c, X, aZ, aZ, bX);     /* aZ = new aZ */
      ntt_mw_add (S, c, bX, t1, aX);              /* bX = DA + CB */
      ntt_mw_sub (S, c, t1, t1, aX);              /* t1 = DA - CB */
      ntt_mont_mul (sm, S, c, X, bX, bX, bX);     /* bX = new bX */
      ntt_mont_mul (sm, S, c, X, t1, t1, t1);
      if (x0 == NULL)
        ntt_mw_dbl (c, t1, t1);                   /* t1 = new bZ */
      else
        ntt_mont_mul (sm, S, c, X, t1, t1, x0);
      /* (aX, bZ, t1) <- (bZ, t1, aX) */
      uint32_t *t = aX;
      aX = bZ;
      bZ = t1;
      t1 = t;
    }
  if (swapped)
    {
      uint32_t *t;
      t = aX; aX = bX; bX = t;
      t = aZ; aZ = bZ; bZ = t;
    }

  ntt_copy (c, dp, aX);
  ntt_copy (c, dp + W, aZ);
  ntt_copy (c, dp + 2 * W, bX);
  ntt_copy (c, dp + 3 * W, bZ);
#ifdef NTT_PROFILE
  if (threadIdx.x == 0)
    for (int i = 0; i < NTT_NPROF; i++)
      atomicAdd (&ntt_prof_g[i], S.prof[i]);
#endif
}

/* ---- stage 2 (see gpu_stage2.cu and gpu_stage2_kernel.cuh) ----------- */

/* The same steps as the schoolbook stage 2 kernels, on numbers in global
 * memory: per curve, STATE holds S_COUNT numbers and BABY 2 NBABY (X of
 * baby step b at b, Z at NBABY + b), W limbs each, contiguous per curve. */

/* (Xo : Zo) = 2 (X1 : Z1); TMP: three numbers.  May alias. */
static __device__ void
ntt_xdbl (uint32_t *sm, ntt_shared &S, const ntt_ctx &c, ntt_scratch &X,
          uint32_t *const tmp[3], uint32_t *Xo, uint32_t *Zo,
          const uint32_t *X1, const uint32_t *Z1, uint32_t d)
{
  uint32_t *t1 = tmp[0], *t2 = tmp[1], *t3 = tmp[2];
  ntt_mw_add (S, c, t1, X1, Z1);                  /* < 4N */
  ntt_mw_sub (S, c, t2, X1, Z1);                  /* < 2N */
  ntt_mont_mul (sm, S, c, X, t1, t1, t1);         /* A = (X+Z)^2 */
  ntt_mont_mul (sm, S, c, X, t2, t2, t2);         /* B = (X-Z)^2 */
  ntt_mw_sub (S, c, t3, t1, t2);                  /* C = A - B */
  ntt_mont_mul (sm, S, c, X, Xo, t1, t2);         /* X = A B */
  ntt_mw_mul_ui (S, c, X, t1, t3, d);             /* a24 C < 3N */
  ntt_mw_add (S, c, t1, t1, t2);                  /* B + a24 C < 5N */
  ntt_mont_mul (sm, S, c, X, Zo, t3, t1);         /* Z = C (B + a24 C) */
}

/* (Xo : Zo) = P + Q given D = P - Q; may alias P or Q, not D */
static __device__ void
ntt_xadd (uint32_t *sm, ntt_shared &S, const ntt_ctx &c, ntt_scratch &X,
          uint32_t *const tmp[3], uint32_t *Xo, uint32_t *Zo,
          const uint32_t *XP, const uint32_t *ZP,
          const uint32_t *XQ, const uint32_t *ZQ,
          const uint32_t *XD, const uint32_t *ZD)
{
  uint32_t *t1 = tmp[0], *t2 = tmp[1], *t3 = tmp[2];
  ntt_mw_sub (S, c, t1, XP, ZP);
  ntt_mw_add (S, c, t2, XQ, ZQ);
  ntt_mont_mul (sm, S, c, X, t1, t1, t2);         /* U = (XP-ZP)(XQ+ZQ) */
  ntt_mw_add (S, c, t2, XP, ZP);
  ntt_mw_sub (S, c, t3, XQ, ZQ);
  ntt_mont_mul (sm, S, c, X, t2, t2, t3);         /* V = (XP+ZP)(XQ-ZQ) */
  ntt_mw_add (S, c, t3, t1, t2);                  /* U + V < 4N */
  ntt_mw_sub (S, c, t1, t1, t2);                  /* U - V < 2N */
  ntt_mont_mul (sm, S, c, X, t3, t3, t3);
  ntt_mont_mul (sm, S, c, X, t1, t1, t1);
  ntt_mont_mul (sm, S, c, X, Xo, ZD, t3);
  ntt_mont_mul (sm, S, c, X, Zo, XD, t1);
}

#define NTT_S2_PROLOGUE                                                    \
  extern __shared__ uint32_t sm[];                                       \
  __shared__ ntt_shared S;                                               \
  const uint32_t inst = blockIdx.x;                                      \
  const int W = c.W;                                                     \
  ntt_scratch X;                                                         \
  uint32_t *buf = ntt_curve_scratch (c, scratch, per_curve, X);          \
  uint32_t *const tmp[3] = { buf, buf + W, buf + 2 * W };                \
  uint32_t *st = state + (size_t) inst * S_COUNT * W;                    \
  (void) tmp; (void) st

#define SL(s) (st + (size_t) (s) * W)
#define BB(b) (bb + (size_t) (b) * W)

/* Baby steps, DQ, G_1 and G_2: steps [STEP0, STEP1) of s2_setup_kernel */
__global__ void __launch_bounds__ (NTT_MAX_THREADS)
ntt_s2_setup_kernel (const uint32_t * __restrict__ xin,
                     const uint32_t * __restrict__ oneg, uint32_t halfD,
                     const uint32_t * __restrict__ jlist, uint32_t nbaby,
                     uint32_t * __restrict__ baby,
                     uint32_t * __restrict__ state,
                     uint32_t step0, uint32_t step1, uint32_t next0,
                     uint32_t * __restrict__ scratch, size_t per_curve,
                     const ntt_ctx c, uint32_t sigma0,
                     const uint32_t * __restrict__ dlist)
{
  NTT_S2_PROLOGUE;
  uint32_t *bb = baby + (size_t) inst * 2 * nbaby * W;
  uint32_t *nX = buf + 3 * W, *nZ = buf + 4 * W;
  const uint32_t d = dlist != NULL ? dlist[inst] : sigma0 + inst;
  const uint32_t nodd = (halfD + 1) / 2;   /* odd j in [1, D/2] */
  const uint32_t *x = xin + (size_t) inst * W;

  uint32_t next = next0;
  if (step0 == 0)
    {
      /* cur = Q, prev = -Q (x(-Q) = x(Q)): j = 1 */
      ntt_copy (c, SL (S_CUR_X), x);
      ntt_copy (c, SL (S_CUR_Z), oneg);
      ntt_copy (c, SL (S_PREV_X), x);
      ntt_copy (c, SL (S_PREV_Z), oneg);
      ntt_copy (c, BB (0), x);
      ntt_copy (c, BB (nbaby), oneg);
      next = 1;
    }
  for (uint32_t step = step0; step < step1; step++)
    {
      if (step == 0 || step >= nodd)
        {
          /* Q2 = 2Q (into AUX), DQ = 2 cur, G2 = 2 DQ (into cur) */
          const int src = step <= nodd ? S_CUR_X : S_DQ_X;
          const int dst = step == 0 ? S_AUX_X : (step == nodd ? S_DQ_X : S_CUR_X);
          ntt_xdbl (sm, S, c, X, tmp, SL (dst), SL (dst + 1), SL (src),
                    SL (src + 1), d);
        }
      else
        {
          ntt_xadd (sm, S, c, X, tmp, nX, nZ, SL (S_CUR_X), SL (S_CUR_Z),
                    SL (S_AUX_X), SL (S_AUX_Z), SL (S_PREV_X), SL (S_PREV_Z));
          ntt_copy (c, SL (S_PREV_X), SL (S_CUR_X));
          ntt_copy (c, SL (S_PREV_Z), SL (S_CUR_Z));
          ntt_copy (c, SL (S_CUR_X), nX);
          ntt_copy (c, SL (S_CUR_Z), nZ);
          if (next < nbaby && jlist[next] == 2 * step + 1)
            {
              ntt_copy (c, BB (next), SL (S_CUR_X));
              ntt_copy (c, BB (nbaby + next), SL (S_CUR_Z));
              next++;
            }
        }
    }
  if (step1 == nodd + 2)
    {
      /* done: prev = G_1 = DQ, cur = G_2, acc = 1 */
      ntt_copy (c, SL (S_PREV_X), SL (S_DQ_X));
      ntt_copy (c, SL (S_PREV_Z), SL (S_DQ_Z));
      ntt_copy (c, SL (S_ACC), oneg);
      ntt_copy (c, SL (S_AUX_X), oneg);
    }
}

/* The shared denominator: iterations [I0, I1) of s2_norm_kernel */
__global__ void __launch_bounds__ (NTT_MAX_THREADS)
ntt_s2_norm_kernel (uint32_t * __restrict__ baby, uint32_t nbaby,
                    uint32_t * __restrict__ state, uint32_t i0, uint32_t i1,
                    uint32_t * __restrict__ scratch, size_t per_curve,
                    const ntt_ctx c)
{
  NTT_S2_PROLOGUE;
  uint32_t *bb = baby + (size_t) inst * 2 * nbaby * W;
  uint32_t *acc = SL (S_AUX_X);
  for (uint32_t i = i0; i < i1; i++)
    {
      const bool fwd = i < nbaby;
      const uint32_t b = fwd ? i : 2 * nbaby - 1 - i;
      const bool first = i == 0 || i == nbaby;
      if (i == nbaby)
        ntt_copy (c, SL (S_BZ), acc);
      if (!first)
        ntt_mont_mul (sm, S, c, X, BB (b), BB (b), acc);
      if (first)
        ntt_copy (c, acc, BB (nbaby + b));
      else if (fwd || b > 0)
        ntt_mont_mul (sm, S, c, X, acc, acc, BB (nbaby + b));
    }
}

/* COUNT giant steps of s2_main_kernel */
__global__ void __launch_bounds__ (NTT_MAX_THREADS)
ntt_s2_main_kernel (const uint32_t * __restrict__ masks, uint32_t count,
                    uint32_t mw, const uint32_t * __restrict__ baby,
                    uint32_t nbaby, uint32_t * __restrict__ state,
                    uint32_t * __restrict__ scratch, size_t per_curve,
                    const ntt_ctx c)
{
  NTT_S2_PROLOGUE;
  const uint32_t *bb = baby + (size_t) inst * 2 * nbaby * W;
  uint32_t *nX = buf + 3 * W, *nZ = buf + 4 * W, *gx = buf + 5 * W;
  for (uint32_t k = 0; k < count; k++)
    {
      /* X(G) BZ, shared by the giant step's terms */
      ntt_mont_mul (sm, S, c, X, gx, SL (S_BZ), SL (S_CUR_X));
      for (uint32_t w = 0; w < mw; w++)
        {
          uint32_t bits = masks[k * mw + w];
          while (bits != 0)
            {
              const uint32_t bi = w * 32 + __ffs (bits) - 1;
              bits &= bits - 1;
              /* acc *= X(G) BZ - bx_j Z(G) */
              ntt_mont_mul (sm, S, c, X, tmp[0], BB (bi), SL (S_CUR_Z));
              ntt_mw_sub (S, c, tmp[1], gx, tmp[0]);
              ntt_mont_mul (sm, S, c, X, SL (S_ACC), SL (S_ACC), tmp[1]);
            }
        }
      /* G_{k+1} = G_k + DQ, difference G_{k-1} */
      ntt_xadd (sm, S, c, X, tmp, nX, nZ, SL (S_CUR_X), SL (S_CUR_Z),
                SL (S_DQ_X), SL (S_DQ_Z), SL (S_PREV_X), SL (S_PREV_Z));
      ntt_copy (c, SL (S_PREV_X), SL (S_CUR_X));
      ntt_copy (c, SL (S_PREV_Z), SL (S_CUR_Z));
      ntt_copy (c, SL (S_CUR_X), nX);
      ntt_copy (c, SL (S_CUR_Z), nZ);
    }
}

#undef SL
#undef BB
#undef NTT_S2_PROLOGUE

/* ---------------------------------------------------------------------- */

struct gpu_ntt_plan {
  int lg, L, W, threads;
  size_t smem;
  ntt_ctx ctx;               /* device pointers inside */
  std::vector<void *> dev;   /* device allocations owned by the plan */
  uint32_t *g_scratch;
  uint32_t scratch_curves;
};

static uint64_t
powmod (uint64_t b, uint64_t e, uint64_t m)
{
  uint64_t r = 1;
  b %= m;
  while (e)
    {
      if (e & 1)
        r = r * b % m;
      b = b * b % m;
      e >>= 1;
    }
  return r;
}

static uint32_t
primitive_root (uint32_t p)
{
  std::vector<uint32_t> f;
  uint32_t m = p - 1;
  for (uint32_t q = 2; q * q <= m; q++)
    if (m % q == 0)
      {
        f.push_back (q);
        while (m % q == 0)
          m /= q;
      }
  if (m > 1)
    f.push_back (m);
  for (uint32_t g = 2;; g++)
    {
      bool ok = true;
      for (size_t i = 0; i < f.size () && ok; i++)
        ok = powmod (g, (p - 1) / f[i], p) != 1;
      if (ok)
        return g;
    }
}

static uint32_t
shoup_const (uint32_t w, uint32_t p)
{
  return (uint32_t) (((uint64_t) w << 32) / p);
}

/* the device's forward transform, on the host (same output order) */
static void
host_forward (std::vector<uint32_t> &x, uint32_t p, int lg,
              const std::vector<uint32_t> &wtab)
{
  const int L = 1 << lg;
  for (int s = lg - 1; s >= 0; s--)
    {
      const int h = 1 << s, sh = lg - 1 - s;
      for (int j = 0; j < L / 2; j++)
        {
          const int k = j & (h - 1);
          const int i1 = ((j >> s) << (s + 1)) + k, i2 = i1 + h;
          uint64_t a = x[i1], b = x[i2];
          x[i1] = (uint32_t) ((a + b) % p);
          x[i2] = (uint32_t) ((a + p - b) % p * wtab[k << sh] % p);
        }
    }
}

template<class T>
static T *
dev_copy (gpu_ntt_plan *P, const T *src, size_t n)
{
  T *d;
  CUDA_CHECK (cudaMalloc ((void **) &d, n * sizeof (T)));
  CUDA_CHECK (cudaMemcpy (d, src, n * sizeof (T), cudaMemcpyHostToDevice));
  P->dev.push_back (d);
  return d;
}

static void
to_words (std::vector<uint32_t> &w, size_t n, const mpz_t a)
{
  w.assign (n, 0);
  size_t cnt = 0;
  assert (mpz_sizeinbase (a, 2) <= 32 * n);
  mpz_export (w.data (), &cnt, -1, sizeof (uint32_t), 0, 0, a);
}

int
gpu_ntt_lg_for_bits (size_t bits)
{
  for (int lg = NTT_MIN_LG; lg <= NTT_MAX_LG; lg++)
    if (bits + 4 <= (size_t) 32 * ((1u << lg) / 2))
      return lg;
  return -1;
}

int
gpu_ntt_device_max_lg (void)
{
  int dev, optin = 0;
  CUDA_CHECK (cudaGetDevice (&dev));
  CUDA_CHECK (cudaDeviceGetAttribute (&optin,
                  cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const void *kernels[] = { (const void *) ntt_stage1_kernel,
                            (const void *) ntt_s2_setup_kernel,
                            (const void *) ntt_s2_norm_kernel,
                            (const void *) ntt_s2_main_kernel };
  size_t stat = 0;
  for (size_t i = 0; i < sizeof (kernels) / sizeof (kernels[0]); i++)
    {
      cudaFuncAttributes fa;
      CUDA_CHECK (cudaFuncGetAttributes (&fa, kernels[i]));
      stat = std::max (stat, fa.sharedSizeBytes);
    }
  int best = -1;
  for (int lg = NTT_MIN_LG; lg <= NTT_MAX_LG; lg++)
    if (stat + NTT_PADDED ((size_t) 1 << lg) * sizeof (uint32_t)
        <= (size_t) optin)
      best = lg;
  return best;
}

size_t
gpu_ntt_max_bits (void)
{
  return (size_t) 32 * ((1u << NTT_MAX_LG) / 2) - 4;
}

gpu_ntt_plan *
gpu_ntt_create (const mpz_t N, int lg)
{
  static const uint32_t primes[NTT_NPRIMES] = { NTT_P0, NTT_P1, NTT_P2 };
  gpu_ntt_plan *P = new gpu_ntt_plan;
  const int L = 1 << lg, W = L / 2;
  P->lg = lg;
  P->L = L;
  P->W = W;
  P->smem = (size_t) NTT_PADDED (L) * sizeof (uint32_t);
  {
    const void *kernels[] = { (const void *) ntt_stage1_kernel,
                              (const void *) ntt_s2_setup_kernel,
                              (const void *) ntt_s2_norm_kernel,
                              (const void *) ntt_s2_main_kernel };
    for (size_t i = 0; i < sizeof (kernels) / sizeof (kernels[0]); i++)
      CUDA_CHECK (cudaFuncSetAttribute (kernels[i],
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      (int) P->smem));
  }
  /* L/8 threads (one radix-8 group each per pass) or L/16 (two), at least
     128: whichever keeps more threads resident per SM, and on a tie more
     blocks (shared memory, not threads, often limits the blocks) */
  P->threads = 0;
  int best = -1;
  for (int t = L / 8; t >= L / 16 && t >= 128; t /= 2)
    {
      const int tt = t > NTT_MAX_THREADS ? NTT_MAX_THREADS : t;
      int blocks = 0;
      CUDA_CHECK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (
                      &blocks, ntt_stage1_kernel, tt, P->smem));
      if (blocks * tt >= best && tt != P->threads)
        {
          best = blocks * tt;
          P->threads = tt;
        }
    }
  if (P->threads == 0)
    P->threads = 128;
  /* GMPECM_NTT_THREADS=n (power of 2, 32..min(L/8, NTT_MAX_THREADS)), for
     tuning; ntt_crt_carry needs runs of L/(4n) >= 2 */
  const char *env = getenv ("GMPECM_NTT_THREADS");
  if (env != NULL)
    {
      int t = atoi (env);
      if (t >= 32 && t <= NTT_MAX_THREADS && (t & (t - 1)) == 0
          && t <= L / 8 && (L / 2) / t <= 64)
        P->threads = t;
    }
  P->g_scratch = NULL;
  P->scratch_curves = 0;
  ntt_ctx &c = P->ctx;
  c.lg = lg;
  c.L = L;
  c.W = W;

  /* N, 2N, N' = -1/N mod R, as W limbs */
  mpz_t R, t;
  mpz_init (R);
  mpz_init (t);
  mpz_setbit (R, 32 * (size_t) W);
  std::vector<uint32_t> Nw, N2w, Npw;
  to_words (Nw, W, N);
  mpz_mul_2exp (t, N, 1);
  to_words (N2w, W, t);
  mpz_invert (t, N, R);
  mpz_sub (t, R, t);
  to_words (Npw, W, t);
  c.N = dev_copy (P, Nw.data (), W);
  c.N2 = dev_copy (P, N2w.data (), W);
  c.np0 = Npw[0];
  mpz_clear (R);
  mpz_clear (t);

  for (int i = 0; i < NTT_NPRIMES; i++)
    {
      const uint32_t p = primes[i];
      ntt_prime &Q = c.pr[i];
      Q.p = p;
      uint32_t inv = p;               /* p^-1 mod 2^32 by Newton */
      for (int k = 0; k < 5; k++)
        inv *= 2 - p * inv;
      Q.pinv = 0 - inv;
      Q.one_sh = shoup_const (1, p);
      uint32_t Linv = (uint32_t) powmod (L, p - 2, p);
      Q.scale = (uint32_t) ((((uint64_t) 1 << 32) % p) * Linv % p);
      Q.scale_sh = shoup_const (Q.scale, p);

      uint32_t g = primitive_root (p);
      uint32_t w = (uint32_t) powmod (g, (p - 1) / L, p);
      uint32_t wi = (uint32_t) powmod (w, p - 2, p);
      std::vector<uint32_t> wt (L / 2), wit (L / 2);
      std::vector<uint2> wd (L / 2), wid (L / 2);
      uint64_t a = 1, b = 1;
      for (int k = 0; k < L / 2; k++)
        {
          wt[k] = (uint32_t) a;
          wit[k] = (uint32_t) b;
          wd[k] = make_uint2 (wt[k], shoup_const (wt[k], p));
          wid[k] = make_uint2 (wit[k], shoup_const (wit[k], p));
          a = a * w % p;
          b = b * wi % p;
        }
      Q.w = dev_copy (P, wd.data (), L / 2);
      Q.wi = dev_copy (P, wid.data (), L / 2);

      std::vector<uint32_t> x (L, 0);
      for (int j = 0; j < W; j++)
        x[j] = Nw[j] % p;
      host_forward (x, p, lg, wt);
      Q.Nhat = dev_copy (P, x.data (), L);
      x.assign (L, 0);
      for (int j = 0; j < W; j++)
        x[j] = Npw[j] % p;
      host_forward (x, p, lg, wt);
      Q.Nphat = dev_copy (P, x.data (), L);
    }

  /* Garner constants */
  const uint32_t p0 = primes[0], p1 = primes[1], p2 = primes[2];
  c.c01 = (uint32_t) powmod (p0 % p1, p1 - 2, p1);
  c.c01_sh = shoup_const (c.c01, p1);
  c.c02 = (uint32_t) powmod (p0 % p2, p2 - 2, p2);
  c.c02_sh = shoup_const (c.c02, p2);
  c.c12 = (uint32_t) powmod (p1 % p2, p2 - 2, p2);
  c.c12_sh = shoup_const (c.c12, p2);
  c.p01 = (uint64_t) p0 * p1;

  return P;
}

void
gpu_ntt_destroy (gpu_ntt_plan *P)
{
  for (size_t i = 0; i < P->dev.size (); i++)
    CUDA_CHECK (cudaFree (P->dev[i]));
  if (P->g_scratch != NULL)
    CUDA_CHECK (cudaFree (P->g_scratch));
  delete P;
}

uint32_t
gpu_ntt_limbs (const gpu_ntt_plan *P)
{
  return P->W;
}

uint32_t
gpu_ntt_fill_curves (const gpu_ntt_plan *P)
{
  int device, sms, blocks = 0;
  CUDA_CHECK (cudaGetDevice (&device));
  CUDA_CHECK (cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount,
                                      device));
  CUDA_CHECK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (
                  &blocks, ntt_stage1_kernel, P->threads, P->smem));
  if (blocks < 1)
    blocks = 1;
  /* two waves' worth, so the tail of one wave overlaps the next */
  return (uint32_t) blocks * sms * 2;
}

/* per-curve scratch for CURVES curves */
static void
ensure_scratch (gpu_ntt_plan *P, uint32_t curves)
{
  if (P->scratch_curves < curves)
    {
      if (P->g_scratch != NULL)
        CUDA_CHECK (cudaFree (P->g_scratch));
      CUDA_CHECK (cudaMalloc ((void **) &P->g_scratch,
                              (size_t) curves * per_curve_words (P->L)
                              * sizeof (uint32_t)));
      P->scratch_curves = curves;
    }
}

void
gpu_ntt_s2_setup (gpu_ntt_plan *P, uint32_t curves, const uint32_t *xin,
                  const uint32_t *one, uint32_t halfD, const uint32_t *jlist,
                  uint32_t nbaby, uint32_t *baby, uint32_t *state,
                  uint32_t step0, uint32_t step1, uint32_t next0,
                  uint32_t sigma0, const uint32_t *dlist)
{
  ensure_scratch (P, curves);
  ntt_s2_setup_kernel<<<curves, P->threads, P->smem>>> (
      xin, one, halfD, jlist, nbaby, baby, state, step0, step1, next0,
      P->g_scratch, per_curve_words (P->L), P->ctx, sigma0, dlist);
  CUDA_CHECK (cudaGetLastError ());
}

void
gpu_ntt_s2_norm (gpu_ntt_plan *P, uint32_t curves, uint32_t *baby,
                 uint32_t nbaby, uint32_t *state, uint32_t i0, uint32_t i1)
{
  ensure_scratch (P, curves);
  ntt_s2_norm_kernel<<<curves, P->threads, P->smem>>> (
      baby, nbaby, state, i0, i1, P->g_scratch, per_curve_words (P->L),
      P->ctx);
  CUDA_CHECK (cudaGetLastError ());
}

void
gpu_ntt_s2_main (gpu_ntt_plan *P, uint32_t curves, const uint32_t *masks,
                 uint32_t count, uint32_t mw, const uint32_t *baby,
                 uint32_t nbaby, uint32_t *state)
{
  ensure_scratch (P, curves);
  ntt_s2_main_kernel<<<curves, P->threads, P->smem>>> (
      masks, count, mw, baby, nbaby, state, P->g_scratch,
      per_curve_words (P->L), P->ctx);
  CUDA_CHECK (cudaGetLastError ());
}

void
gpu_ntt_launch (gpu_ntt_plan *P, uint32_t curves, const uint32_t *s_bits,
                uint64_t s_nbits, uint64_t start, uint64_t count,
                uint32_t *data, uint32_t sigma0, const uint32_t *dlist,
                const uint32_t *x0)
{
  const size_t pc = per_curve_words (P->L);
  ensure_scratch (P, curves);
  ntt_stage1_kernel<<<curves, P->threads, P->smem>>> (
      s_bits, s_nbits, start, count, data, P->g_scratch, pc, P->ctx, sigma0,
      dlist, x0);
  CUDA_CHECK (cudaGetLastError ());
}
