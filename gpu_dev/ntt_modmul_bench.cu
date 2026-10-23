/* ntt_modmul_bench.cu: a transform-based Montgomery product mod a general N
   (George Woltman's suggestion: transform each number in the number
   dimension, prime95 style) against the schoolbook products of the GPU
   roots (gm_mul, gpu_mont.cuh, the TPI x L shapes gpu_roots.cu picks).

   The library version of these products is gpu_nmul.cuh (used by
   gpu_roots.cu for large N; tested by nmul_test.cu); this file keeps the
   prototype as measured.

   nvcc -O2 -arch=sm_86 -I.. -o ntt_modmul_bench ntt_modmul_bench.cu -lgmp
   ./ntt_modmul_bench [bits ...]      (default 4096 8192 16384 32768 65536)

   Numbers are d = 2^LG digits of b bits (b about 17), b d >= bits + 7.
   Digits are signed int32 and are never normalised (prime95 style): one
   carry step leaves them in (-2^15, 2^(b+1) + 2^15), so a convolution
   coefficient stays below d 2^36.2 < 2^49 (d <= 4096), far under
   p0 p1 / 2 = 2^58.6 for the two primes p < 2^30 (2^23 | p - 1), and
   Garner gives it exactly, sign included.

   Montgomery reduction with R = B^d - 1 (B = 2^b) instead of B^d, so every
   transform is of length d and none is zero padded:
     T = a b                     T_c = T mod (B^d - 1)  (cyclic, length d)
                                 T_n = T mod (B^d + 1)  (negacyclic)
     q = T_c N' mod (B^d - 1),   N' = -1/N mod (B^d - 1)   (cyclic)
     T + q N = 0 mod R, so r = (T + q N) / R, and as R = -2 mod (B^d + 1):
     r = (T_n + q N mod (B^d + 1)) / -2  mod (B^d + 1)     (negacyclic)
   The representative of r comes from its top digit.  With digits not
   normalised, q's value is in (-B^d / 4, 2.3 B^d), so with B^d >= 128 N
   inputs in (-N/2, 2.5 N) give r in (-0.3 N, 2.35 N) (the bench checks
   (-N/2, 2.5 N)).  Per product:
   a and b each a cyclic and a negacyclic forward transform (or b's
   given: "bhat"), two inverses, one forward and one inverse each for q and
   for q N: 10 (generic) or 8 (bhat) length-d transforms per prime, against
   14 / 12 for R = 2^k (length 2d for a b and for q, d for q N with
   wrap-around, as GMP's redc with mulmod_bnm1).  Needs gcd (N, B^d - 1) =
   1, so the bench draws N without small factors (ECM inputs are
   trial-factored; another b d would do otherwise).

   One group of d/8 threads per product.  Thread TID holds the digits of
   indices TID + t d/8 (t < 8) in registers: what the first forward pass
   (the three largest strides, radix 8) reads and the last inverse pass
   writes, so digit vectors never leave registers; the CRT and the carry
   step are fused onto that last pass.  Transforms run in shared memory,
   both primes per pass, as gpu_ntt.cuh (DIF forward to bit-reversed order,
   DIT inverse back, Shoup twiddles, lazy [0, 2p), Montgomery pointwise
   products with 2^-32 folded into the scaling).

   Both kinds run the same benchmark: x <- x b / R, ITERS times per
   instance, many instances; products/s = instances ITERS / time.  The
   schoolbook kernels keep x, b, N in registers (no memory traffic) and
   iterate gm_mul, so they are an upper bound for the roots' products.
   Every NTT result (and a sample of the schoolbook ones) is checked
   against GMP.  NM_NOSB=1 skips the schoolbook shapes. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <gmp.h>
#include "gpu_mont.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf (stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString (e_)); exit (1); } } while (0)

static int sms;
static int bad = 0;

/* ======================= schoolbook (gm_mul) ======================= */

template<int L, bool WIDE>
__global__ void __launch_bounds__ (128)
sb_kernel (uint32_t tpi, int iters, const uint32_t *Ng, uint32_t np0,
           uint32_t *xg, const uint32_t *bg)
{
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / tpi;
  const uint32_t n = tpi * L;
  uint32_t x[L], b[L], N[L];
  if constexpr (WIDE)
    {
      __shared__ uint32_t smem[GM_WSHARED_WORDS (128, L)];
      gm_wgroup g;
      g.init (tpi, smem, L);
      const uint32_t off = g.lane * L;
      for (int k = 0; k < L; k++)
        {
          x[k] = xg[(size_t) inst * n + off + k];
          b[k] = bg[(size_t) inst * n + off + k];
          N[k] = Ng[off + k];
        }
      for (int it = 0; it < iters; it++)
        gm_mul<L, false> (g, x, x, b, N, np0);
      for (int k = 0; k < L; k++)
        xg[(size_t) inst * n + off + k] = x[k];
    }
  else
    {
      gm_group g;
      g.init (tpi);
      const uint32_t off = g.lane * L;
      for (int k = 0; k < L; k++)
        {
          x[k] = xg[(size_t) inst * n + off + k];
          b[k] = bg[(size_t) inst * n + off + k];
          N[k] = Ng[off + k];
        }
      for (int it = 0; it < iters; it++)
        gm_mul<L, false> (g, x, x, b, N, np0);
      for (int k = 0; k < L; k++)
        xg[(size_t) inst * n + off + k] = x[k];
    }
}

static void
check_result (const char *what, const mpz_t N, const mpz_t Rinv, int iters,
              const mpz_t x0, const mpz_t b, const mpz_t got, const mpz_t lo,
              const mpz_t hi)
{
  mpz_t e, t;
  mpz_inits (e, t, NULL);
  mpz_mul (t, b, Rinv);
  mpz_powm_ui (t, t, iters, N);
  mpz_mul (e, x0, t);
  mpz_mod (e, e, N);
  mpz_mod (t, got, N);
  if (mpz_cmp (e, t) != 0 || mpz_cmp (got, lo) < 0 || mpz_cmp (got, hi) >= 0)
    {
      if (bad < 5)
        gmp_printf ("%s: WRONG (got %s range)\n", what,
                    mpz_cmp (e, t) == 0 ? "out of" : "wrong value, in");
      bad++;
    }
  mpz_clears (e, t, NULL);
}

template<int L, bool WIDE>
static double
sb_run (const mpz_t Nz, uint32_t tpi, gmp_randstate_t rs)
{
  const uint32_t n = tpi * L;
  mpz_t t, R, Rinv, lim;
  mpz_inits (t, R, Rinv, lim, NULL);
  mpz_set_ui (t, 0);
  mpz_setbit (t, 32);
  mpz_invert (t, Nz, t);
  const uint32_t np0 = - (uint32_t) mpz_get_ui (t);
  mpz_set_ui (R, 0);
  mpz_setbit (R, 32 * n);
  mpz_invert (Rinv, R, Nz);
  mpz_mul_2exp (lim, Nz, 1);

  int blocks = 0;
  CK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (&blocks, sb_kernel<L, WIDE>, 128, 0));
  const uint32_t inst = blocks * sms * (128 / tpi) * 2;
  std::vector<uint32_t> N (n, 0), X ((size_t) inst * n, 0), Bv ((size_t) inst * n, 0);
  mpz_export (N.data (), NULL, -1, 4, 0, 0, Nz);
  std::vector<__mpz_struct> x0 (inst), b0 (inst);
  for (uint32_t i = 0; i < inst; i++)
    {
      mpz_init (&x0[i]);
      mpz_init (&b0[i]);
      mpz_urandomm (&x0[i], rs, lim);
      mpz_urandomm (&b0[i], rs, lim);
      mpz_export (&X[(size_t) i * n], NULL, -1, 4, 0, 0, &x0[i]);
      mpz_export (&Bv[(size_t) i * n], NULL, -1, 4, 0, 0, &b0[i]);
    }
  uint32_t *gN, *gX, *gB;
  CK (cudaMalloc (&gN, n * 4));
  CK (cudaMalloc (&gX, X.size () * 4));
  CK (cudaMalloc (&gB, Bv.size () * 4));
  CK (cudaMemcpy (gN, N.data (), n * 4, cudaMemcpyHostToDevice));
  CK (cudaMemcpy (gB, Bv.data (), Bv.size () * 4, cudaMemcpyHostToDevice));
  const uint32_t nb = inst * tpi / 128;
  cudaEvent_t e0, e1;
  CK (cudaEventCreate (&e0));
  CK (cudaEventCreate (&e1));
  /* calibrate: about 300 ms */
  int iters = 4;
  float ms = 0;
  for (;;)
    {
      CK (cudaMemcpy (gX, X.data (), X.size () * 4, cudaMemcpyHostToDevice));
      CK (cudaEventRecord (e0));
      sb_kernel<L, WIDE><<<nb, 128>>> (tpi, iters, gN, np0, gX, gB);
      CK (cudaEventRecord (e1));
      CK (cudaEventSynchronize (e1));
      CK (cudaGetLastError ());
      CK (cudaEventElapsedTime (&ms, e0, e1));
      if (ms > 150 || iters >= 1 << 20)
        break;
      iters = (int) (iters * std::min (64.0, 300.0 / std::max (ms, 0.01f)));
    }
  std::vector<uint32_t> out (X.size ());
  CK (cudaMemcpy (out.data (), gX, out.size () * 4, cudaMemcpyDeviceToHost));
  mpz_t got, zero;
  mpz_inits (got, zero, NULL);
  for (uint32_t i = 0; i < inst; i += std::max (1u, inst / 64))
    {
      mpz_import (got, n, -1, 4, 0, 0, &out[(size_t) i * n]);
      check_result ("schoolbook", Nz, Rinv, iters, &x0[i], &b0[i], got, zero, lim);
    }
  for (uint32_t i = 0; i < inst; i++)
    {
      mpz_clear (&x0[i]);
      mpz_clear (&b0[i]);
    }
  mpz_clears (got, zero, t, R, Rinv, lim, NULL);
  cudaFree (gN);
  cudaFree (gX);
  cudaFree (gB);
  return (double) inst * iters / (ms / 1e3);
}

/* wide groups only for L >= 6, as the roots */
template<int L>
static double
sb_wide (const mpz_t Nz, uint32_t tpi, gmp_randstate_t rs)
{
  if constexpr (L >= 6)
    return sb_run<L, true> (Nz, tpi, rs);
  else
    return 0;
}

/* ======================= transform product ======================= */

#define NP0 998244353u    /* 119 * 2^23 + 1 */
#define NP1 897581057u    /* 107 * 2^23 + 1 */
#define NM_PAD(i) ((i) + ((i) >> 5))
#define NM_PADDED(n) ((n) + ((n) >> 5))

struct nm_prime {
  uint32_t p, pinv;          /* -1/p mod 2^32 */
  uint32_t sc, sc_sh;        /* 2^32 / d mod p (cyclic output scale) */
  const uint2 *w, *wi;       /* w^k, w^-k (Shoup pairs), k < d/2 */
  const uint2 *psi;          /* psi^i, i < d (negacyclic input weights) */
  const uint2 *ipsi;         /* psi^-i 2^32 / d (negacyclic output) */
  const uint32_t *Nph;       /* cyclic transform of N' */
  const uint32_t *Nnh;       /* negacyclic transform of N */
};

struct nm_ctx {
  nm_prime pr[2];
  uint32_t g, g_sh;          /* 1/p0 mod p1 */
  uint64_t p01;
  int b;
};

#define NM_DEV __device__ __forceinline__

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

/* Layout: one group of TPG = d/8 threads per product; thread TID holds
   the digits (and coefficients) of indices TID + t TPG, t < 8 ("column
   layout"), which is exactly what the first forward pass reads and the
   last inverse pass writes (radix-8 passes on the three largest strides),
   so digit vectors never leave registers.  Both primes go through each
   pass together (X0, X1 in shared memory). */

/* twiddle loads: plain (ld.global.nc loads may be hoisted across the
   barriers, all passes' twiddles at once) unless NM_LDG */
#ifdef NM_LDG
#define NM_LD(x) __ldg (&(x))
#else
#define NM_LD(x) (x)
#endif

/* threads per SM the register budget allows (from the A10: 512, i.e. up
   to 128 registers, below d = 4096; 1024 at 4096) */
#ifdef NM_THREADS
#define NM_THREADS_FOR(LG) NM_THREADS
#else
#define NM_THREADS_FOR(LG) ((LG) >= 12 ? 1024 : 512)
#endif

/* forward DIF, first pass (stages LG-1 .. LG-3), from digits in registers */
template<int LG, bool NEG>
NM_DEV void
nm_first (uint32_t *X0, uint32_t *X1, const nm_ctx &c,
          const int32_t (&dg)[8], int tid)
{
  constexpr int TPG = (1 << LG) / 8;
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      uint32_t *x = j ? X1 : X0;
      const uint32_t p = P.p, p2 = 2 * p;
      uint32_t v[8];
#pragma unroll
      for (int t = 0; t < 8; t++)
        {
          const uint32_t xm = dg[t] < 0 ? (uint32_t) (dg[t] + (int32_t) p)
                                        : (uint32_t) dg[t];
          if (NEG)
            {
              const uint2 w = NM_LD (P.psi[tid + t * TPG]);
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
                const uint2 w = NM_LD (P.w[(tid + (t & (hs - 1)) * TPG) << r]);
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
nm_fpass (uint32_t *X0, uint32_t *X1, const nm_ctx &c, int s, int tid)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int lq = s - R + 1, q = 1 << lq;
  for (int g = tid; g < (D >> R); g += TPG)
    {
      const int k = g & (q - 1);
      const int base = ((g >> lq) << (s + 1)) + k;
#pragma unroll
      for (int j = 0; j < 2; j++)
        {
          const nm_prime &P = c.pr[j];
          uint32_t *x = j ? X1 : X0;
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
                    const uint2 w = NM_LD (P.w[(k + (t & (hs - 1)) * q) << sh]);
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
nm_ipass (uint32_t *X0, uint32_t *X1, const nm_ctx &c, int s0, int tid)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  const int q = 1 << s0;
  for (int g = tid; g < (D >> R); g += TPG)
    {
      const int k = g & (q - 1);
      const int base = ((g >> s0) << (s0 + R)) + k;
#pragma unroll
      for (int j = 0; j < 2; j++)
        {
          const nm_prime &P = c.pr[j];
          uint32_t *x = j ? X1 : X0;
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
                    const uint2 w = NM_LD (P.wi[(k + (t & (hs - 1)) * q) << sh]);
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

enum { MID_SAVE, MID_MUL };

/* contiguous groups of 8, both primes: the last three forward stages;
   SAVE: write the transforms to H0, H1 (global) and stop; MUL: multiply
   by H0, H1 pointwise, then the first three inverse stages */
template<int LG, int MODE>
NM_DEV void
nm_mid (uint32_t *X0, uint32_t *X1, const nm_ctx &c, const uint32_t *H0,
        const uint32_t *H1, int tid)
{
  const int base = tid << 3;
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      uint32_t *x = j ? X1 : X0;
      uint4 *B4 = (uint4 *) ((j ? H1 : H0) + base);
      const uint32_t p = P.p, p2 = 2 * p;
      uint32_t v[8];
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
                const uint2 w = NM_LD (P.w[(t & (hs - 1)) << sh]);
                const uint32_t a = v[t], b = v[t + hs];
                v[t] = nm_red (a + b, p2);
                v[t + hs] = nm_shoup (a - b + p2, w.x, w.y, p);
              }
        }
      if (MODE == MID_SAVE)
        {
          B4[0] = make_uint4 (v[0], v[1], v[2], v[3]);
          B4[1] = make_uint4 (v[4], v[5], v[6], v[7]);
          continue;
        }
      const uint4 b0 = B4[0], b1 = B4[1];
      v[0] = nm_mont (v[0], b0.x, p, P.pinv);
      v[1] = nm_mont (v[1], b0.y, p, P.pinv);
      v[2] = nm_mont (v[2], b0.z, p, P.pinv);
      v[3] = nm_mont (v[3], b0.w, p, P.pinv);
      v[4] = nm_mont (v[4], b1.x, p, P.pinv);
      v[5] = nm_mont (v[5], b1.y, p, P.pinv);
      v[6] = nm_mont (v[6], b1.z, p, P.pinv);
      v[7] = nm_mont (v[7], b1.w, p, P.pinv);
#pragma unroll
      for (int r = 0; r < 3; r++)
        {
          const int hs = 1 << r, sh = LG - 1 - r;
#pragma unroll
          for (int t = 0; t < 8; t++)
            if (!(t & hs))
              {
                const uint2 w = NM_LD (P.wi[(t & (hs - 1)) << sh]);
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

/* inverse DIT, last pass (stages LG-3 .. LG-1), scaled (NEG: with the
   negacyclic weights), and the CRT: CO = the signed coefficients */
template<int LG, bool NEG>
NM_DEV void
nm_last (uint32_t *X0, uint32_t *X1, const nm_ctx &c, int tid,
         int64_t (&co)[8])
{
  constexpr int TPG = (1 << LG) / 8;
  uint32_t res[2][8];
#pragma unroll
  for (int j = 0; j < 2; j++)
    {
      const nm_prime &P = c.pr[j];
      uint32_t *x = j ? X1 : X0;
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
                const uint2 w = NM_LD (P.wi[(tid + (t & (hs - 1)) * TPG) << (2 - r)]);
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
            const uint2 w = NM_LD (P.ipsi[tid + t * TPG]);
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

/* the whole transform pipeline for one (nega)cyclic convolution of DG by
   the transforms H0, H1; or (SAVE) DG's transforms into H0, H1 */
template<int LG, bool NEG, int MODE>
NM_DEV void
nm_conv (uint32_t *X0, uint32_t *X1, const nm_ctx &c, const int32_t (&dg)[8],
         const uint32_t *H0, const uint32_t *H1, int tid, int64_t (&co)[8])
{
  constexpr int M = LG - 6;      /* stages between first/last pass and mid */
  nm_first<LG, NEG> (X0, X1, c, dg, tid);
  int s = LG - 4;
  if constexpr (M % 3 == 1)
    { nm_fpass<LG, 1> (X0, X1, c, s, tid); s -= 1; }
  if constexpr (M % 3 == 2)
    { nm_fpass<LG, 2> (X0, X1, c, s, tid); s -= 2; }
  for (; s >= 5; s -= 3)
    nm_fpass<LG, 3> (X0, X1, c, s, tid);
  nm_mid<LG, MODE> (X0, X1, c, H0, H1, tid);
  if (MODE == MID_SAVE)
    return;
  int s0 = 3;
  for (int k = 0; k < M / 3; k++, s0 += 3)
    nm_ipass<LG, 3> (X0, X1, c, s0, tid);
  if constexpr (M % 3 == 1)
    nm_ipass<LG, 1> (X0, X1, c, s0, tid);
  if constexpr (M % 3 == 2)
    nm_ipass<LG, 2> (X0, X1, c, s0, tid);
  nm_last<LG, NEG> (X0, X1, c, tid, co);
}

/* coefficients -> digits, one carry step: c_i = lo + mid B + hi B^2,
   digit_i = lo_i + mid_(i-1) + hi_(i-2), wrapping with sign SGN.  Digits
   are then in (-2^15, 2^b+1 + 2^15), not normalised (coefficients stay
   below 2^49, far under p0 p1 / 2).  Uses X0, X1 as scratch. */
template<int LG>
NM_DEV void
nm_carry (uint32_t *X0, uint32_t *X1, const int64_t (&co)[8],
          int32_t (&dg)[8], int sgn, int b, int tid)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  int32_t *Mi = (int32_t *) X0, *Hi = (int32_t *) X1;
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

/* x <- x b / R mod N, ITERS times per instance; BHAT: b given transformed
   (bh: 4 d words per instance: cyclic p0, p1, negacyclic p0, p1) */
template<int LG, int PPB, bool BHAT>
__global__ void __launch_bounds__ ((1 << LG) / 8 * PPB,
                                   NM_THREADS_FOR (LG) / ((1 << LG) / 8 * PPB))
nm_kernel (const nm_ctx c, int iters, int32_t *xg, const int32_t *bg,
           const uint32_t *bh, uint32_t *scr)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  extern __shared__ uint32_t sm[];
  const int grp = threadIdx.x / TPG, tid = threadIdx.x % TPG;
  const size_t inst = (size_t) blockIdx.x * PPB + grp;
  uint32_t *X0 = sm + grp * 2 * NM_PADDED (D), *X1 = X0 + NM_PADDED (D);
  int32_t *xp = xg + inst * D;
  const uint32_t *bt = BHAT ? bh + inst * 4 * D : scr + inst * 4 * D;
  uint32_t *Yh = scr + inst * 4 * D;
  const int b = c.b, B = 1 << b;
  int32_t x[8], bd[8];
#pragma unroll
  for (int t = 0; t < 8; t++)
    {
      x[t] = xp[tid + t * TPG];
      bd[t] = BHAT ? 0 : bg[inst * D + tid + t * TPG];
    }

  for (int it = 0; it < iters; it++)
    {
      int64_t co[8];
      int32_t Tc[8], Tn[8], q[8], s[8];
      /* T mod B^d - 1 */
      if (!BHAT)
        nm_conv<LG, false, MID_SAVE> (X0, X1, c, bd, Yh, Yh + D, tid, co);
      nm_conv<LG, false, MID_MUL> (X0, X1, c, x, bt, bt + D, tid, co);
      nm_carry<LG> (X0, X1, co, Tc, 1, b, tid);
      /* T mod B^d + 1 */
      if (!BHAT)
        nm_conv<LG, true, MID_SAVE> (X0, X1, c, bd, Yh + 2 * D, Yh + 3 * D, tid, co);
      nm_conv<LG, true, MID_MUL> (X0, X1, c, x, bt + 2 * D, bt + 3 * D, tid, co);
      nm_carry<LG> (X0, X1, co, Tn, -1, b, tid);
      /* q = T_c N' mod B^d - 1 */
      nm_conv<LG, false, MID_MUL> (X0, X1, c, Tc, c.pr[0].Nph, c.pr[1].Nph, tid, co);
      nm_carry<LG> (X0, X1, co, q, 1, b, tid);
      /* S = T_n + q N mod B^d + 1 */
      nm_conv<LG, true, MID_MUL> (X0, X1, c, q, c.pr[0].Nnh, c.pr[1].Nnh, tid, co);
#pragma unroll
      for (int t = 0; t < 8; t++)
        co[t] += Tn[t];
      nm_carry<LG> (X0, X1, co, s, -1, b, tid);
      /* r = S / -2 mod B^d + 1: U = S (+ B^d + 1 if odd), H = U / 2 */
      int32_t *Si = (int32_t *) X0;
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
      /* x = -H, then + k (B^d + 1) to land in [-tiny, B^d / 4] */
#pragma unroll
      for (int t = 0; t < 8; t++)
        x[t] = -((s[t] >> 1) + ((nx[t] & 1) << (b - 1)));
      if (tid == TPG - 1)
        X1[0] = (uint32_t) x[7];
      __syncthreads ();
      const int32_t y = (int32_t) X1[0];
      const int32_t kk = ((5 * B / 8) - y) >> b;
      __syncthreads ();
      if (tid == 0)
        x[0] += kk;
      if (tid == TPG - 1)
        x[7] += kk * B;
    }
#pragma unroll
  for (int t = 0; t < 8; t++)
    xp[tid + t * TPG] = x[t];
}

/* b's transforms (for BHAT) */
template<int LG, int PPB>
__global__ void __launch_bounds__ ((1 << LG) / 8 * PPB)
nm_hat_kernel (const nm_ctx c, const int32_t *bg, uint32_t *bh)
{
  constexpr int D = 1 << LG, TPG = D / 8;
  extern __shared__ uint32_t sm[];
  const int grp = threadIdx.x / TPG, tid = threadIdx.x % TPG;
  const size_t inst = (size_t) blockIdx.x * PPB + grp;
  uint32_t *X0 = sm + grp * 2 * NM_PADDED (D), *X1 = X0 + NM_PADDED (D);
  int32_t bd[8];
  int64_t co[8];
#pragma unroll
  for (int t = 0; t < 8; t++)
    bd[t] = bg[inst * D + tid + t * TPG];
  uint32_t *o = bh + inst * 4 * D;
  nm_conv<LG, false, MID_SAVE> (X0, X1, c, bd, o, o + D, tid, co);
  nm_conv<LG, true, MID_SAVE> (X0, X1, c, bd, o + 2 * D, o + 3 * D, tid, co);
}

/* ---- host ---- */

static uint32_t
powmod (uint64_t a, uint64_t e, uint32_t p)
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

static uint2
shoup_pair (uint32_t w, uint32_t p)
{
  return make_uint2 (w, (uint32_t) (((uint64_t) w << 32) / p));
}

/* a primitive 2^k-th root of unity mod p (2^k | p - 1) */
static uint32_t
root2k (uint32_t p, int k)
{
  for (uint32_t g = 2;; g++)
    {
      /* g is a generator iff g^((p-1)/f) != 1 for the prime factors f */
      uint32_t m = p - 1, f = 2, ok = 1;
      while (m > 1 && ok)
        {
          if (m % f == 0)
            {
              if (powmod (g, (p - 1) / f, p) == 1)
                ok = 0;
              while (m % f == 0)
                m /= f;
            }
          f++;
        }
      if (ok)
        return powmod (g, (p - 1) >> k, p);
    }
}

template<class T>
static T *
upload (const std::vector<T> &h)
{
  T *d;
  CK (cudaMalloc (&d, h.size () * sizeof (T)));
  CK (cudaMemcpy (d, h.data (), h.size () * sizeof (T), cudaMemcpyHostToDevice));
  return d;
}

/* signed digits of b bits <-> mpz */
static void
digits_to_mpz (mpz_t r, const int32_t *dg, int d, int b)
{
  mpz_set_ui (r, 0);
  for (int i = d - 1; i >= 0; i--)
    {
      mpz_mul_2exp (r, r, b);
      if (dg[i] >= 0)
        mpz_add_ui (r, r, dg[i]);
      else
        mpz_sub_ui (r, r, - (long) dg[i]);
    }
}

static void
mpz_to_digits (int32_t *dg, const mpz_t x, int d, int b)
{
  mpz_t t;
  mpz_init_set (t, x);
  for (int i = 0; i < d; i++)
    {
      dg[i] = (int32_t) mpz_fdiv_ui (t, 1ul << b);
      mpz_fdiv_q_2exp (t, t, b);
    }
  if (mpz_sgn (t) != 0)
    {
      fprintf (stderr, "digits overflow\n");
      exit (1);
    }
  mpz_clear (t);
}

/* LG and b for an N of BITS bits, or false: b d >= bits + 7, and b <= 21
   so that coefficients of digits below 2^(b+1) + 2^15 stay under 2^56.2
   (d <= 4096), with margin below p0 p1 / 2 = 2^58.6.  So d = 1024 takes N
   up to 21497 bits, 2048 to 43001, 4096 to 86009. */
static bool
nm_shape (size_t bits, int *lg, int *b)
{
  for (int l = 8; l <= 12; l++)
    {
      const size_t d = 1u << l;
      const int bb = (int) ((bits + 7 + d - 1) / d);
      if (bb <= 21)
        {
          *lg = l;
          *b = std::max (bb, 2);
          return true;
        }
    }
  return false;
}

template<int LG, int PPB>
static void
nm_bench (const mpz_t Nz, int b, gmp_randstate_t rs, double *rate_gen,
          double *rate_hat)
{
  constexpr int D = 1 << LG, TPG = D / 8, NT = TPG * PPB;
  mpz_t R, Rinv, Np, t, lim;
  mpz_inits (R, Rinv, Np, t, lim, NULL);
  mpz_set_ui (R, 0);
  mpz_setbit (R, (unsigned long) b * D);
  mpz_sub_ui (R, R, 1);                 /* R = B^d - 1 */
  mpz_gcd (t, R, Nz);
  if (mpz_cmp_ui (t, 1) != 0)
    {
      printf ("gcd (N, B^d - 1) != 1: change b\n");
      exit (1);
    }
  mpz_invert (Rinv, R, Nz);
  mpz_invert (Np, Nz, R);
  mpz_sub (Np, R, Np);                  /* -1/N mod R */
  mpz_mul_2exp (lim, Nz, 1);

  nm_ctx c;
  memset (&c, 0, sizeof c);
  c.b = b;
  const uint32_t ps[2] = { NP0, NP1 };
  std::vector<int32_t> Nd (D), Npd (D);
  mpz_to_digits (Nd.data (), Nz, D, b);
  mpz_to_digits (Npd.data (), Np, D, b);
  for (int j = 0; j < 2; j++)
    {
      const uint32_t p = ps[j];
      nm_prime &P = c.pr[j];
      P.p = p;
      uint32_t inv = 1;
      for (int i = 0; i < 5; i++)
        inv *= 2 - p * inv;
      P.pinv = -inv;
      const uint32_t psi = root2k (p, LG + 1), w = (uint32_t) ((uint64_t) psi * psi % p);
      const uint32_t wi = powmod (w, p - 2, p), psii = powmod (psi, p - 2, p);
      const uint32_t s32 = (uint32_t) ((1ull << 32) % p);
      const uint32_t sc = (uint32_t) ((uint64_t) s32 * powmod (D, p - 2, p) % p);
      P.sc = sc;
      P.sc_sh = shoup_pair (sc, p).y;
      std::vector<uint2> tw (D / 2), twi (D / 2), tp (D), tip (D);
      uint64_t a = 1, ai = 1;
      for (int k = 0; k < D / 2; k++)
        {
          tw[k] = shoup_pair ((uint32_t) a, p);
          twi[k] = shoup_pair ((uint32_t) ai, p);
          a = a * w % p;
          ai = ai * wi % p;
        }
      a = 1;
      ai = sc;
      for (int i = 0; i < D; i++)
        {
          tp[i] = shoup_pair ((uint32_t) a, p);
          tip[i] = shoup_pair ((uint32_t) ai, p);
          a = a * psi % p;
          ai = ai * psii % p;
        }
      P.w = upload (tw);
      P.wi = upload (twi);
      P.psi = upload (tp);
      P.ipsi = upload (tip);
    }
  c.p01 = (uint64_t) NP0 * NP1;
  c.g = powmod (NP0, NP1 - 2, NP1);
  c.g_sh = shoup_pair (c.g, NP1).y;

  const size_t smem = (size_t) PPB * 2 * NM_PADDED (D) * 4;
  /* transforms of N' (cyclic) and N (negacyclic): one-instance hat runs */
  {
    int32_t *gd;
    uint32_t *gh;
    std::vector<int32_t> two (2 * PPB * D, 0);
    std::copy (Npd.begin (), Npd.end (), two.begin ());
    std::copy (Nd.begin (), Nd.end (), two.begin () + D);
    gd = upload (two);
    CK (cudaMalloc (&gh, (size_t) 2 * PPB * 4 * D * 4));
    CK (cudaFuncSetAttribute (nm_hat_kernel<LG, PPB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    nm_hat_kernel<LG, PPB><<<2, NT, smem>>> (c, gd, gh);
    CK (cudaDeviceSynchronize ());
    for (int j = 0; j < 2; j++)
      {
        uint32_t *nph, *nnh;
        CK (cudaMalloc (&nph, D * 4));
        CK (cudaMalloc (&nnh, D * 4));
        /* instance 0 = N': its cyclic transforms; instance 1 = N: negacyclic */
        CK (cudaMemcpy (nph, gh + (size_t) j * D, D * 4, cudaMemcpyDeviceToDevice));
        CK (cudaMemcpy (nnh, gh + (size_t) 4 * D + (2 + j) * D, D * 4, cudaMemcpyDeviceToDevice));
        c.pr[j].Nph = nph;
        c.pr[j].Nnh = nnh;
      }
    cudaFree (gd);
    cudaFree (gh);
  }

  CK (cudaFuncSetAttribute (nm_kernel<LG, PPB, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
  CK (cudaFuncSetAttribute (nm_kernel<LG, PPB, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
  int blocks = 0;
  CK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (&blocks, nm_kernel<LG, PPB, true>, NT, smem));
  const uint32_t nblk = blocks * sms * 2, inst = nblk * PPB;

  std::vector<int32_t> X ((size_t) inst * D), Bd ((size_t) inst * D);
  std::vector<__mpz_struct> x0 (inst), b0 (inst);
  for (uint32_t i = 0; i < inst; i++)
    {
      mpz_init (&x0[i]);
      mpz_init (&b0[i]);
      mpz_urandomm (&x0[i], rs, lim);
      mpz_urandomm (&b0[i], rs, lim);
      mpz_to_digits (&X[(size_t) i * D], &x0[i], D, b);
      mpz_to_digits (&Bd[(size_t) i * D], &b0[i], D, b);
    }
  int32_t *gX, *gB;
  uint32_t *gH, *gS;
  CK (cudaMalloc (&gX, X.size () * 4));
  gB = upload (Bd);
  CK (cudaMalloc (&gH, (size_t) inst * 4 * D * 4));
  CK (cudaMalloc (&gS, (size_t) inst * 4 * D * 4));
  nm_hat_kernel<LG, PPB><<<nblk, NT, smem>>> (c, gB, gH);
  CK (cudaDeviceSynchronize ());
  CK (cudaGetLastError ());

  cudaEvent_t e0, e1;
  CK (cudaEventCreate (&e0));
  CK (cudaEventCreate (&e1));
  /* outputs in (-0.3 N, 2.35 N) (see the top of the file) */
  mpz_t got, lo, hi;
  mpz_inits (got, lo, hi, NULL);
  mpz_tdiv_q_ui (lo, Nz, 2);
  mpz_neg (lo, lo);
  mpz_mul_ui (hi, Nz, 5);
  mpz_tdiv_q_ui (hi, hi, 2);
  for (int hat = 0; hat < 2; hat++)
    {
      int iters = 2;
      float ms = 0;
      for (;;)
        {
          CK (cudaMemcpy (gX, X.data (), X.size () * 4, cudaMemcpyHostToDevice));
          CK (cudaEventRecord (e0));
          if (hat)
            nm_kernel<LG, PPB, true><<<nblk, NT, smem>>> (c, iters, gX, gB, gH, gS);
          else
            nm_kernel<LG, PPB, false><<<nblk, NT, smem>>> (c, iters, gX, gB, gH, gS);
          CK (cudaEventRecord (e1));
          CK (cudaEventSynchronize (e1));
          CK (cudaGetLastError ());
          CK (cudaEventElapsedTime (&ms, e0, e1));
          if (ms > 150 || iters >= 1 << 20)
            break;
          iters = (int) (iters * std::min (64.0, 300.0 / std::max (ms, 0.01f)));
        }
      std::vector<int32_t> out (X.size ());
      CK (cudaMemcpy (out.data (), gX, out.size () * 4, cudaMemcpyDeviceToHost));
      for (uint32_t i = 0; i < inst; i++)
        {
          digits_to_mpz (got, &out[(size_t) i * D], D, b);
          check_result (hat ? "ntt bhat" : "ntt", Nz, Rinv, iters, &x0[i], &b0[i], got, lo, hi);
        }
      (hat ? *rate_hat : *rate_gen) = (double) inst * iters / (ms / 1e3);
    }
  for (uint32_t i = 0; i < inst; i++)
    {
      mpz_clear (&x0[i]);
      mpz_clear (&b0[i]);
    }
  mpz_clears (got, lo, hi, R, Rinv, Np, t, lim, NULL);
  cudaFree (gX);
  cudaFree (gB);
  cudaFree (gH);
  cudaFree (gS);
  for (int j = 0; j < 2; j++)
    {
      cudaFree ((void *) c.pr[j].w);
      cudaFree ((void *) c.pr[j].wi);
      cudaFree ((void *) c.pr[j].psi);
      cudaFree ((void *) c.pr[j].ipsi);
      cudaFree ((void *) c.pr[j].Nph);
      cudaFree ((void *) c.pr[j].Nnh);
    }
}

/* ======================= driver ======================= */

/* the schoolbook shapes worth trying for n limbs (the roots' choice,
   rt_shape with L <= 12 first, and neighbours) */
static double
sb_best (const mpz_t N, uint32_t need, gmp_randstate_t rs, char *desc)
{
  struct S { uint32_t tpi, L; };
  static const S shapes[] = {
    { 2, 5 }, { 4, 5 }, { 8, 5 }, { 16, 5 }, { 32, 5 },
    { 4, 9 }, { 8, 9 }, { 16, 9 }, { 32, 9 }, { 64, 9 }, { 128, 9 },
    { 64, 6 }, { 128, 6 }, { 64, 12 }, { 128, 12 },
    { 8, 17 }, { 16, 17 }, { 32, 17 }, { 64, 17 }, { 128, 17 } };
  double best = 0;
  desc[0] = 0;
  char *p = desc;
  for (const S &s : shapes)
    {
      const uint32_t n = s.tpi * s.L;
      /* only shapes with at most 1.5x the limbs needed */
      if (n < need || 2 * n > 3 * need + 64)
        continue;
      double r = 0;
#define SB(LL) if (s.L == LL) r = s.tpi > 32 ? sb_wide<LL> (N, s.tpi, rs) \
                                             : sb_run<LL, false> (N, s.tpi, rs);
      SB (5) SB (6) SB (9) SB (12) SB (17)
#undef SB
      p += sprintf (p, " %ux%u:%.0f", s.tpi, s.L, r);
      best = std::max (best, r);
    }
  return best;
}

int
main (int argc, char **argv)
{
  std::vector<size_t> sizes;
  for (int i = 1; i < argc; i++)
    sizes.push_back (strtoul (argv[i], NULL, 10));
  if (sizes.empty ())
    sizes = { 4096, 8192, 16384, 32768, 65536 };
  int dev;
  cudaDeviceProp prop;
  CK (cudaGetDevice (&dev));
  CK (cudaGetDeviceProperties (&prop, dev));
  CK (cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount, dev));
  printf ("%s, %d SMs\n", prop.name, sms);
  gmp_randstate_t rs;
  gmp_randinit_default (rs);
  gmp_randseed_ui (rs, 17);
  printf ("%6s | %-44s | %10s | %5s %3s | %10s %10s | %6s %6s\n", "bits",
          "schoolbook shapes (products/s)", "best", "d", "b", "ntt", "ntt bhat",
          "ratio", "bhat");
  for (size_t bits : sizes)
    {
      int lg = 0, b = 0;
      const bool nm_ok = nm_shape (bits, &lg, &b);
      /* N with gcd (N, 2^(b d) - 1) = 1 (an N with a factor of 2^(b d) - 1
         would be split by that gcd; ECM inputs are trial-factored) */
      mpz_t N, M, t;
      mpz_inits (N, M, t, NULL);
      mpz_set_ui (M, 0);
      mpz_setbit (M, (unsigned long) b << lg);
      mpz_sub_ui (M, M, 1);
      do
        {
          mpz_urandomb (N, rs, bits);
          mpz_setbit (N, bits - 1);
          mpz_setbit (N, 0);
          mpz_gcd (t, N, M);
        }
      while (nm_ok && mpz_cmp_ui (t, 1) != 0);
      const uint32_t need = (bits + 4 + 31) / 32;
      char desc[512];
      desc[0] = 0;
      /* NM_NOSB set: the transform products only */
      const double sb = getenv ("NM_NOSB") ? 1 : sb_best (N, need, rs, desc);
      double g = 0, h = 0;
      if (nm_ok)
        switch (lg)
          {
          case 8: nm_bench<8, 4> (N, b, rs, &g, &h); break;
          case 9: nm_bench<9, 2> (N, b, rs, &g, &h); break;
          case 10: nm_bench<10, 1> (N, b, rs, &g, &h); break;
          case 11: nm_bench<11, 1> (N, b, rs, &g, &h); break;
          case 12: nm_bench<12, 1> (N, b, rs, &g, &h); break;
          }
      printf ("%6zu | %-44s | %10.0f | %5d %3d | %10.0f %10.0f | %6.2f %6.2f\n",
              bits, desc, sb, 1 << lg, b, g, h, g / sb, h / sb);
      fflush (stdout);
      mpz_clears (N, M, t, NULL);
    }
  gmp_randclear (rs);
  printf ("%s\n", bad ? "MISMATCHES" : "all products checked against GMP");
  return bad != 0;
}
