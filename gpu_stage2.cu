/* gpu_stage2.cu: GPU ECM stage 2 (baby-step giant-step prime continuation).

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

/* For each curve, with Q the stage 1 point (x-only, Montgomery form of the
 * param 3 curve, a24 = sigma / 2^32 mod N):
 *
 *   baby steps  j Q for 1 <= j <= D/2, gcd (j, D) = 1   (table in memory)
 *   giant steps G_k = k D Q, G_{k+1} = G_k + DQ (difference G_{k-1})
 *
 * Every prime p in (B2min, B2] is k D + j or k D - j for one such j, and
 * p Q = O mod q implies x(G_k) = x(jQ) mod q, so the product of
 *   X(G_k) Z(jQ) - X(jQ) Z(G_k)
 * over the (k, j) pairs hit by some prime has a factor q in common with N.
 * Which pairs to use is the same for all curves: the host sieves the primes
 * into one bit mask per giant step.  The baby steps are put on one shared
 * denominator once, (X_j : Z_j) -> (X_j prod_{i != j} Z_i : prod_i Z_i)
 * (see gpu_stage2_kernel.cuh), so a pair costs 2 multiplications,
 *   X(G_k) BZ - bx_j Z(G_k)  and the product into the accumulator,
 * plus one differential addition (6) and X(G_k) BZ (1) per giant step.  All
 * values stay below 2N (see the bounds in gpu_stage1.cu).  A curve whose
 * product has gcd N is redone on the CPU with finer gcds (s2_replay).
 *
 * Every launch is sized to take about 100 ms, the one-off setup kernels
 * included: they work in slices, with their state in device memory.
 *
 * The numbers use the same (TPI, L) shape as stage 1 (gpu_ecm_stage1_plan).
 * Per-curve arrays in device memory are laid out so that a warp's loads are
 * contiguous: limb k of number s for thread t is at (s * L + k) * T + t,
 * T = total threads.
 */

#ifndef __CUDACC__
#error "This file should only be compiled with nvcc"
#endif

#include "gpu_stage2.h"
#include "gpu_stage1.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include <gmp.h>
#include <cuda.h>

#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_kernels.h"
#include "gpu_stage2_kernel.cuh"

#include "ecm.h"
#include "ecm-gpu.h"
#include "getprime_r.h"

/* giant steps sieved into masks at a time */
#define S2_MAX_STEPS (1 << 16)

static unsigned long
gcd_ul (unsigned long a, unsigned long b)
{
  while (b)
    {
      unsigned long t = a % b;
      a = b;
      b = t;
    }
  return a;
}

/* Sieve the primes of (lo, hi] into per-giant-step masks: bit b of giant
 * step k is set when k D +- jlist[b] is such a prime.  Fills steps
 * [k0, k0 + count); *p is the next prime to place, advanced as used. */
struct s2_sieve {
  prime_info_t pi;
  uint64_t p;         /* next prime not yet placed */
  uint64_t hi;
  uint32_t D;
  std::vector<int> slot;  /* slot[j] = baby index of j, or -1 */
  uint64_t pairs;     /* number of (k, j) terms */
};

static void
s2_fill_masks (s2_sieve &sv, uint32_t *mask, uint64_t k0, uint32_t count,
               uint32_t mw)
{
  memset (mask, 0, (size_t) count * mw * sizeof (uint32_t));
  while (sv.p <= sv.hi)
    {
      uint64_t k = (sv.p + sv.D / 2) / sv.D;
      if (k >= k0 + count)
        break;
      uint64_t r = sv.p - k * sv.D + sv.D;   /* p - kD in [-D/2, D/2] + D */
      uint32_t j = (uint32_t) (r > sv.D ? r - sv.D : sv.D - r);
      int b = sv.slot[j];
      assert (b >= 0 && k >= k0);
      uint32_t *w = mask + (size_t) (k - k0) * mw + b / 32;
      if (!(*w & (1u << (b % 32))))
        sv.pairs++;
      *w |= 1u << (b % 32);
      sv.p = getprime_mt (sv.pi);
    }
}

/* Stage 2 arithmetic on the host, mod N, for s2_replay: the same xDBL and
   xADD as the GPU (gpu_stage2_kernel.cuh), a24 = d. */
struct s2_cpu {
  const __mpz_struct *N;
  mpz_t d, t1, t2, t3, t4;
  s2_cpu (const mpz_t n, uint32_t sigma) : N (n)
  {
    mpz_init (d);
    mpz_init (t1);
    mpz_init (t2);
    mpz_init (t3);
    mpz_init (t4);
    /* a24 = sigma / 2^32 mod N */
    mpz_set_ui (d, 1);
    mpz_mul_2exp (d, d, 32);
    mpz_invert (d, d, N);
    mpz_mul_ui (d, d, sigma);
    mpz_mod (d, d, N);
  }
  ~s2_cpu ()
  {
    mpz_clear (d);
    mpz_clear (t1);
    mpz_clear (t2);
    mpz_clear (t3);
    mpz_clear (t4);
  }
  void mul (mpz_t r, const mpz_t a, const mpz_t b)
  {
    mpz_mul (r, a, b);
    mpz_mod (r, r, N);
  }
  /* (X : Z) = 2 (X1 : Z1); may alias */
  void dbl (mpz_t X, mpz_t Z, const mpz_t X1, const mpz_t Z1)
  {
    mpz_add (t1, X1, Z1);
    mul (t1, t1, t1);                 /* A */
    mpz_sub (t2, X1, Z1);
    mul (t2, t2, t2);                 /* B */
    mpz_sub (t3, t1, t2);             /* C */
    mul (X, t1, t2);
    mul (t1, t3, d);
    mpz_add (t1, t1, t2);
    mul (Z, t3, t1);
  }
  /* (X : Z) = P + Q, P - Q = D; may alias P or Q, not D */
  void add (mpz_t X, mpz_t Z, const mpz_t XP, const mpz_t ZP,
            const mpz_t XQ, const mpz_t ZQ, const mpz_t XD, const mpz_t ZD)
  {
    mpz_sub (t1, XP, ZP);
    mpz_add (t2, XQ, ZQ);
    mul (t1, t1, t2);                 /* U */
    mpz_add (t2, XP, ZP);
    mpz_sub (t3, XQ, ZQ);
    mul (t2, t2, t3);                 /* V */
    mpz_add (t3, t1, t2);
    mpz_sub (t4, t1, t2);
    mul (t3, t3, t3);
    mul (t4, t4, t4);
    mul (X, ZD, t3);
    mul (Z, XD, t4);
  }
};

/* G = gcd (A, N); returns 1 if it is a proper factor. */
static int
s2_proper (mpz_t g, const mpz_t a, const mpz_t N)
{
  mpz_gcd (g, a, N);
  return mpz_cmp_ui (g, 1) != 0 && mpz_cmp (g, N) != 0;
}

/* When the GPU's product for a curve has gcd N (factors found together, or
 * one baby step's Z sharing a factor with N, which spreads that factor to
 * all of the curve's terms; see gpu_stage2_kernel.cuh), redo the curve's
 * stage 2 here, as cuda-sieve does: the baby steps' Z each get a gcd, and
 * the terms, now per point, X(G) Z_j - X_j Z(G), a gcd per chunk of giant
 * steps and, in a chunk whose gcd is N, one each.  Same pairs as the GPU.
 * Rare, so plain GMP arithmetic.  X0 is the curve's stage 1 x.  Returns 1
 * with a proper factor of N in F. */
static int
s2_replay (mpz_t f, const mpz_t x0, uint32_t sigma, const mpz_t N,
           uint32_t D, const std::vector<uint32_t> &jlist,
           const std::vector<int> &slot, uint64_t lo, uint64_t hi,
           uint64_t kend)
{
  const uint32_t nbaby = jlist.size ();
  const uint32_t mw = (nbaby + 31) / 32;
  const uint32_t chunk = 16;   /* giant steps per gcd */
  s2_cpu e (N, sigma);
  std::vector<__mpz_struct> bX (nbaby), bZ (nbaby),
                            term ((size_t) chunk * nbaby);
  for (auto &z : bX) mpz_init (&z);
  for (auto &z : bZ) mpz_init (&z);
  for (auto &z : term) mpz_init (&z);
  mpz_t QX, QZ, Q2X, Q2Z, cX, cZ, pX, pZ, nX, nZ, DX, DZ, prod;
  mpz_init_set (QX, x0);
  mpz_init_set_ui (QZ, 1);
  mpz_init (Q2X);
  mpz_init (Q2Z);
  mpz_init_set (cX, QX);
  mpz_init_set (cZ, QZ);
  mpz_init_set (pX, QX);
  mpz_init_set (pZ, QZ);
  mpz_init (nX);
  mpz_init (nZ);
  mpz_init (DX);
  mpz_init (DZ);
  mpz_init (prod);

  int found = 0;
  s2_sieve sv;
  sv.D = D;
  sv.hi = hi;
  sv.pairs = 0;
  sv.slot = slot;
  prime_info_init (sv.pi);
  uint32_t *mask = (uint32_t *) malloc ((size_t) chunk * mw * 4);

  /* baby steps j Q, j odd <= D/2, as on the GPU */
  e.dbl (Q2X, Q2Z, QX, QZ);
  mpz_set (&bX[0], QX);
  mpz_set (&bZ[0], QZ);
  for (uint32_t j = 3; j <= D / 2; j += 2)
    {
      e.add (nX, nZ, cX, cZ, Q2X, Q2Z, pX, pZ);
      mpz_swap (pX, cX);
      mpz_swap (pZ, cZ);
      mpz_swap (cX, nX);
      mpz_swap (cZ, nZ);
      if (slot[j] >= 0)
        {
          mpz_set (&bX[slot[j]], cX);
          mpz_set (&bZ[slot[j]], cZ);
        }
    }
  for (uint32_t b = 0; b < nbaby && !found; b++)
    found = s2_proper (f, &bZ[b], N);
  if (found)
    goto done;

  /* prev = G_1 = DQ, cur = G_2 */
  e.dbl (DX, DZ, cX, cZ);
  mpz_set (pX, DX);
  mpz_set (pZ, DZ);
  e.dbl (cX, cZ, DX, DZ);
  do
    sv.p = getprime_mt (sv.pi);
  while (sv.p <= lo || sv.p * 2 < D * 3);
  for (uint64_t k = 2; k < kend && !found; k += chunk)
    {
      uint32_t cnt = (uint32_t) std::min ((uint64_t) chunk, kend - k);
      uint32_t nterm = 0;
      s2_fill_masks (sv, mask, k, cnt, mw);
      mpz_set_ui (prod, 1);
      for (uint32_t c = 0; c < cnt; c++)
        {
          for (uint32_t b = 0; b < nbaby; b++)
            if (mask[c * mw + b / 32] & (1u << (b % 32)))
              {
                __mpz_struct *t = &term[nterm++];
                e.mul (t, cX, &bZ[b]);
                e.mul (nX, &bX[b], cZ);
                mpz_sub (t, t, nX);
                e.mul (prod, prod, t);
              }
          e.add (nX, nZ, cX, cZ, DX, DZ, pX, pZ);
          mpz_swap (pX, cX);
          mpz_swap (pZ, cZ);
          mpz_swap (cX, nX);
          mpz_swap (cZ, nZ);
        }
      mpz_gcd (f, prod, N);
      if (mpz_cmp_ui (f, 1) == 0)
        continue;
      found = mpz_cmp (f, N) != 0;
      for (uint32_t i = 0; i < nterm && !found; i++)
        found = s2_proper (f, &term[i], N);
    }

 done:
  prime_info_clear (sv.pi);
  free (mask);
  for (auto &z : bX) mpz_clear (&z);
  for (auto &z : bZ) mpz_clear (&z);
  for (auto &z : term) mpz_clear (&z);
  mpz_clear (QX);
  mpz_clear (QZ);
  mpz_clear (Q2X);
  mpz_clear (Q2Z);
  mpz_clear (cX);
  mpz_clear (cZ);
  mpz_clear (pX);
  mpz_clear (pZ);
  mpz_clear (nX);
  mpz_clear (nZ);
  mpz_clear (DX);
  mpz_clear (DZ);
  mpz_clear (prod);
  return found;
}

/* Time the launches between start and stop: milliseconds. */
static float
s2_wait (cudaEvent_t start, cudaEvent_t stop)
{
  float ms = 0;
  CUDA_CHECK (cudaEventRecord (stop));
  CUDA_CHECK (cudaEventSynchronize (stop));
  CUDA_CHECK (cudaEventElapsedTime (&ms, start, stop));
  return ms;
}

/* Relative costs for pacing the launches, in modular products. */
#define S2_COST_SETUP 6    /* one xadd or xdbl */
#define S2_COST_NORM 2
#define S2_COST_GIANT 7    /* one xadd, X(G) BZ */
#define S2_COST_PAIR 2

extern "C"
int
gpu_ecm_stage2 (mpz_t *xs, int *array_found, const mpz_t N,
                const mpz_t B2min, const mpz_t B2,
                uint32_t curves, uint32_t sigma, const uint32_t *sigmas,
                int config, float *gputime, int verbose)
{
  uint32_t tpi, L;
  gpu_ecm_plan_shape (config, &tpi, &L);
  const uint32_t n = tpi * L;
  const uint32_t padded = (curves + GPU_TPB / tpi - 1) / (GPU_TPB / tpi)
                          * (GPU_TPB / tpi);
  const uint32_t T = padded * tpi;
  const uint32_t blocks = T / GPU_TPB;
  const gpu_kernel_set *ks = gpu_kernels (tpi, L);
  assert (ks != NULL);

  *gputime = 0;
  if (mpz_sizeinbase (B2, 2) > 62)
    {
      outputf (OUTPUT_ERROR, "GPU: Error, B2 too large for GPU stage 2\n");
      return ECM_ERROR;
    }
  uint64_t lo = mpz_sgn (B2min) > 0 ? (uint64_t) mpz_get_d (B2min) : 0;
  uint64_t hi = (uint64_t) mpz_get_d (B2);
  if (hi <= lo)
    return ECM_NO_FACTOR_FOUND;

  /* D: the largest primorial-type modulus whose first giant step index
     is >= 2 (giant steps start at G_2) and whose baby table fits. */
  size_t free_mem, total_mem;
  CUDA_CHECK (cudaMemGetInfo (&free_mem, &total_mem));
  static const uint32_t Ds[] = { 2310, 210, 30, 6 };
  uint32_t D = 6;
  std::vector<uint32_t> jlist;
  for (size_t i = 0; i < sizeof (Ds) / sizeof (Ds[0]); i++)
    {
      uint32_t d = Ds[i];
      std::vector<uint32_t> jl;
      for (uint32_t j = 1; j <= d / 2; j++)
        if (gcd_ul (j, d) == 1)
          jl.push_back (j);
      /* X and Z per baby step */
      size_t bytes = (size_t) T * L * 2 * jl.size () * sizeof (uint32_t);
      if ((lo + d / 2) / d >= 2 && bytes < free_mem / 2)
        {
          D = d;
          jlist = jl;
          break;
        }
      if (d == 6)
        {
          D = d;
          jlist = jl;
        }
    }
  const uint32_t nbaby = jlist.size ();
  const uint32_t mw = (nbaby + 31) / 32;

  s2_sieve sv;
  sv.D = D;
  sv.hi = hi;
  sv.pairs = 0;
  sv.slot.assign (D / 2 + 1, -1);
  for (uint32_t b = 0; b < nbaby; b++)
    sv.slot[jlist[b]] = b;
  prime_info_init (sv.pi);
  do
    sv.p = getprime_mt (sv.pi);
  while (sv.p <= lo || sv.p * 2 < D * 3);  /* need k >= 2: p >= 1.5 D */
  uint64_t k0 = (sv.p + D / 2) / D;
  uint64_t kend = (hi + D / 2) / D + 1;    /* giant steps [k0, kend) */

  outputf (OUTPUT_VERBOSE, "GPU stage 2: D=%u, %u baby steps, giant steps "
           "%lu..%lu, %'lu bytes of baby steps\n", D, nbaby,
           (unsigned long) k0, (unsigned long) kend - 1,
           (size_t) T * L * 2 * nbaby * sizeof (uint32_t));

  /* host data: x residues and constants in Montgomery form */
  uint32_t *xin = (uint32_t *) calloc ((size_t) padded * n, sizeof (uint32_t));
  uint32_t *Nl = (uint32_t *) malloc (n * sizeof (uint32_t));
  uint32_t *onel = (uint32_t *) malloc (n * sizeof (uint32_t));
  {
    mpz_t R, x;
    mpz_init (R);
    mpz_init (x);
    mpz_setbit (R, 32 * (size_t) n);
    mpz_mod (R, R, N);
    gpu_to_limbs (onel, n, R);
    gpu_to_limbs (Nl, n, N);
    for (uint32_t i = 0; i < padded; i++)
      {
        mpz_mul (x, xs[i < curves ? i : 0], R);
        mpz_mod (x, x, N);
        gpu_to_limbs (xin + (size_t) i * n, n, x);
      }
    mpz_clear (R);
    mpz_clear (x);
  }
  uint32_t np0 = gpu_find_np0 (N);

  uint32_t *g_xin, *g_N, *g_one, *g_jl, *g_baby, *g_state, *g_mask;
  CUDA_CHECK (cudaMalloc ((void **) &g_xin, (size_t) padded * n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_N, n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_one, n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_jl, nbaby * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_baby, (size_t) T * L * 2 * nbaby * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_state, (size_t) T * L * S_COUNT * 4));
  CUDA_CHECK (cudaMalloc ((void **) &g_mask, (size_t) S2_MAX_STEPS * mw * 4));
  CUDA_CHECK (cudaMemcpy (g_xin, xin, (size_t) padded * n * 4,
                          cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMemcpy (g_N, Nl, n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMemcpy (g_one, onel, n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMemcpy (g_jl, jlist.data (), nbaby * 4,
                          cudaMemcpyHostToDevice));

  /* per-curve sigmas of a resumed batch (padding curves repeat curve 0's) */
  uint32_t *g_dl = NULL;
  if (sigmas != NULL)
    {
      std::vector<uint32_t> dl (padded, sigmas[0]);
      std::copy (sigmas, sigmas + curves, dl.begin ());
      CUDA_CHECK (cudaMalloc ((void **) &g_dl, padded * 4));
      CUDA_CHECK (cudaMemcpy (g_dl, dl.data (), padded * 4,
                              cudaMemcpyHostToDevice));
    }

  cudaEvent_t start, bstart, stop;
  CUDA_CHECK (cudaEventCreate (&start));
  CUDA_CHECK (cudaEventCreate (&bstart));
  CUDA_CHECK (cudaEventCreate (&stop));
  CUDA_CHECK (cudaEventRecord (start));

  /* Every launch gets about BUDGET modular products' worth of work per
     curve, paced to ~100 ms (gpu_pace); the first ones are small, since
     at the largest sizes a few thousand products per curve take seconds. */
  uint64_t budget = 64;
  float batch_ms, max_ms = 0;

  /* baby steps and DQ, G_1, G_2 */
  const uint32_t nsetup = (D / 2 + 1) / 2 + 2;
  for (uint32_t s = 0; s < nsetup; )
    {
      uint64_t want = std::max ((uint64_t) 1, budget / S2_COST_SETUP);
      uint32_t cnt = (uint32_t) std::min ((uint64_t) nsetup - s, want);
      uint32_t jmax = s > 0 ? 2 * s - 1 : 1, next0 = 0;
      while (next0 < nbaby && jlist[next0] <= jmax)
        next0++;
      CUDA_CHECK (cudaEventRecord (bstart));
      ks->s2_setup<<<blocks, GPU_TPB>>> (g_xin, g_N, g_one, np0, sigma, tpi,
                                         D / 2, g_jl, nbaby, g_baby, g_state,
                                         s, s + cnt, next0, g_dl);
      CUDA_CHECK (cudaGetLastError ());
      batch_ms = s2_wait (bstart, stop);
      max_ms = std::max (max_ms, batch_ms);
      budget = gpu_pace (budget, (uint64_t) cnt * S2_COST_SETUP, batch_ms,
                         cnt < want);
      s += cnt;
    }

  /* the shared denominator */
  for (uint32_t i = 0; i < 2 * nbaby; )
    {
      uint64_t want = std::max ((uint64_t) 1, budget / S2_COST_NORM);
      uint32_t cnt = (uint32_t) std::min ((uint64_t) 2 * nbaby - i, want);
      CUDA_CHECK (cudaEventRecord (bstart));
      ks->s2_norm<<<blocks, GPU_TPB>>> (g_baby, nbaby, g_state, g_N, np0,
                                        tpi, i, i + cnt);
      CUDA_CHECK (cudaGetLastError ());
      batch_ms = s2_wait (bstart, stop);
      max_ms = std::max (max_ms, batch_ms);
      budget = gpu_pace (budget, (uint64_t) cnt * S2_COST_NORM, batch_ms,
                         cnt < want);
      i += cnt;
    }

  /* giant steps: sieve a chunk of masks, then launch slices of it */
  uint32_t *mask = (uint32_t *) malloc ((size_t) S2_MAX_STEPS * mw * 4);
  std::vector<uint32_t> cost (S2_MAX_STEPS);
  uint64_t k = 2;   /* setup leaves G_2; steps below k0 have empty masks */
  int batches = 0;
  while (k < kend)
    {
      uint32_t chunk = (uint32_t) std::min ((uint64_t) S2_MAX_STEPS, kend - k);
      s2_fill_masks (sv, mask, k, chunk, mw);
      for (uint32_t c = 0; c < chunk; c++)
        {
          uint32_t pairs = 0;
          for (uint32_t w = 0; w < mw; w++)
            pairs += __builtin_popcount (mask[(size_t) c * mw + w]);
          cost[c] = S2_COST_GIANT + S2_COST_PAIR * pairs;
        }
      CUDA_CHECK (cudaMemcpy (g_mask, mask, (size_t) chunk * mw * 4,
                              cudaMemcpyHostToDevice));
      for (uint32_t pos = 0; pos < chunk; )
        {
          uint32_t cnt = 1;
          uint64_t units = cost[pos];
          while (pos + cnt < chunk && units + cost[pos + cnt] <= budget)
            units += cost[pos + cnt++];
          CUDA_CHECK (cudaEventRecord (bstart));
          ks->s2_main<<<blocks, GPU_TPB>>> (g_mask + (size_t) pos * mw, cnt,
                                            mw, g_baby, g_state, g_N, np0,
                                            tpi);
          CUDA_CHECK (cudaGetLastError ());
          batch_ms = s2_wait (bstart, stop);
          max_ms = std::max (max_ms, batch_ms);
          budget = gpu_pace (budget, units, batch_ms, pos + cnt == chunk);
          pos += cnt;
          k += cnt;
          batches++;
          if (verbose >= OUTPUT_VERBOSE && (batches < 3 || batches % 100 == 0))
            printf ("GPU stage 2: giant step %lu/%lu (%.1f%%), %u steps/call\n",
                    (unsigned long) k, (unsigned long) kend,
                    100.0 * (k - 2) / (kend - 2), cnt);
        }
    }
  prime_info_clear (sv.pi);

  /* gather the accumulators */
  uint32_t *acc = (uint32_t *) malloc ((size_t) T * L * 4);
  CUDA_CHECK (cudaMemcpy (acc, g_state + (size_t) S_ACC * L * T,
                          (size_t) T * L * 4, cudaMemcpyDeviceToHost));
  cudaEventElapsedTime (gputime, start, stop);

  outputf (OUTPUT_VERBOSE, "GPU stage 2: %'lu (giant, baby) pairs, %lu giant "
           "steps, longest launch %.0f ms\n", (unsigned long) sv.pairs,
           (unsigned long) (kend - k0), max_ms);

  int youpi = ECM_NO_FACTOR_FOUND;
  {
    uint32_t *a = (uint32_t *) malloc (n * 4);
    mpz_t v, f;
    mpz_init (v);
    mpz_init (f);
    for (uint32_t i = 0; i < curves; i++)
      {
        for (uint32_t ln = 0; ln < tpi; ln++)
          for (uint32_t w = 0; w < L; w++)
            a[ln * L + w] = acc[(size_t) w * T + i * tpi + ln];
        gpu_from_limbs (v, a, n);
        mpz_gcd (f, v, N);
        if (mpz_cmp (f, N) == 0)
          {
            outputf (OUTPUT_VERBOSE, "GPU stage 2: curve %u (-sigma 3:%u) "
                     "has gcd N, redoing it on the CPU\n", i,
                     sigmas != NULL ? sigmas[i] : sigma + i);
            if (s2_replay (v, xs[i], sigmas != NULL ? sigmas[i] : sigma + i,
                           N, D, jlist, sv.slot, lo, hi,
                           kend))
              mpz_set (f, v);
          }
        if (mpz_cmp_ui (f, 1) != 0 && array_found[i] == ECM_NO_FACTOR_FOUND)
          {
            mpz_set (xs[i], f);
            array_found[i] = ECM_FACTOR_FOUND_STEP2;
            youpi = ECM_FACTOR_FOUND_STEP2;
          }
      }
    mpz_clear (v);
    mpz_clear (f);
    free (a);
  }

  CUDA_CHECK (cudaFree (g_xin));
  CUDA_CHECK (cudaFree (g_N));
  CUDA_CHECK (cudaFree (g_one));
  CUDA_CHECK (cudaFree (g_jl));
  if (g_dl != NULL)
    CUDA_CHECK (cudaFree (g_dl));
  CUDA_CHECK (cudaFree (g_baby));
  CUDA_CHECK (cudaFree (g_state));
  CUDA_CHECK (cudaFree (g_mask));
  CUDA_CHECK (cudaEventDestroy (start));
  CUDA_CHECK (cudaEventDestroy (bstart));
  CUDA_CHECK (cudaEventDestroy (stop));
  free (xin);
  free (Nl);
  free (onel);
  free (mask);
  free (acc);
  return youpi;
}
