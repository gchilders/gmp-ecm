/* gpu_s2poly.cu: the polynomial (FFT) stage 2 on the GPU (see gpu_s2poly.h
   and gpu_dev/poly_stage2_design.md).

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

/* GMP-ECM's stage2() (stage2.c) curve by curve, every step on the GPU:
 *   roots f_i of F (gpu_roots), F = prod (x + f_i) and its product tree,
 *   1/F; for each of the k blocks the roots g of G, G = prod (x + g),
 *   H <- (G - F) H mod F (H = G - F for the first); the transposed
 *   evaluation gives H(-f_i) = prod (g - f_i) over all g, whose product
 *   (on the host) has a factor q of N when some prime of (B2min, B2] (times
 *   the Brent-Suyama polynomial) kills the point mod q.  The value at f_0
 *   is what GMP-ECM prints as "Product of G(f_i)", for comparison.
 *
 * B curves at a time, side by side in every launch (the roots, the
 * polynomial algorithms): one curve's polynomials of a few thousand
 * coefficients leave most of the GPU idle. */

#include "gpu_s2poly.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <time.h>

#include <gmp.h>
#include <cuda.h>

#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_rns.h"
#include "gpu_polyalg.h"
#include "gpu_roots.h"

#include "ecm.h"
#include "ecm-gpu.h"

/* GMPECM_S2POLY_PROF set (not 0): time the phases (synchronising after
   each, so without overlap) */
enum { PH_RSETUP, PH_RF, PH_TREE, PH_INV, PH_RG, PH_G, PH_MULMOD, PH_EVAL,
       PH_HOST, PH_COUNT };
static const char *s2p_phase[PH_COUNT] = {
  "roots setup", "roots of F", "F and tree", "1/F and prep", "roots of G",
  "G", "G H mod F", "evaluation", "product, gcd" };

struct s2p_prof {
  bool on;
  double t[PH_COUNT], last;
  s2p_prof () : on (false), last (0)
  {
    const char *e = getenv ("GMPECM_S2POLY_PROF");
    on = e != NULL && *e != 0 && strcmp (e, "0") != 0;
    for (double &x : t)
      x = 0;
    mark (-1);
  }
  void mark (int ph)
  {
    if (!on)
      return;
    CUDA_CHECK (cudaDeviceSynchronize ());
    timespec ts;
    clock_gettime (CLOCK_MONOTONIC, &ts);
    double now = ts.tv_sec + 1e-9 * ts.tv_nsec;
    if (ph >= 0)
      t[ph] += now - last;
    last = now;
  }
  void print (uint32_t curves)
  {
    if (!on)
      return;
    double tot = 0;
    for (double x : t)
      tot += x;
    printf ("GPU stage 2 phases (ms per curve):");
    for (int i = 0; i < PH_COUNT; i++)
      printf (" %s %.2f,", s2p_phase[i], t[i] * 1e3 / curves);
    printf (" total %.2f\n", tot * 1e3 / curves);
  }
};

static int
s2p_lg (unsigned long dF)
{
  int l = 0;
  while ((1ul << l) < dF)
    l++;
  return l;
}

int
gpu_ecm_stage2_poly_supported (const mpz_t N, unsigned long dF)
{
  if (dF == 0 || (dF & (dF - 1)) != 0
      || mpz_sizeinbase (N, 2) > gpu_roots_max_bits ())
    return 0;
  /* enough NTT primes for products of length 2 dF */
  const int l = s2p_lg (dF);
  gpu_rns *R = gpu_rns_create (N, l + 1, 1);
  int ok = gpu_rns_lg (R) >= l + 1;
  gpu_rns_destroy (R);
  return ok;
}

/* device memory for B curves at a time */
static double
s2p_mem (const mpz_t N, unsigned long dF, unsigned long B)
{
  const double n = (double) dF, l = s2p_lg (dF);
  const double nl = (double) ((mpz_sizeinbase (N, 2) + 4 + 31) / 32);
  const double P = (2 * 32 * nl + l + 1) / 29 + 4;   /* NTT primes */
  double curve = (l + 7) * n * nl * 4    /* tree, F, I, H, G, values, roots x2 */
                 + 12 * n * nl * 4       /* polyalg scratch */
                 + 11 * P * n * 4        /* RNS transforms and scratch */
                 + 3 * n * (nl + 32) * 4 /* roots: numerators, inversions */
                 + 1e5 * (nl + 32);      /* roots: fixed-base table */
  return 1.25 * (B * curve + 64e6);
}

double
gpu_ecm_stage2_poly_mem (const mpz_t N, unsigned long dF)
{
  return s2p_mem (N, dF, 1);
}

unsigned long
gpu_ecm_stage2_poly_fit (const mpz_t N, unsigned long dF)
{
  size_t free_mem, total_mem;
  CUDA_CHECK (cudaMemGetInfo (&free_mem, &total_mem));
  unsigned long d = 1;
  while (2 * d <= dF)
    d *= 2;
  for (; d >= 1; d /= 2)
    if (gpu_ecm_stage2_poly_mem (N, d) <= (double) free_mem
        && gpu_ecm_stage2_poly_supported (N, d))
      return d;
  return 0;
}

/* Curves per batch: about 2^18 coefficients (enough to fill a GPU), within
   the free memory; GMPECM_S2POLY_BATCH overrides. */
static uint32_t
s2p_batch (const mpz_t N, unsigned long dF, uint32_t curves)
{
  const char *env = getenv ("GMPECM_S2POLY_BATCH");
  uint32_t B = env != NULL && atoi (env) > 0 ? (uint32_t) atoi (env)
               : (uint32_t) std::max (1ul, (1ul << 18) / dF);
  B = std::min (B, curves);
  size_t free_mem, total_mem;
  CUDA_CHECK (cudaMemGetInfo (&free_mem, &total_mem));
  while (B > 1 && s2p_mem (N, dF, B) > 0.9 * free_mem)
    B--;
  return B;
}

int
gpu_ecm_stage2_poly (mpz_t *xs, int *array_found, mpz_t *As, const mpz_t N,
                     int S, unsigned long d1, unsigned long d2,
                     const mpz_t i0, unsigned long dF, unsigned long k,
                     uint32_t curves, float *gputime, int verbose)
{
  *gputime = 0;
  if (!gpu_ecm_stage2_poly_supported (N, dF))
    {
      outputf (OUTPUT_ERROR, "GPU: Error, no polynomial stage 2 for this "
               "size of N and dF = %lu\n", dF);
      return ECM_ERROR;
    }
  const int l = s2p_lg (dF);
  const uint32_t B = s2p_batch (N, dF, curves);
  gpu_rns *R = gpu_rns_create (N, l + 1, 1);
  const int nl = gpu_rns_limbs (R);
  gpu_polyalg *PA = gpu_polyalg_create (R, dF, B);
  gpu_roots *RT = gpu_roots_create (N, 0, 0, S, d1, d2, i0, dF, k, nl, B, 0);
  if (RT == NULL)
    {
      gpu_polyalg_destroy (PA);
      gpu_rns_destroy (R);
      outputf (OUTPUT_ERROR, "GPU: Error, no roots of F and G for these "
               "parameters\n");
      return ECM_ERROR;
    }

  /* the low levels of the product trees in Z/N: each costs D + 1 products
     per coefficient, against two conversions and transforms through RNS
     (GMPECM_S2POLY_LOWMAX overrides) */
  {
    const char *env = getenv ("GMPECM_S2POLY_LOWMAX");
    gpu_polyalg_lowmul (PA, gpu_roots_tree, RT,
                        env != NULL ? (size_t) atoi (env) : 16);
    env = getenv ("GMPECM_S2POLY_MPMAX");
    gpu_polyalg_lowmp (PA, gpu_roots_mp, RT,
                       env != NULL ? (size_t) atoi (env) : 16);
  }

  const size_t bytes = (size_t) B * dF * nl * 4;
  uint32_t *d_R[2], *d_F, *d_I, *d_H, *d_G, *d_V;
  std::vector<uint32_t *> tree (l > 0 ? l : 1);
  CUDA_CHECK (cudaMalloc ((void **) &d_R[0], bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_R[1], bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_F, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_I, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_H, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_G, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_V, bytes));
  for (uint32_t *&p : tree)
    CUDA_CHECK (cudaMalloc ((void **) &p, bytes));

  /* The roots of the next block of G are computed on their own stream
     while the polynomial work on the last ones runs (on stream 0): two
     buffers of roots; READ[x] marks the end of the work reading buffer x,
     which the roots must wait for before they overwrite it.
     GMPECM_S2POLY_NOOVERLAP: everything on stream 0. */
  cudaStream_t rstream = 0;
  if (getenv ("GMPECM_S2POLY_NOOVERLAP") == NULL)
    CUDA_CHECK (cudaStreamCreateWithFlags (&rstream, cudaStreamNonBlocking));
  gpu_roots_stream (RT, rstream);
  cudaEvent_t read[2];
  CUDA_CHECK (cudaEventCreateWithFlags (&read[0], cudaEventDisableTiming));
  CUDA_CHECK (cudaEventCreateWithFlags (&read[1], cudaEventDisableTiming));

  outputf (OUTPUT_VERBOSE, "GPU stage 2: polynomial continuation, dF=%lu, "
           "k=%lu, %d NTT primes, %u curves at a time\n", dF, k,
           gpu_rns_primes (R), B);

  cudaEvent_t start, stop;
  CUDA_CHECK (cudaEventCreate (&start));
  CUDA_CHECK (cudaEventCreate (&stop));
  CUDA_CHECK (cudaEventRecord (start));

  std::vector<uint32_t> h ((size_t) B * std::max (dF, 2ul) * nl);
  mpz_t f, v, t;
  mpz_inits (f, v, t, NULL);
  /* the batch: curve indices (repeated to fill it), and shallow copies of
     their A and x for gpu_roots_curves */
  std::vector<uint32_t> idx;
  mpz_t *bA = new mpz_t[B], *bx = new mpz_t[B];
  int youpi = ECM_NO_FACTOR_FOUND;
  s2p_prof prof;
  uint32_t next = 0;
  for (;;)
    {
      /* the curves still to do, B at a time */
      idx.clear ();
      for (; next < curves && idx.size () < B; next++)
        if (array_found[next] == ECM_NO_FACTOR_FOUND)
          idx.push_back (next);
      if (idx.empty ())
        break;
      const size_t live = idx.size ();
      while (idx.size () < B)
        idx.push_back (idx[0]);
      /* a batch where a root meets a factor is redone without its curve */
      for (;;)
        {
          for (uint32_t b = 0; b < B; b++)
            {
              bA[b][0] = As[idx[b]][0];
              bx[b][0] = xs[idx[b]][0];
            }
          int cv = -1, found;
          prof.mark (-1);
          found = gpu_roots_curves (RT, bA, bx, f, &cv);
          prof.mark (PH_RSETUP);
          if (!found)
            found = gpu_roots_F (RT, d_R[0], f, &cv);
          prof.mark (PH_RF);
          int cur = 1;
          if (!found)
            {
              gpu_poly_from_roots (PA, d_F, tree.data (), d_R[0]);
              prof.mark (PH_TREE);
              gpu_poly_inverse (PA, d_I, d_F);
              gpu_poly_prepare (PA, d_F, d_I);
              prof.mark (PH_INV);
              CUDA_CHECK (cudaEventRecord (read[0], 0));
            }
          for (unsigned long b = 0; b < k && !found; b++)
            {
              /* overlaps the polynomial work queued so far */
              CUDA_CHECK (cudaStreamWaitEvent (rstream, read[cur], 0));
              found = gpu_roots_G (RT, d_R[cur], f, &cv);
              prof.mark (PH_RG);
              if (!found)
                {
                  gpu_poly_from_roots (PA, d_G, NULL, d_R[cur]);
                  prof.mark (PH_G);
                  gpu_poly_mulmod (PA, d_H, d_G, b == 0);
                  prof.mark (PH_MULMOD);
                  CUDA_CHECK (cudaEventRecord (read[cur], 0));
                  cur ^= 1;
                }
            }
          if (!found)
            break;
          CUDA_CHECK (cudaDeviceSynchronize ());
          /* the factor of curve cv (or, if unknown, of the first curve) */
          const uint32_t c = idx[cv >= 0 ? cv : 0];
          mpz_set (xs[c], f);
          array_found[c] = ECM_FACTOR_FOUND_STEP2;
          youpi = ECM_FACTOR_FOUND_STEP2;
          uint32_t keep = B;
          for (uint32_t b = 0; b < B; b++)
            if (idx[b] != c)
              keep = b;
          if (keep == B)
            break;              /* no other curve in this batch */
          for (uint32_t b = 0; b < B; b++)
            if (idx[b] == c)
              idx[b] = idx[keep];
        }
      if (std::all_of (idx.begin (), idx.end (), [&] (uint32_t c)
                       { return array_found[c] != ECM_NO_FACTOR_FOUND; }))
        continue;

      /* the values H(-f_i) of every curve, and their products: the value
         at f_0 (what GMP-ECM prints) first, as the product overwrites them */
      gpu_poly_tpeval (PA, d_V, d_H, tree.data ());
      prof.mark (PH_EVAL);
      CUDA_CHECK (cudaMemcpy2D (h.data (), nl * 4, d_V, dF * nl * 4, nl * 4,
                                B, cudaMemcpyDeviceToHost));
      gpu_poly_prodvals (PA, d_G, d_V);
      CUDA_CHECK (cudaMemcpy (h.data () + (size_t) B * nl, d_G,
                              (size_t) B * nl * 4, cudaMemcpyDeviceToHost));
      for (size_t b = 0; b < live; b++)
        {
          const uint32_t c = idx[b];
          if (array_found[c] != ECM_NO_FACTOR_FOUND
              || std::find (idx.begin (), idx.begin () + b, c)
                 != idx.begin () + b)
            continue;           /* found, or a repeat of a redone batch */
          gpu_from_limbs (v, &h[(B + b) * nl], nl);
          mpz_gcd (f, v, N);
          if (mpz_cmp_ui (f, 1) > 0)
            {
              mpz_set (xs[c], f);
              array_found[c] = ECM_FACTOR_FOUND_STEP2;
              youpi = ECM_FACTOR_FOUND_STEP2;
            }
          else
            {
              gpu_from_limbs (t, &h[b * nl], nl);
              mpz_mod (t, t, N);
              outputf (OUTPUT_RESVERBOSE, "Product of G(f_i) = %Zd\n", t);
            }
        }
      prof.mark (PH_HOST);
    }

  CUDA_CHECK (cudaEventRecord (stop));
  CUDA_CHECK (cudaEventSynchronize (stop));
  prof.print (curves);
  CUDA_CHECK (cudaEventElapsedTime (gputime, start, stop));
  CUDA_CHECK (cudaEventDestroy (start));
  CUDA_CHECK (cudaEventDestroy (stop));

  delete[] bA;
  delete[] bx;
  mpz_clears (f, v, t, NULL);
  for (uint32_t *p : tree)
    CUDA_CHECK (cudaFree (p));
  CUDA_CHECK (cudaFree (d_R[0]));
  CUDA_CHECK (cudaFree (d_R[1]));
  CUDA_CHECK (cudaEventDestroy (read[0]));
  CUDA_CHECK (cudaEventDestroy (read[1]));
  gpu_roots_stream (RT, 0);
  if (rstream != 0)
    CUDA_CHECK (cudaStreamDestroy (rstream));
  CUDA_CHECK (cudaFree (d_F));
  CUDA_CHECK (cudaFree (d_I));
  CUDA_CHECK (cudaFree (d_H));
  CUDA_CHECK (cudaFree (d_G));
  CUDA_CHECK (cudaFree (d_V));
  gpu_roots_destroy (RT);
  gpu_polyalg_destroy (PA);
  gpu_rns_destroy (R);
  (void) verbose;
  return youpi;
}
