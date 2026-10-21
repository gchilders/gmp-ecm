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
 * One curve at a time for now: a batch dimension (several curves per
 * launch) is for the tuning milestone. */

#include "gpu_s2poly.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <gmp.h>
#include <cuda.h>

#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_rns.h"
#include "gpu_polyalg.h"
#include "gpu_roots.h"

#include "ecm.h"
#include "ecm-gpu.h"

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
  gpu_rns *R = gpu_rns_create (N, l + 1);
  int ok = gpu_rns_lg (R) >= l + 1;
  gpu_rns_destroy (R);
  return ok;
}

double
gpu_ecm_stage2_poly_mem (const mpz_t N, unsigned long dF)
{
  const double n = (double) dF, l = s2p_lg (dF);
  const double nl = (double) ((mpz_sizeinbase (N, 2) + 4 + 31) / 32);
  const double P = (2 * 32 * nl + l + 1) / 29 + 4;   /* NTT primes */
  double bytes = (l + 6) * n * nl * 4    /* tree, F, I, H, G, values, roots */
                 + 12 * n * nl * 4       /* polyalg scratch */
                 + 10 * P * n * 4        /* RNS transforms and scratch */
                 + 3 * n * (nl + 32) * 4 /* roots: numerators, inversions */
                 + 64e6;                 /* roots' tables, slack */
  return 1.25 * bytes;
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
  gpu_rns *R = gpu_rns_create (N, l + 1);
  const int nl = gpu_rns_limbs (R);
  gpu_polyalg *PA = gpu_polyalg_create (R, dF);
  gpu_roots *RT = gpu_roots_create (N, 0, 0, S, d1, d2, i0, dF, k, nl, 1,
                                    0);
  if (RT == NULL)
    {
      gpu_polyalg_destroy (PA);
      gpu_rns_destroy (R);
      outputf (OUTPUT_ERROR, "GPU: Error, no roots of F and G for these "
               "parameters\n");
      return ECM_ERROR;
    }

  const size_t bytes = (size_t) dF * nl * 4;
  uint32_t *d_roots, *d_F, *d_I, *d_H, *d_G, *d_V;
  std::vector<uint32_t *> tree (l > 0 ? l : 1);
  CUDA_CHECK (cudaMalloc ((void **) &d_roots, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_F, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_I, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_H, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_G, bytes));
  CUDA_CHECK (cudaMalloc ((void **) &d_V, bytes));
  for (uint32_t *&p : tree)
    CUDA_CHECK (cudaMalloc ((void **) &p, bytes));

  outputf (OUTPUT_VERBOSE, "GPU stage 2: polynomial continuation, dF=%lu, "
           "k=%lu, %d NTT primes\n", dF, k, gpu_rns_primes (R));

  cudaEvent_t start, stop;
  CUDA_CHECK (cudaEventCreate (&start));
  CUDA_CHECK (cudaEventCreate (&stop));
  CUDA_CHECK (cudaEventRecord (start));

  std::vector<uint32_t> h ((size_t) dF * nl);
  mpz_t f, v, t;
  mpz_inits (f, v, t, NULL);
  int youpi = ECM_NO_FACTOR_FOUND;
  for (uint32_t c = 0; c < curves; c++)
    {
      int cv, found;
      if (array_found[c] != ECM_NO_FACTOR_FOUND)
        continue;
      found = gpu_roots_curves (RT, &As[c], &xs[c], f, &cv);
      if (!found)
        found = gpu_roots_F (RT, d_roots, f, &cv);
      if (!found)
        {
          gpu_poly_from_roots (PA, d_F, tree.data (), d_roots);
          gpu_poly_inverse (PA, d_I, d_F);
          gpu_poly_prepare (PA, d_F, d_I);
        }
      for (unsigned long b = 0; b < k && !found; b++)
        {
          found = gpu_roots_G (RT, d_roots, f, &cv);
          if (!found)
            {
              gpu_poly_from_roots (PA, d_G, NULL, d_roots);
              gpu_poly_mulmod (PA, d_H, d_G, b == 0);
            }
        }
      if (!found)
        {
          /* the values H(-f_i) and their product */
          gpu_poly_tpeval (PA, d_V, d_H, tree.data ());
          CUDA_CHECK (cudaMemcpy (h.data (), d_V, bytes,
                                  cudaMemcpyDeviceToHost));
          mpz_set_ui (v, 1);
          for (unsigned long i = 0; i < dF; i++)
            {
              gpu_from_limbs (t, &h[i * nl], nl);
              mpz_mul (v, v, t);
              mpz_mod (v, v, N);
            }
          mpz_gcd (f, v, N);
          if (mpz_cmp_ui (f, 1) > 0)
            found = 1;
          else
            {
              gpu_from_limbs (t, h.data (), nl);
              mpz_mod (t, t, N);
              outputf (OUTPUT_RESVERBOSE, "Product of G(f_i) = %Zd\n", t);
            }
        }
      if (found)
        {
          mpz_set (xs[c], f);
          array_found[c] = ECM_FACTOR_FOUND_STEP2;
          youpi = ECM_FACTOR_FOUND_STEP2;
        }
    }

  CUDA_CHECK (cudaEventRecord (stop));
  CUDA_CHECK (cudaEventSynchronize (stop));
  CUDA_CHECK (cudaEventElapsedTime (gputime, start, stop));
  CUDA_CHECK (cudaEventDestroy (start));
  CUDA_CHECK (cudaEventDestroy (stop));

  mpz_clears (f, v, t, NULL);
  for (uint32_t *p : tree)
    CUDA_CHECK (cudaFree (p));
  CUDA_CHECK (cudaFree (d_roots));
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
