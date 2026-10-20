/* gpu_stage1.cu: GPU ECM stage 1 with run-time sized arithmetic.

Copyright 2021 Seth Troisi, 2026 Greg Childers

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

/* Each curve's numbers have n = TPI * L 32-bit limbs, spread over TPI
 * threads (see gpu_mont.cuh).  Kernels are compiled for every L up to
 * GPU_SOLO_MAX_L (TPI = 1) and GPU_DIST_MAX_L (TPI = 2..32, chosen at run
 * time), so any N below 2^(32 * 32 * GPU_DIST_MAX_L - 4) runs without
 * recompiling.  Which (TPI, L) to use for a given N is decided by timing
 * the candidates on the GPU (gpu_ecm_stage1_plan).
 *
 * Bounds (R = 2^(32n), 16N <= R).  Montgomery products whose inputs
 * multiply to less than 16N^2 come out below 2N.  One ladder step, with
 * inputs aX, aZ, bX < 2N and bZ < 4N:
 *
 *   t1 = bZ + bX           < 6N
 *   v  = bZ - bX (+2N)     < 4N     (sub adds 2N iff the difference < 0)
 *   w  = aZ + aX           < 4N
 *   u  = aZ - aX (+2N)     < 2N
 *   CB = t1 * u            12N^2 -> < 2N
 *   DA = v * w             16N^2 -> < 2N
 *   AA = w^2, BB = u^2             < 2N
 *   K  = AA - BB (+2N)     < 2N
 *   aX' = AA * BB                  < 2N
 *   dK = K * d / 2^32      < K + N < 3N
 *   aZ' = K * (BB + dK)    10N^2 -> < 2N
 *   bX' = (DA + CB)^2      16N^2 -> < 2N
 *   bZ' = 2 (DA - CB)^2            < 4N
 */

#ifndef __CUDACC__
#error "This file should only be compiled with nvcc"
#endif

#include "gpu_stage1.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <gmp.h>
#include <cuda.h>

#include "cudacommon.h"
#include "gpu_mont.cuh"

#include "ecm.h"
#include "ecm-gpu.h"

#define GPU_TPB 128
#define GPU_SOLO_MAX_L 16
#define GPU_DIST_MAX_L 24
/* N must be below 2^(32n - GPU_SPARE_BITS) */
#define GPU_SPARE_BITS 4

/* ---------------------------------------------------------------------- */

template<int L, bool SOLO>
__device__ __forceinline__ void
ladder_step (const gm_group &g,
             uint32_t (&aX)[L], uint32_t (&aZ)[L],
             uint32_t (&bX)[L], uint32_t (&bZ)[L],
             uint32_t d, const uint32_t (&N)[L], const uint32_t (&N2)[L],
             uint32_t np0)
{
  uint32_t t1[L], t2[L];

  gm_add<L, SOLO> (g, t1, bZ, bX);
  gm_sub<L, SOLO> (g, bZ, bZ, bX, N2);
  gm_add<L, SOLO> (g, bX, aZ, aX);
  gm_sub<L, SOLO> (g, aZ, aZ, aX, N2);
  gm_mul<L, SOLO> (g, aX, t1, aZ, N, np0);     /* aX = CB */
  gm_mul<L, SOLO> (g, t1, bZ, bX, N, np0);     /* t1 = DA */
  gm_sqr<L, SOLO> (g, bZ, bX, N, np0);         /* bZ = AA */
  gm_sqr<L, SOLO> (g, bX, aZ, N, np0);         /* bX = BB */
  gm_sub<L, SOLO> (g, aZ, bZ, bX, N2);         /* aZ = K */
  gm_mul<L, SOLO> (g, bZ, bZ, bX, N, np0);     /* bZ = AA*BB = new aX */
  gm_mul_ui<L, SOLO> (g, t2, aZ, d, N, np0);   /* t2 = dK */
  gm_add<L, SOLO> (g, bX, bX, t2);             /* bX = BB + dK */
  gm_mul<L, SOLO> (g, aZ, aZ, bX, N, np0);     /* aZ = new aZ */
  gm_add<L, SOLO> (g, bX, t1, aX);             /* bX = DA + CB */
  gm_sub<L, SOLO> (g, t1, t1, aX, N2);         /* t1 = DA - CB */
  gm_sqr<L, SOLO> (g, bX, bX, N, np0);         /* bX = new bX */
  gm_sqr<L, SOLO> (g, t1, t1, N, np0);
  gm_dbl<L, SOLO> (g, t1, t1);                 /* t1 = new bZ */

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      aX[k] = bZ[k];
      bZ[k] = t1[k];
    }
}

template<int L>
__device__ __forceinline__ void
swap_limbs (uint32_t (&a)[L], uint32_t (&b)[L])
{
#pragma unroll
  for (int k = 0; k < L; k++)
    {
      uint32_t t = a[k];
      a[k] = b[k];
      b[k] = t;
    }
}

/* Process bits [b_start, b_start + b_count) of s, most significant first.
 * data holds, per curve, aX, aZ, bX, bZ (n limbs each, Montgomery form).
 * Every thread of the grid owns a real (or padding) curve: no early exits,
 * the shuffles need whole warps. */
template<int L, bool SOLO>
__global__ void __launch_bounds__ (GPU_TPB)
gpu_stage1_kernel (const uint32_t * __restrict__ s_bits, uint64_t s_nbits,
                   uint64_t b_start, uint64_t b_count,
                   uint32_t * __restrict__ data,
                   const uint32_t * __restrict__ Ng,
                   uint32_t np0, uint32_t sigma0, uint32_t tpi)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);

  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / g.tpi;
  const uint32_t n = g.tpi * L;
  const uint32_t off = g.lane * L;

  uint32_t N[L], N2[L], aX[L], aZ[L], bX[L], bZ[L];
  uint32_t *dp = data + (size_t) inst * 4 * n + off;

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      N[k] = Ng[off + k];
      aX[k] = dp[k];
      aZ[k] = dp[n + k];
      bX[k] = dp[2 * n + k];
      bZ[k] = dp[3 * n + k];
    }
  gm_dbl<L, SOLO> (g, N2, N);

  const uint32_t d = sigma0 + inst;

  int swapped = 0;
  for (uint64_t b = b_start; b < b_start + b_count; b++)
    {
      uint64_t nth = s_nbits - 1 - b;
      int bit = (s_bits[nth >> 5] >> (nth & 31)) & 1;
      if (bit != swapped)
        {
          swapped = bit;
          swap_limbs<L> (aX, bX);
          swap_limbs<L> (aZ, bZ);
        }
      ladder_step<L, SOLO> (g, aX, aZ, bX, bZ, d, N, N2, np0);
    }
  if (swapped)
    {
      swap_limbs<L> (aX, bX);
      swap_limbs<L> (aZ, bZ);
    }

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      dp[k] = aX[k];
      dp[n + k] = aZ[k];
      dp[2 * n + k] = bX[k];
      dp[3 * n + k] = bZ[k];
    }
}

/* ---------------------------------------------------------------------- */

typedef void (*gpu_kernel_fn) (const uint32_t *, uint64_t, uint64_t, uint64_t,
                               uint32_t *, const uint32_t *,
                               uint32_t, uint32_t, uint32_t);

/* kernels[L] for L = 0..MAX_L (index 0 unused) */
template<int L, bool SOLO>
struct kernel_table {
  static void fill (gpu_kernel_fn *t)
  {
    t[L] = gpu_stage1_kernel<L, SOLO>;
    kernel_table<L - 1, SOLO>::fill (t);
  }
};
template<bool SOLO>
struct kernel_table<0, SOLO> {
  static void fill (gpu_kernel_fn *t) { t[0] = NULL; }
};

static gpu_kernel_fn
get_kernel (uint32_t tpi, uint32_t L)
{
  static gpu_kernel_fn solo[GPU_SOLO_MAX_L + 1];
  static gpu_kernel_fn dist[GPU_DIST_MAX_L + 1];
  static int init = 0;
  if (!init)
    {
      kernel_table<GPU_SOLO_MAX_L, true>::fill (solo);
      kernel_table<GPU_DIST_MAX_L, false>::fill (dist);
      init = 1;
    }
  if (tpi == 1)
    return (L >= 1 && L <= GPU_SOLO_MAX_L) ? solo[L] : NULL;
  return (L >= 1 && L <= GPU_DIST_MAX_L) ? dist[L] : NULL;
}

struct gpu_plan_t {
  uint32_t n_min;        /* key: minimal limb count */
  uint32_t req_curves;   /* key: requested curves (0 = fill GPU) */
  uint32_t tpi, L;
  uint32_t curves;
  gpu_kernel_fn fn;
};

static std::vector<gpu_plan_t> plans;

static uint32_t
round_up (uint32_t a, uint32_t m)
{
  return (a + m - 1) / m * m;
}

static void
launch (const gpu_plan_t &p, uint32_t padded, const uint32_t *gpu_s_bits,
        uint64_t s_nbits, uint64_t start, uint64_t count, uint32_t *gpu_data,
        const uint32_t *gpu_N, uint32_t np0, uint32_t sigma)
{
  uint32_t blocks = padded * p.tpi / GPU_TPB;
  p.fn<<<blocks, GPU_TPB>>> (gpu_s_bits, s_nbits, start, count, gpu_data,
                             gpu_N, np0, sigma, p.tpi);
  CUDA_CHECK (cudaGetLastError ());
}

static uint32_t
padded_curves (const gpu_plan_t &p, uint32_t curves)
{
  return round_up (curves, GPU_TPB / p.tpi);
}

/* Curves needed to fill the GPU with kernel FN at TPI threads per curve. */
static uint32_t
fill_curves (gpu_kernel_fn fn, uint32_t tpi)
{
  int device, sms, blocks = 0;
  CUDA_CHECK (cudaGetDevice (&device));
  CUDA_CHECK (cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount,
                                      device));
  CUDA_CHECK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (
                  &blocks, (const void *) fn, GPU_TPB, 0));
  if (blocks < 1)
    blocks = 1;
  return (uint32_t) blocks * sms * (GPU_TPB / tpi);
}

/* Seconds per (curve * bit) for plan P, measured on zeroed data.  The
 * kernels have no data-dependent branches, so zeros time like real data. */
static double
time_plan (const gpu_plan_t &p, uint32_t n_min)
{
  const uint32_t n = p.tpi * p.L;
  const uint32_t padded = padded_curves (p, p.curves);
  const uint64_t s_nbits = 4096;
  uint32_t *gpu_s_bits, *gpu_data, *gpu_N;
  uint32_t s_host[s_nbits / 32];
  for (unsigned i = 0; i < s_nbits / 32; i++)
    s_host[i] = 0x9e3779b9u * (i + 1);

  CUDA_CHECK (cudaMalloc ((void **) &gpu_s_bits, sizeof (s_host)));
  CUDA_CHECK (cudaMemcpy (gpu_s_bits, s_host, sizeof (s_host),
                          cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMalloc ((void **) &gpu_data,
                          (size_t) padded * 4 * n * sizeof (uint32_t)));
  CUDA_CHECK (cudaMemset (gpu_data, 0,
                          (size_t) padded * 4 * n * sizeof (uint32_t)));
  CUDA_CHECK (cudaMalloc ((void **) &gpu_N, n * sizeof (uint32_t)));
  CUDA_CHECK (cudaMemset (gpu_N, 0, n * sizeof (uint32_t)));

  cudaEvent_t start, stop;
  CUDA_CHECK (cudaEventCreate (&start));
  CUDA_CHECK (cudaEventCreate (&stop));

  /* warm up (module load), then grow the bit count until the run takes
     long enough to time */
  launch (p, padded, gpu_s_bits, s_nbits, 0, 1, gpu_data, gpu_N, 1, 2);
  CUDA_CHECK (cudaDeviceSynchronize ());

  uint64_t bits = 4;
  float ms = 0;
  for (;;)
    {
      CUDA_CHECK (cudaEventRecord (start));
      launch (p, padded, gpu_s_bits, s_nbits, 0, bits, gpu_data, gpu_N, 1, 2);
      CUDA_CHECK (cudaEventRecord (stop));
      CUDA_CHECK (cudaEventSynchronize (stop));
      CUDA_CHECK (cudaEventElapsedTime (&ms, start, stop));
      if (ms >= 20 || bits >= s_nbits)
        break;
      uint64_t next = (uint64_t) (bits * 30.0 / (ms > 0.1 ? ms : 0.1));
      bits = next > 2 * bits ? (next < s_nbits ? next : s_nbits) : 2 * bits;
    }

  CUDA_CHECK (cudaEventDestroy (start));
  CUDA_CHECK (cudaEventDestroy (stop));
  CUDA_CHECK (cudaFree (gpu_s_bits));
  CUDA_CHECK (cudaFree (gpu_data));
  CUDA_CHECK (cudaFree (gpu_N));

  (void) n_min;
  return ms / 1000.0 / ((double) bits * p.curves);
}

extern "C"
size_t
gpu_ecm_stage1_max_bits (void)
{
  return 32 * 32 * GPU_DIST_MAX_L - GPU_SPARE_BITS - 1;
}

extern "C"
int
gpu_ecm_stage1_plan (const mpz_t N, unsigned int *curves, int verbose)
{
  size_t bits = mpz_sizeinbase (N, 2);
  if (bits > gpu_ecm_stage1_max_bits ())
    {
      outputf (OUTPUT_ERROR, "GPU: Error, input number should be at most "
               "%zu bits\n", gpu_ecm_stage1_max_bits ());
      return ECM_ERROR;
    }
  uint32_t n_min = (uint32_t) ((bits + GPU_SPARE_BITS + 31) / 32);

  for (size_t i = 0; i < plans.size (); i++)
    if (plans[i].n_min == n_min && plans[i].req_curves == *curves)
      {
        if (*curves == 0)
          *curves = plans[i].curves;
        return (int) i;
      }

  /* GMPECM_GPU_KERNEL=TPI:L or TPI forces a configuration */
  uint32_t force_tpi = 0, force_L = 0;
  const char *env = getenv ("GMPECM_GPU_KERNEL");
  if (env != NULL && *env != '\0')
    {
      if (sscanf (env, "%u:%u", &force_tpi, &force_L) < 1)
        force_tpi = 0;
    }

  std::vector<gpu_plan_t> cand;
  for (uint32_t tpi = 1; tpi <= 32; tpi *= 2)
    {
      if (force_tpi && tpi != force_tpi)
        continue;
      uint32_t L = (n_min + tpi - 1) / tpi;
      if (force_L)
        {
          if (force_L < L)
            continue;
          L = force_L;
        }
      else if (tpi > 1 && L < 2 && n_min > 1)
        continue;   /* never better than fewer threads with more limbs */
      gpu_kernel_fn fn = get_kernel (tpi, L);
      if (fn == NULL)
        continue;
      gpu_plan_t p;
      p.n_min = n_min;
      p.req_curves = *curves;
      p.tpi = tpi;
      p.L = L;
      p.fn = fn;
      p.curves = *curves ? *curves : fill_curves (fn, tpi);
      cand.push_back (p);
    }

  if (cand.empty ())
    {
      outputf (OUTPUT_ERROR, "GPU: no kernel for %zu-bit input%s\n", bits,
               force_tpi ? " (check GMPECM_GPU_KERNEL)" : "");
      return ECM_ERROR;
    }

  size_t best = 0;
  if (cand.size () > 1)
    {
      double best_t = 0;
      for (size_t i = 0; i < cand.size (); i++)
        {
          double t = time_plan (cand[i], n_min);
          if (verbose >= OUTPUT_VERBOSE)
            printf ("GPU: kernel TPI=%u L=%u (%u bits): %u curves, "
                    "%.3g us per curve-bit\n", cand[i].tpi, cand[i].L,
                    32 * cand[i].tpi * cand[i].L, cand[i].curves, t * 1e6);
          if (i == 0 || t < best_t)
            {
              best = i;
              best_t = t;
            }
        }
    }

  plans.push_back (cand[best]);
  if (*curves == 0)
    *curves = cand[best].curves;
  return (int) plans.size () - 1;
}

/* ---------------------------------------------------------------------- */

static void
to_limbs (uint32_t *x, uint32_t n, const mpz_t s)
{
  size_t words = 0;
  assert (mpz_sgn (s) >= 0 && mpz_sizeinbase (s, 2) <= 32 * (size_t) n);
  mpz_export (x, &words, -1, sizeof (uint32_t), 0, 0, s);
  while (words < n)
    x[words++] = 0;
}

static void
from_limbs (mpz_t r, const uint32_t *x, uint32_t n)
{
  mpz_import (r, n, -1, sizeof (uint32_t), 0, 0, x);
}

static uint32_t
find_np0 (const mpz_t N)
{
  uint32_t np0;
  mpz_t t;
  mpz_init (t);
  mpz_ui_pow_ui (t, 2, 32);
  int ok = mpz_invert (t, N, t);
  assert (ok);
  (void) ok;
  np0 = - (uint32_t) mpz_get_ui (t);
  mpz_clear (t);
  return np0;
}

static int
print_nth_batch (int n)
{
  return ((n < 3) ||
          (n < 30 && n % 10 == 0) ||
          (n < 500 && n % 100 == 0) ||
          (n < 5000 && n % 1000 == 0) ||
          (n % 10000 == 0));
}

extern "C"
int
gpu_ecm_stage1 (mpz_t *factors, int *array_found,
                const mpz_t N, const mpz_t s,
                uint32_t curves, uint32_t sigma, int config,
                float *gputime, int verbose)
{
  assert (sigma > 0);
  assert (((uint64_t) sigma + curves) <= 0xFFFFFFFF);
  assert (config >= 0 && (size_t) config < plans.size ());
  const gpu_plan_t &p = plans[config];

  if (mpz_even_p (N))
    {
      outputf (OUTPUT_ERROR, "GPU: Error, N must be odd\n");
      return ECM_ERROR;
    }

  const uint32_t n = p.tpi * p.L;
  const uint32_t padded = padded_curves (p, curves);
  assert (mpz_sizeinbase (N, 2) + GPU_SPARE_BITS <= 32 * (size_t) n);

  /* s, most significant bit first in the kernel */
  uint64_t s_nbits = mpz_sizeinbase (s, 2);
  uint64_t s_words = (s_nbits + 31) / 32;
  if (s_nbits >= 4000000000ULL)
    outputf (OUTPUT_ALWAYS, "GPU: Very Large B1! Check magnitute of B1.\n");
  if (s_nbits >= 100000000)
    outputf (OUTPUT_NORMAL, "GPU: Large B1, S = %'lu bits = %d MB\n",
             s_nbits, (int) (s_nbits >> 23));
  uint32_t *s_bits = (uint32_t *) malloc (s_words * sizeof (uint32_t));
  assert (s_bits != NULL);
  {
    size_t cnt;
    mpz_export (s_bits, &cnt, -1, sizeof (uint32_t), 0, 0, s);
    assert (cnt == s_words);
  }

  /* Initial points in Montgomery form (R = 2^(32n)):
   *   P  = (2 : 1)
   *   2P = (9 : 64 d + 8), d = sigma_i / 2^32 mod N   */
  size_t data_words = (size_t) padded * 4 * n;
  uint32_t *data = (uint32_t *) malloc (data_words * sizeof (uint32_t));
  assert (data != NULL);
  {
    mpz_t R, x, inv32;
    mpz_init (R);
    mpz_init (x);
    mpz_init (inv32);
    mpz_setbit (R, 32 * (size_t) n);
    mpz_mod (R, R, N);
    mpz_ui_pow_ui (inv32, 2, 32);
    mpz_invert (inv32, inv32, N);

    mpz_mul_ui (x, R, 2);
    mpz_mod (x, x, N);
    uint32_t *P0 = data;
    to_limbs (P0, n, x);                 /* aX */
    to_limbs (P0 + n, n, R);             /* aZ */
    mpz_mul_ui (x, R, 9);
    mpz_mod (x, x, N);
    to_limbs (P0 + 2 * n, n, x);         /* bX */

    for (uint32_t i = 0; i < padded; i++)
      {
        uint32_t *dp = data + (size_t) i * 4 * n;
        if (i > 0)
          memcpy (dp, P0, 3 * n * sizeof (uint32_t));
        mpz_mul_ui (x, inv32, sigma + i);
        mpz_mul_ui (x, x, 64);
        mpz_add_ui (x, x, 8);
        mpz_mod (x, x, N);
        if (i == 0)
          outputf (OUTPUT_TRACE, "sigma %u => P2_z: %Zd\n", sigma, x);
        mpz_mul (x, x, R);
        mpz_mod (x, x, N);
        to_limbs (dp + 3 * n, n, x);     /* bZ */
      }
    mpz_clear (R);
    mpz_clear (x);
    mpz_clear (inv32);
  }

  uint32_t *Nlimbs = (uint32_t *) malloc (n * sizeof (uint32_t));
  to_limbs (Nlimbs, n, N);
  uint32_t np0 = find_np0 (N);

  cudaEvent_t global_start, batch_start, stop;
  CUDA_CHECK (cudaEventCreate (&global_start));
  CUDA_CHECK (cudaEventCreate (&batch_start));
  CUDA_CHECK (cudaEventCreate (&stop));
  CUDA_CHECK (cudaEventRecord (global_start));

  uint32_t *gpu_s_bits, *gpu_data, *gpu_N;
  CUDA_CHECK (cudaMalloc ((void **) &gpu_s_bits, s_words * sizeof (uint32_t)));
  CUDA_CHECK (cudaMemcpy (gpu_s_bits, s_bits, s_words * sizeof (uint32_t),
                          cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMalloc ((void **) &gpu_N, n * sizeof (uint32_t)));
  CUDA_CHECK (cudaMemcpy (gpu_N, Nlimbs, n * sizeof (uint32_t),
                          cudaMemcpyHostToDevice));
  outputf (OUTPUT_VERBOSE, "Copying %'lu bytes of curves data to GPU\n",
           data_words * sizeof (uint32_t));
  CUDA_CHECK (cudaMalloc ((void **) &gpu_data, data_words * sizeof (uint32_t)));
  CUDA_CHECK (cudaMemcpy (gpu_data, data, data_words * sizeof (uint32_t),
                          cudaMemcpyHostToDevice));

  kernel_info ((const void *) p.fn, verbose);
  outputf (OUTPUT_VERBOSE, "GPU: %u-bit arithmetic (TPI=%u, L=%u), "
           "%u blocks x %u threads, input number is %zu bits\n",
           32 * n, p.tpi, p.L, padded * p.tpi / GPU_TPB, GPU_TPB,
           mpz_sizeinbase (N, 2));

  /* The first bit (doubling) is handled by the initial 2P. */
  uint64_t s_partial = 1;
  /* Start with small batches and increase till timing is ~100ms */
  uint64_t batch_size = 200;
  int batches_complete = 0;
  float batch_time = 0;

  while (s_partial < s_nbits)
    {
      batch_size = std::min (s_nbits - s_partial, batch_size);

      if (print_nth_batch (batches_complete))
        {
          outputf (OUTPUT_VERBOSE, "Computing %lu bits/call, %lu/%lu (%.1f%%)",
                   batch_size, s_partial, s_nbits,
                   100.0 * s_partial / s_nbits);
          if (batches_complete < 2 || *gputime < 1000)
            outputf (OUTPUT_VERBOSE, "\n");
          else
            {
              float estimated_total = (*gputime) * ((float) s_nbits) / s_partial;
              float eta = estimated_total - (*gputime);
              outputf (OUTPUT_VERBOSE, ", ETA %.f + %.f = %.f seconds "
                       "(~%.f ms/curves)\n", eta / 1000, *gputime / 1000,
                       estimated_total / 1000, estimated_total / curves);
            }
        }

      CUDA_CHECK (cudaEventRecord (batch_start));
      launch (p, padded, gpu_s_bits, s_nbits, s_partial, batch_size,
              gpu_data, gpu_N, np0, sigma);

      s_partial += batch_size;
      batches_complete++;

      CUDA_CHECK (cudaEventRecord (stop));
      CUDA_CHECK (cudaEventSynchronize (stop));
      cudaEventElapsedTime (&batch_time, batch_start, stop);
      cudaEventElapsedTime (gputime, global_start, stop);
      /* Adjust batch_size to aim for 100ms */
      if (batch_time < 80)
        batch_size = 11 * batch_size / 10;
      else if (batch_time > 120)
        batch_size = std::max ((uint64_t) 100, 9 * batch_size / 10);
    }

  outputf (OUTPUT_VERBOSE, "Copying results back to CPU ...\n");
  CUDA_CHECK (cudaMemcpy (data, gpu_data, data_words * sizeof (uint32_t),
                          cudaMemcpyDeviceToHost));
  cudaEventElapsedTime (gputime, global_start, stop);

  /* Results: x = aX / aZ (Montgomery factors cancel). */
  int youpi = ECM_NO_FACTOR_FOUND;
  int errors = 0;
  {
    mpz_t x, z, Rinv;
    mpz_init (x);
    mpz_init (z);
    mpz_init (Rinv);
    mpz_setbit (Rinv, 32 * (size_t) n);
    mpz_invert (Rinv, Rinv, N);

    for (uint32_t i = 0; i < curves; i++)
      {
        const uint32_t *dp = data + (size_t) i * 4 * n;
        from_limbs (x, dp, n);
        from_limbs (z, dp + n, n);
        mpz_mul (x, x, Rinv);
        mpz_mod (x, x, N);
        mpz_mul (z, z, Rinv);
        mpz_mod (z, z, N);

        if (i == 0 && test_verbose (OUTPUT_TRACE))
          outputf (OUTPUT_TRACE, "index: 0 pA: (%Zd, %Zd)\n", x, z);

        /* (2 : 1) is the starting point: the kernel didn't run */
        if (mpz_cmp_ui (x, 2) == 0 && mpz_cmp_ui (z, 1) == 0)
          {
            errors += 1;
            if (errors < 10 || errors % 100 == 1)
              outputf (OUTPUT_ERROR, "GPU: curve %u didn't compute?\n", i);
          }

        if (mpz_invert (factors[i], z, N))
          {
            mpz_mul (factors[i], x, factors[i]);
            mpz_mod (factors[i], factors[i], N);
            array_found[i] = ECM_NO_FACTOR_FOUND;
          }
        else
          {
            mpz_gcd (factors[i], z, N);
            array_found[i] = ECM_FACTOR_FOUND_STEP1;
            youpi = ECM_FACTOR_FOUND_STEP1;
            outputf (OUTPUT_NORMAL, "GPU: factor %Zd found in Step 1 with "
                     "curve %u (-sigma %d:%u)\n", factors[i], i,
                     ECM_PARAM_BATCH_32BITS_D, sigma + i);
          }
      }
    mpz_clear (x);
    mpz_clear (z);
    mpz_clear (Rinv);
  }
  if (errors > 2)
    youpi = ECM_ERROR;

  CUDA_CHECK (cudaFree (gpu_s_bits));
  CUDA_CHECK (cudaFree (gpu_data));
  CUDA_CHECK (cudaFree (gpu_N));
  CUDA_CHECK (cudaEventDestroy (global_start));
  CUDA_CHECK (cudaEventDestroy (batch_start));
  CUDA_CHECK (cudaEventDestroy (stop));

  free (s_bits);
  free (data);
  free (Nlimbs);

  return youpi;
}

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ < 350
    #error "Unsupported architecture"
  #endif
#endif
