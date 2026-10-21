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
 *
 * That is the unswapped order.  When the bit changes the two points swap,
 * and the doubled bZ < 4N comes in as aZ: then w < 6N, AA < 3.25N and
 * aZ' < 2.3N, above the 2N of the table.  Iterating the worst cases (a
 * product of x and y is below x y / R + N) over any pattern of swaps, the
 * inputs settle below aX < 1.2N, aZ < 1.6N, bX < 1.6N, bZ < 2.5N: every
 * subtrahend stays below 2N, as gm_sub needs, and every sum below 4.5N,
 * far from R.  Nothing else is needed for the results mod N to be right.
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
#include <algorithm>
#include <cuda.h>

#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_kernels.h"
#include "gpu_ntt.h"

#include "ecm.h"
#include "ecm-gpu.h"

struct gpu_plan_t {
  uint32_t n_min;        /* key: minimal limb count */
  uint32_t req_curves;   /* key: requested curves (0 = fill GPU) */
  uint32_t tpi, L;       /* schoolbook shape (tpi = 0 for the NTT path) */
  uint32_t curves;
  gpu_s1_fn fn;
  gpu_ntt_plan *ntt;     /* NTT path: tables for this N, else NULL */
  mpz_t ntt_N;           /* key of an NTT plan: the number itself */
};

static std::vector<gpu_plan_t> plans;

/* (n_min, requested curves) for which the NTT beat the wide kernels */
static std::vector<std::pair<uint32_t, uint32_t> > ntt_wins;

static uint32_t
round_up (uint32_t a, uint32_t m)
{
  return (a + m - 1) / m * m;
}

static void
launch (const gpu_plan_t &p, uint32_t padded, const uint32_t *gpu_s_bits,
        uint64_t s_nbits, uint64_t start, uint64_t count, uint32_t *gpu_data,
        const uint32_t *gpu_N, uint32_t np0, uint32_t sigma,
        const uint32_t *gpu_dlist = NULL, const uint32_t *gpu_x0 = NULL)
{
  if (p.ntt != NULL)
    {
      gpu_ntt_launch (p.ntt, padded, gpu_s_bits, s_nbits, start, count,
                      gpu_data, sigma, gpu_dlist, gpu_x0);
      return;
    }
  uint32_t blocks = padded * p.tpi / GPU_TPB;
  p.fn<<<blocks, GPU_TPB>>> (gpu_s_bits, s_nbits, start, count, gpu_data,
                             gpu_N, np0, sigma, p.tpi, gpu_dlist, gpu_x0);
  CUDA_CHECK (cudaGetLastError ());
}

static uint32_t
padded_curves (const gpu_plan_t &p, uint32_t curves)
{
  if (p.ntt != NULL)
    return curves;   /* one thread block per curve */
  return round_up (curves, GPU_TPB / p.tpi);
}

/* limbs per number */
static uint32_t
plan_limbs (const gpu_plan_t &p)
{
  return p.ntt != NULL ? gpu_ntt_limbs (p.ntt) : p.tpi * p.L;
}

/* Curves needed to fill the GPU with kernel FN at TPI threads per curve. */
static uint32_t
fill_curves (gpu_s1_fn fn, uint32_t tpi)
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
  const uint32_t n = plan_limbs (p);
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
void
gpu_ecm_plan_shape (int config, uint32_t *tpi, uint32_t *L)
{
  assert (config >= 0 && (size_t) config < plans.size ());
  *tpi = plans[config].tpi;
  *L = plans[config].L;
}

extern "C"
struct gpu_ntt_plan *
gpu_ecm_plan_ntt (int config)
{
  assert (config >= 0 && (size_t) config < plans.size ());
  return plans[config].ntt;
}

/* Largest input of the schoolbook kernels within a warp: n limbs take
   inputs of up to 32 n - GPU_SPARE_BITS bits (see n_min). */
static size_t
warp_max_bits (void)
{
  return 32 * 32 * GPU_DIST_MAX_L - GPU_SPARE_BITS;
}

/* Largest input of the schoolbook kernels; above it only the NTT path. */
static size_t
school_max_bits (void)
{
  return 32 * GPU_WIDE_MAX_TPI * GPU_DIST_MAX_L - GPU_SPARE_BITS;
}

/* Free the NTT plans of numbers other than N: they hold tables and
   per-curve scratch on the GPU, and the numbers come one after another.
   (Configuration handles only live through one gpu_ecm call.) */
static void
drop_ntt_plans (const mpz_t N)
{
  for (size_t i = 0; i < plans.size (); )
    if (plans[i].ntt != NULL && mpz_cmp (plans[i].ntt_N, N) != 0)
      {
        gpu_ntt_destroy (plans[i].ntt);
        mpz_clear (plans[i].ntt_N);
        plans.erase (plans.begin () + i);
      }
    else
      i++;
}

static gpu_plan_t
ntt_plan (const mpz_t N, int lg, uint32_t n_min, uint32_t req_curves)
{
  gpu_plan_t p;
  p.n_min = n_min;
  p.req_curves = req_curves;
  p.tpi = 0;
  p.L = 0;
  p.fn = NULL;
  p.ntt = gpu_ntt_create (N, lg);
  mpz_init_set (p.ntt_N, N);
  p.curves = req_curves ? req_curves : gpu_ntt_fill_curves (p.ntt);
  return p;
}

extern "C"
size_t
gpu_ecm_stage1_max_bits (void)
{
  return std::max (school_max_bits (), gpu_ntt_max_bits ());
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
  drop_ntt_plans (N);

  /* GMPECM_GPU_KERNEL=TPI:L or TPI forces a configuration, =ntt the NTT */
  const char *env = getenv ("GMPECM_GPU_KERNEL");
  const bool force_ntt = env != NULL && strcmp (env, "ntt") == 0;
  uint32_t force_tpi = 0, force_L = 0;
  if (env != NULL && *env != '\0' && !force_ntt)
    {
      if (sscanf (env, "%u:%u", &force_tpi, &force_L) < 1)
        force_tpi = 0;
    }
  /* Up to warp_max_bits, the kernels within a warp (GPU stage 2 needs
     them).  Above, the wide kernels and the NTT are timed against each
     other, unless only the NTT can take the input. */
  const bool beyond_warp = bits > warp_max_bits ();
  bool use_ntt = force_ntt || bits > school_max_bits ()
                 || std::find (ntt_wins.begin (), ntt_wins.end (),
                               std::make_pair (n_min, *curves))
                    != ntt_wins.end ();

  for (size_t i = 0; i < plans.size (); i++)
    if (plans[i].req_curves == *curves
        && (use_ntt ? plans[i].ntt != NULL && mpz_cmp (plans[i].ntt_N, N) == 0
                    : plans[i].ntt == NULL && plans[i].n_min == n_min))
      {
        if (*curves == 0)
          *curves = plans[i].curves;
        return (int) i;
      }

  const int lg = gpu_ntt_lg_for_bits (bits);
  const bool ntt_fits = lg >= 0 && lg <= gpu_ntt_device_max_lg ();
  if (use_ntt)
    {
      if (!ntt_fits)
        {
          outputf (OUTPUT_ERROR, "GPU: no NTT size for a %zu-bit input on "
                   "this GPU%s\n", bits,
                   lg >= 0 ? " (too little shared memory)" : "");
          return ECM_ERROR;
        }
      gpu_plan_t p = ntt_plan (N, lg, n_min, *curves);
      if (verbose >= OUTPUT_VERBOSE)
        printf ("GPU: NTT multiplication, transform length 2^%d, %u-limb "
                "numbers, %u curves\n", lg, gpu_ntt_limbs (p.ntt), p.curves);
      plans.push_back (p);
      if (*curves == 0)
        *curves = p.curves;
      return (int) plans.size () - 1;
    }

  std::vector<gpu_plan_t> cand;
  const uint32_t max_tpi = beyond_warp || force_tpi > 32 ? GPU_WIDE_MAX_TPI
                                                        : 32;
  for (uint32_t tpi = 1; tpi <= max_tpi; tpi *= 2)
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
      const gpu_kernel_set *ks = gpu_kernels (tpi, L);
      gpu_s1_fn fn = ks != NULL ? ks->s1 : NULL;
      if (fn == NULL)
        continue;
      gpu_plan_t p;
      p.n_min = n_min;
      p.req_curves = *curves;
      p.tpi = tpi;
      p.L = L;
      p.fn = fn;
      p.ntt = NULL;
      p.curves = *curves ? *curves : fill_curves (fn, tpi);
      cand.push_back (p);
    }
  if (beyond_warp && !force_tpi && ntt_fits)
    cand.push_back (ntt_plan (N, lg, n_min, *curves));

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
            {
              if (cand[i].ntt != NULL)
                printf ("GPU: NTT, transform length 2^%d: %u curves, "
                        "%.3g us per curve-bit\n", lg, cand[i].curves, t * 1e6);
              else
                printf ("GPU: kernel TPI=%u L=%u (%u bits): %u curves, "
                        "%.3g us per curve-bit\n", cand[i].tpi, cand[i].L,
                        32 * cand[i].tpi * cand[i].L, cand[i].curves, t * 1e6);
            }
          if (i == 0 || t < best_t)
            {
              best = i;
              best_t = t;
            }
        }
    }
  for (size_t i = 0; i < cand.size (); i++)
    if (i != best && cand[i].ntt != NULL)
      {
        gpu_ntt_destroy (cand[i].ntt);
        mpz_clear (cand[i].ntt_N);
      }
  if (cand[best].ntt != NULL)
    {
      ntt_wins.push_back (std::make_pair (n_min, *curves));
      if (verbose >= OUTPUT_VERBOSE)
        printf ("GPU: NTT multiplication, transform length 2^%d, %u-limb "
                "numbers, %u curves\n", lg, gpu_ntt_limbs (cand[best].ntt),
                cand[best].curves);
    }

  plans.push_back (cand[best]);
  if (*curves == 0)
    *curves = cand[best].curves;
  return (int) plans.size () - 1;
}

/* ---------------------------------------------------------------------- */

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
                uint32_t curves, uint32_t sigma, const uint32_t *sigmas,
                int from_x, int config, float *gputime, int verbose)
{
  assert (sigma > 0);
  assert (sigmas != NULL || ((uint64_t) sigma + curves) <= 0xFFFFFFFF);
  assert (config >= 0 && (size_t) config < plans.size ());
  const gpu_plan_t &p = plans[config];

  if (mpz_even_p (N))
    {
      outputf (OUTPUT_ERROR, "GPU: Error, N must be odd\n");
      return ECM_ERROR;
    }

  const uint32_t n = plan_limbs (p);
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

#define CURVE_SIGMA(i) (sigmas != NULL ? sigmas[(i) < curves ? (i) : 0] \
                                       : sigma + (i))

  /* Initial points in Montgomery form (R = 2^(32n)), with
   * a24 = d = sigma_i / 2^32 mod N:
   *   P  = (2 : 1)
   *   2P = (9 : 64 d + 8)
   * or, for a resumed curve (from_x), P = (x0 : 1) with x0 = factors[i] and
   *   2P = ((x0+1)^2 (x0-1)^2 : 4 x0 ((x0-1)^2 + 4 d x0)),
   * the ladder's doubling.  Then the kernel also needs x0 itself, which is
   * the difference in every ladder step. */
  size_t data_words = (size_t) padded * 4 * n;
  uint32_t *data = (uint32_t *) malloc (data_words * sizeof (uint32_t));
  uint32_t *x0limbs = NULL;
  uint32_t *dlist = NULL;
  assert (data != NULL);
  if (from_x)
    {
      x0limbs = (uint32_t *) malloc ((size_t) padded * n * sizeof (uint32_t));
      assert (x0limbs != NULL);
    }
  if (sigmas != NULL)
    {
      dlist = (uint32_t *) malloc (padded * sizeof (uint32_t));
      assert (dlist != NULL);
      for (uint32_t i = 0; i < padded; i++)
        dlist[i] = CURVE_SIGMA (i);
    }
  {
    mpz_t R, x, inv32, d, a, b, c;
    mpz_init (R);
    mpz_init (x);
    mpz_init (inv32);
    mpz_init (d);
    mpz_init (a);
    mpz_init (b);
    mpz_init (c);
    mpz_setbit (R, 32 * (size_t) n);
    mpz_mod (R, R, N);
    mpz_ui_pow_ui (inv32, 2, 32);
    mpz_invert (inv32, inv32, N);

    for (uint32_t i = 0; i < padded; i++)
      {
        uint32_t *dp = data + (size_t) i * 4 * n;
        mpz_mul_ui (d, inv32, CURVE_SIGMA (i));
        mpz_mod (d, d, N);
        if (from_x)
          {
            mpz_mod (x, factors[i < curves ? i : 0], N);
            mpz_add_ui (a, x, 1);
            mpz_mul (a, a, a);                    /* A = (x0+1)^2 */
            mpz_sub_ui (b, x, 1);
            mpz_mul (b, b, b);                    /* B = (x0-1)^2 */
            mpz_sub (c, a, b);                    /* C = A - B */
            mpz_mul (a, a, b);
            mpz_mul (a, a, R);
            mpz_mod (a, a, N);
            gpu_to_limbs (dp + 2 * n, n, a);      /* bX = A B */
            mpz_mul (a, d, c);
            mpz_add (a, a, b);
            mpz_mul (a, a, c);
            mpz_mod (a, a, N);
            mpz_mul (a, a, R);
            mpz_mod (a, a, N);
            gpu_to_limbs (dp + 3 * n, n, a);      /* bZ = C (B + d C) */
            mpz_mul (x, x, R);
            mpz_mod (x, x, N);
            gpu_to_limbs (dp, n, x);              /* aX */
            gpu_to_limbs (x0limbs + (size_t) i * n, n, x);
          }
        else
          {
            mpz_mul_ui (x, R, 2);
            mpz_mod (x, x, N);
            gpu_to_limbs (dp, n, x);              /* aX */
            mpz_mul_ui (x, R, 9);
            mpz_mod (x, x, N);
            gpu_to_limbs (dp + 2 * n, n, x);      /* bX */
            mpz_mul_ui (x, d, 64);
            mpz_add_ui (x, x, 8);
            mpz_mod (x, x, N);
            if (i == 0)
              outputf (OUTPUT_TRACE, "sigma %u => P2_z: %Zd\n",
                       CURVE_SIGMA (0), x);
            mpz_mul (x, x, R);
            mpz_mod (x, x, N);
            gpu_to_limbs (dp + 3 * n, n, x);      /* bZ */
          }
        gpu_to_limbs (dp + n, n, R);              /* aZ */
      }
    mpz_clear (R);
    mpz_clear (x);
    mpz_clear (inv32);
    mpz_clear (d);
    mpz_clear (a);
    mpz_clear (b);
    mpz_clear (c);
  }

  uint32_t *Nlimbs = (uint32_t *) malloc (n * sizeof (uint32_t));
  gpu_to_limbs (Nlimbs, n, N);
  uint32_t np0 = gpu_find_np0 (N);

  cudaEvent_t global_start, batch_start, stop;
  CUDA_CHECK (cudaEventCreate (&global_start));
  CUDA_CHECK (cudaEventCreate (&batch_start));
  CUDA_CHECK (cudaEventCreate (&stop));
  CUDA_CHECK (cudaEventRecord (global_start));
  /* also STOP, for an exponent of one bit (a resumed curve whose B1done
     and B1 have no prime power between them): no launch records it then */
  CUDA_CHECK (cudaEventRecord (stop));

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
  uint32_t *gpu_dlist = NULL, *gpu_x0 = NULL;
  if (dlist != NULL)
    {
      CUDA_CHECK (cudaMalloc ((void **) &gpu_dlist, padded * sizeof (uint32_t)));
      CUDA_CHECK (cudaMemcpy (gpu_dlist, dlist, padded * sizeof (uint32_t),
                              cudaMemcpyHostToDevice));
    }
  if (x0limbs != NULL)
    {
      size_t bytes = (size_t) padded * n * sizeof (uint32_t);
      CUDA_CHECK (cudaMalloc ((void **) &gpu_x0, bytes));
      CUDA_CHECK (cudaMemcpy (gpu_x0, x0limbs, bytes, cudaMemcpyHostToDevice));
    }

  if (p.ntt != NULL)
    outputf (OUTPUT_VERBOSE, "GPU: %u-bit arithmetic (NTT), %u curves, "
             "input number is %zu bits\n", 32 * n, padded,
             mpz_sizeinbase (N, 2));
  else
    {
      kernel_info ((const void *) p.fn, verbose);
      outputf (OUTPUT_VERBOSE, "GPU: %u-bit arithmetic (TPI=%u, L=%u), "
               "%u blocks x %u threads, input number is %zu bits\n",
               32 * n, p.tpi, p.L, padded * p.tpi / GPU_TPB, GPU_TPB,
               mpz_sizeinbase (N, 2));
    }

  /* The first bit (doubling) is handled by the initial 2P. */
  uint64_t s_partial = 1;
  /* Start with small batches and increase till timing is ~100ms: at the
     largest sizes a bit takes milliseconds */
  uint64_t batch_size = 16, want = batch_size;
  int batches_complete = 0;
  float batch_time = 0, max_time = 0;

  while (s_partial < s_nbits)
    {
      batch_size = std::min (s_nbits - s_partial, want);

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
              gpu_data, gpu_N, np0, sigma, gpu_dlist, gpu_x0);

      s_partial += batch_size;
      batches_complete++;

      CUDA_CHECK (cudaEventRecord (stop));
      CUDA_CHECK (cudaEventSynchronize (stop));
      cudaEventElapsedTime (&batch_time, batch_start, stop);
      cudaEventElapsedTime (gputime, global_start, stop);
      max_time = std::max (max_time, batch_time);
      want = gpu_pace (want, batch_size, batch_time, batch_size < want);
    }

  outputf (OUTPUT_VERBOSE, "Copying results back to CPU (longest launch "
           "%.0f ms) ...\n", max_time);
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
        gpu_from_limbs (x, dp, n);
        gpu_from_limbs (z, dp + n, n);
        mpz_mul (x, x, Rinv);
        mpz_mod (x, x, N);
        mpz_mul (z, z, Rinv);
        mpz_mod (z, z, N);

        if (i == 0 && test_verbose (OUTPUT_TRACE))
          outputf (OUTPUT_TRACE, "index: 0 pA: (%Zd, %Zd)\n", x, z);

        /* (2 : 1) is the starting point: the kernel didn't run */
        if (!from_x && mpz_cmp_ui (x, 2) == 0 && mpz_cmp_ui (z, 1) == 0)
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
                     ECM_PARAM_BATCH_32BITS_D, CURVE_SIGMA (i));
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

  if (gpu_dlist != NULL)
    CUDA_CHECK (cudaFree (gpu_dlist));
  if (gpu_x0 != NULL)
    CUDA_CHECK (cudaFree (gpu_x0));
  free (s_bits);
  free (data);
  free (dlist);
  free (x0limbs);
  free (Nlimbs);

  return youpi;
}

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ < 350
    #error "Unsupported architecture"
  #endif
#endif
