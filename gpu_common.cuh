/* gpu_common.cuh: definitions shared by the GPU stage 1 and stage 2 code.

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

#ifndef _GPU_COMMON_CUH
#define _GPU_COMMON_CUH 1

#include <assert.h>
#include <stdint.h>
#include <gmp.h>

#define GPU_TPB 128
#define GPU_SOLO_MAX_L 28
/* above this, solo kernels only for even L (the even/odd product of
   gpu_mont.cuh); an odd number of limbs takes the next even L */
#define GPU_SOLO_ODD_MAX_L 16
#define GPU_DIST_MAX_L 24
#define GPU_WIDE_MAX_TPI 128   /* wide groups: 64 or 128 threads per curve */
/* N must be below 2^(32n - GPU_SPARE_BITS) */
#define GPU_SPARE_BITS 4

/* Stage 2 state slots per curve (gpu_stage2.cu).  AUX holds 2Q during
   setup and the running product of the Z's while the baby steps are put on
   a shared denominator; BZ holds that denominator. */
enum { S_PREV_X, S_PREV_Z, S_CUR_X, S_CUR_Z, S_DQ_X, S_DQ_Z, S_ACC, S_BZ,
       S_AUX_X, S_AUX_Z, S_COUNT };

/* Kernel table indexed by L (index 0 unused), filled by template recursion:
   KT<L, SOLO>::fill (t) stores F<L, SOLO> for every L down to 1. */
#define GPU_KERNEL_TABLE(name, fn_type, kernel)                         \
  template<int L, bool SOLO>                                            \
  struct name {                                                         \
    static void fill (fn_type *t)                                       \
    {                                                                   \
      t[L] = kernel<L, SOLO>;                                           \
      name<L - 1, SOLO>::fill (t);                                      \
    }                                                                   \
  };                                                                    \
  template<bool SOLO>                                                   \
  struct name<0, SOLO> {                                                \
    static void fill (fn_type *t) { t[0] = NULL; }                      \
  };

static inline void
gpu_to_limbs (uint32_t *x, uint32_t n, const mpz_t s)
{
  size_t words = 0;
  assert (mpz_sgn (s) >= 0 && mpz_sizeinbase (s, 2) <= 32 * (size_t) n);
  mpz_export (x, &words, -1, sizeof (uint32_t), 0, 0, s);
  while (words < n)
    x[words++] = 0;
}

static inline void
gpu_from_limbs (mpz_t r, const uint32_t *x, uint32_t n)
{
  mpz_import (r, n, -1, sizeof (uint32_t), 0, 0, x);
}

/* Work for the next launch of a kernel whose launches are sliced: USED
   units of work took MS milliseconds; BUDGET was the allowance, and CUT
   says that the launch got less than that because the work ran out.  Aim
   at ~100 ms per launch, so that a display GPU stays responsive and no
   launch nears the 2 s watchdog of Windows; change by at most a factor of
   2 per launch. */
static inline uint64_t
gpu_pace (uint64_t budget, uint64_t used, float ms, bool cut)
{
  if (ms >= 80 && ms <= 120)
    return budget;
  if (cut && ms < 80)
    return budget;
  double f = ms > 0 ? 100.0 / ms : 2.0;
  f = f > 2.0 ? 2.0 : (f < 0.5 ? 0.5 : f);
  uint64_t b = (uint64_t) (used * f);
  return b < 1 ? 1 : b;
}

/* -N^-1 mod 2^32 */
static inline uint32_t
gpu_find_np0 (const mpz_t N)
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

#endif /* _GPU_COMMON_CUH */
