/* gpu_s2poly.h: the polynomial (FFT) stage 2 on the GPU (see
   gpu_dev/poly_stage2_design.md).

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

#ifndef _GPU_S2POLY_H
#define _GPU_S2POLY_H 1

#include <stdint.h>
#include <stddef.h>

#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Whether the polynomial stage 2 can run for N with dF roots of F (dF a
   power of 2: GMP-ECM's parameters with use_ntt). */
int gpu_ecm_stage2_poly_supported (const mpz_t N, unsigned long dF);

/* Device memory it needs (an estimate, on the high side). */
double gpu_ecm_stage2_poly_mem (const mpz_t N, unsigned long dF);

/* The largest power of 2 dF' <= DF for which it can run in the free device
   memory, or 0. */
unsigned long gpu_ecm_stage2_poly_fit (const mpz_t N, unsigned long dF);

/* GMP-ECM's stage 2 (stage2.c, the same polynomials and values) for
   CURVES curves of N: curve i has the Montgomery A of As[i] and the stage 1
   residue x = xs[i]; GMP-ECM's root parameters S, d1, d2, i0, dF and k
   blocks.  For each curve where a factor is found, xs[i] is replaced by it
   and array_found[i] set.  Returns ECM_NO_FACTOR_FOUND,
   ECM_FACTOR_FOUND_STEP2 (some curve found one) or ECM_ERROR. */
int gpu_ecm_stage2_poly (mpz_t *xs, int *array_found, mpz_t *As,
                         const mpz_t N, int S, unsigned long d1,
                         unsigned long d2, const mpz_t i0, unsigned long dF,
                         unsigned long k, uint32_t curves, float *gputime,
                         int verbose);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_S2POLY_H */
