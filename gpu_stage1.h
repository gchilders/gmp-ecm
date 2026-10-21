/* gpu_stage1.h: header for GPU based ecm stage 1.

  Copyright 2021 Seth Troisi, 2026 Greg Childers

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

#ifndef _GPU_STAGE1_H
#define _GPU_STAGE1_H 1

#include <stdint.h>

#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Largest input (in bits) the GPU stage 1 kernels accept. */
size_t gpu_ecm_stage1_max_bits (void);

/* Choose a kernel configuration for N.  If *curves is 0 it is set to the
   number of curves that fills the GPU for that configuration.  Returns a
   configuration handle (>= 0) for gpu_ecm_stage1, or ECM_ERROR. */
int gpu_ecm_stage1_plan (const mpz_t N, unsigned int *curves, int verbose);

/* (TPI, L) of a configuration from gpu_ecm_stage1_plan; TPI = 0 for the
   NTT path */
void gpu_ecm_plan_shape (int config, uint32_t *tpi, uint32_t *L);

/* The NTT tables of a configuration on the NTT path, else NULL */
struct gpu_ntt_plan *gpu_ecm_plan_ntt (int config);

/* Stage 1 for CURVES curves: multiply each curve's point by S.  Curve i has
   sigma + i, or sigmas[i] when SIGMAS is not NULL.  The points start at
   x = 2 (a fresh run), or at x = factors[i] when FROM_X (resuming a saved
   residue: S is then the exponent from the saved B1done to the new B1).
   On return factors[i] is the residue x, or the factor found. */
int gpu_ecm_stage1 (mpz_t *factors, int *array_found,
                    const mpz_t N, const mpz_t s,
                    uint32_t curves, uint32_t sigma, const uint32_t *sigmas,
                    int from_x, int config, float *gputime, int verbose);

#ifdef __cplusplus
}
#endif


#endif /* _GPU_STAGE1_H */
