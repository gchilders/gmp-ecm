/* gpu_stage2.h: header for GPU based ecm stage 2.

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

#ifndef _GPU_STAGE2_H
#define _GPU_STAGE2_H 1

#include <stdint.h>

#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default B2 for GPU stage 2, as a multiple of B1 (its cost is linear in
   B2, so the CPU default, tuned for the FFT continuation, is far too big). */
#define GPU_STAGE2_DEFAULT_B2_FACTOR 100

/* Stage 2 for CURVES curves (sigma, sigma+1, ..., or sigmas[i] when SIGMAS
   is not NULL, as for a resumed batch) whose stage 1 residues
   x = X/Z are in xs[].  For each curve where a factor is found, xs[i] is
   replaced by it and array_found[i] set.  CONFIG is from
   gpu_ecm_stage1_plan. */
/* Whether stage 2 can run on the GPU for CONFIG (from
   gpu_ecm_stage1_plan) */
int gpu_ecm_stage2_supported (int config);

int gpu_ecm_stage2 (mpz_t *xs, int *array_found, const mpz_t N,
                    const mpz_t B2min, const mpz_t B2,
                    uint32_t curves, uint32_t sigma,
                    const uint32_t *sigmas, int config,
                    float *gputime, int verbose);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_STAGE2_H */
