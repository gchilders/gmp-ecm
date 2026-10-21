/* gpu_ntt.h: host interface of the NTT-based GPU stage 1 (large inputs).

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

#ifndef _GPU_NTT_H
#define _GPU_NTT_H 1

#include <stdint.h>
#include <gmp.h>

/* Used only from C++ (gpu_stage1.cu). */

struct gpu_ntt_plan;

/* log2 of the transform length for an input of BITS bits (numbers of
   W = L/2 limbs with 16 N <= 2^(32 W)), or -1 if too large. */
int gpu_ntt_lg_for_bits (size_t bits);

/* Largest input the NTT path accepts. */
size_t gpu_ntt_max_bits (void);

/* Largest log2 transform length whose shared memory the current GPU has
   (below NTT_MIN_LG if none): lg = 14 needs about 76 KB per block. */
int gpu_ntt_device_max_lg (void);

/* Tables and the transforms of N and N' for transform length 2^lg. */
gpu_ntt_plan *gpu_ntt_create (const mpz_t N, int lg);
void gpu_ntt_destroy (gpu_ntt_plan *p);

/* Limbs per number (W). */
uint32_t gpu_ntt_limbs (const gpu_ntt_plan *p);

/* Curves that fill the GPU (one thread block per curve). */
uint32_t gpu_ntt_fill_curves (const gpu_ntt_plan *p);

/* Stage 1 ladder over bits [start, start + count) of s, MSB first, for
   CURVES curves; DATA holds aX, aZ, bX, bZ per curve (W limbs each,
   Montgomery form with R = 2^(32W)), as gpu_stage1.cu lays it out.
   DLIST / X0 as for the schoolbook kernel (may be NULL). */
void gpu_ntt_launch (gpu_ntt_plan *p, uint32_t curves,
                     const uint32_t *s_bits, uint64_t s_nbits,
                     uint64_t start, uint64_t count, uint32_t *data,
                     uint32_t sigma0, const uint32_t *dlist,
                     const uint32_t *x0);

/* Stage 2 (gpu_stage2.cu) on the NTT path, one thread block per curve.
   Per curve, BABY holds 2 NBABY numbers and STATE S_COUNT (gpu_common.cuh),
   W limbs each, contiguous; XIN one number per curve; ONE is R mod N.  The
   steps are those of the schoolbook kernels (gpu_stage2_kernel.cuh). */
void gpu_ntt_s2_setup (gpu_ntt_plan *p, uint32_t curves, const uint32_t *xin,
                       const uint32_t *one, uint32_t halfD,
                       const uint32_t *jlist, uint32_t nbaby, uint32_t *baby,
                       uint32_t *state, uint32_t step0, uint32_t step1,
                       uint32_t next0, uint32_t sigma0,
                       const uint32_t *dlist);
void gpu_ntt_s2_norm (gpu_ntt_plan *p, uint32_t curves, uint32_t *baby,
                      uint32_t nbaby, uint32_t *state, uint32_t i0,
                      uint32_t i1);
void gpu_ntt_s2_main (gpu_ntt_plan *p, uint32_t curves, const uint32_t *masks,
                      uint32_t count, uint32_t mw, const uint32_t *baby,
                      uint32_t nbaby, uint32_t *state);

#endif /* _GPU_NTT_H */
