/* gpu_rns.h: polynomials over Z/N in residue number system form on the GPU,
   for the polynomial stage 2 (see gpu_dev/poly_stage2_design.md).

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

/* Used only from C++/CUDA.

   A number mod N is NL 32-bit limbs, NL with 4 spare bits above N: any
   value below 2^(32 NL) is accepted, results are below 2N, so sums of a
   few results can go back in.  A polynomial of C coefficients in Z/N form is C numbers,
   coefficient-major (limb j of coefficient c at c * NL + j).

   In RNS form a polynomial is P rows, one per small prime p_i (row i,
   ROWLEN words, entry c the residue of coefficient c), and a batch of B
   polynomials is B P rows, polynomial-major (row b P + i).  The primes are
   p = c 2^K + 1 < 2^30, with 2^K the longest transform, and enough of them
   that a cyclic convolution of length up to 2^K of numbers below 2^(32 NL)
   is exact (as GMP-ECM's mpzspm_init, prod p > 4 2^K (2^(32 NL) sum p)^2).

   Transforms: forward leaves the values in a permuted order (the same for
   every polynomial of a length), in [0, 2p); a pointwise product is a
   Montgomery product (it divides by 2^32); the inverse takes the number of
   such products since the forward transforms (0 or 1) and returns the
   cyclic convolution in natural order, canonical. */

#ifndef _GPU_RNS_H
#define _GPU_RNS_H 1

#include <stdint.h>
#include <stddef.h>
#include <gmp.h>

struct gpu_rns;

/* Context for N with transforms up to 2^LG_WANT (fewer if there are not
   enough primes: see gpu_rns_lg). */
gpu_rns *gpu_rns_create (const mpz_t N, int lg_want);
void gpu_rns_destroy (gpu_rns *R);

int gpu_rns_primes (const gpu_rns *R);   /* P */
int gpu_rns_lg (const gpu_rns *R);       /* K: longest transform 2^K */
int gpu_rns_limbs (const gpu_rns *R);    /* NL */
const uint32_t *gpu_rns_dev_N (const gpu_rns *R);   /* N, NL limbs (device) */

/* RNS rows RES (B P rows of ROWLEN words) <- the C coefficients of each of
   the B polynomials in A (polynomial b at A + b SA numbers); entries
   C .. ROWLEN-1 are zeroed. */
void gpu_rns_from_z (gpu_rns *R, uint32_t *res, size_t rowlen,
                     const uint32_t *a, size_t sa, size_t C, int B);

/* A (polynomial b at A + b SC numbers, each number below 2N) <- the first
   C entries of each of the B P rows of RES (canonical residues). */
void gpu_rns_to_z (gpu_rns *R, uint32_t *a, size_t sc, const uint32_t *res,
                   size_t rowlen, size_t C, int B);

/* In-place transforms of length 2^LG <= 2^K of the first 2^LG entries of
   ROWS rows (ROWS a multiple of P).  INV: inverse, with NMUL Montgomery
   pointwise products to undo (0 or 1). */
void gpu_rns_ntt (gpu_rns *R, uint32_t *res, size_t rowlen, int rows, int lg,
                  int inv, int nmul);

/* R = A * B pointwise (Montgomery) on the first LEN entries of ROWS rows;
   with BBC, B is one polynomial (P rows) for all.  R may be A or B. */
void gpu_rns_mul (gpu_rns *R, uint32_t *r, const uint32_t *a,
                  const uint32_t *b, size_t rowlen, int rows, size_t len,
                  int bbc);

/* OUT (P rows of 2^LG words) <- the forward transform of length 2^LG of B
   (NB coefficients), for gpu_rns_conv's BPREP. */
void gpu_rns_prep (gpu_rns *R, uint32_t *out, const uint32_t *b, size_t nb,
                   int lg);

/* For each of BATCH pairs: entries O0 .. O0+OC-1 of the cyclic convolution
   of length 2^LG of A (NA coefficients, pair j at A + j SA) and B (NB
   coefficients at B + j SB; SB = 0: the same B for all; or, if BPREP is not
   NULL, B's transform from gpu_rns_prep, for all), into C + j SC (numbers
   below 2N).  NA, NB, O0 + OC <= 2^LG <= 2^K; the inputs below 2^(32 NL). */
void gpu_rns_conv (gpu_rns *R, uint32_t *c, size_t sc, size_t o0, size_t oc,
                   const uint32_t *a, size_t sa, size_t na, const uint32_t *b,
                   size_t sb, size_t nb, const uint32_t *bprep, int lg,
                   int batch);

/* Product of polynomials over Z/N: C = A B, A of NA and B of NB
   coefficients (Z/N form, device memory), C of NA + NB - 1, for BATCH
   products side by side.  NA + NB - 1 <= 2^K. */
void gpu_rns_polymul (gpu_rns *R, uint32_t *c, const uint32_t *a, size_t na,
                      const uint32_t *b, size_t nb, int batch);

#endif /* _GPU_RNS_H */
