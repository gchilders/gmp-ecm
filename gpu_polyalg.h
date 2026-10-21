/* gpu_polyalg.h: polynomial algorithms of the stage 2 FFT continuation on
   the GPU (product trees, power series inverse, reduction, transposed
   multipoint evaluation), on top of gpu_rns.  See
   gpu_dev/poly_stage2_design.md.

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

   Conventions are GMP-ECM's (stage2.c): n = 2^l roots r_i give the monic
   F = prod (x + r_i), stored as its n low coefficients (the leading 1 is
   implicit); its product tree has levels tree[t], t = 0 .. l-1, of n
   numbers each, level t holding the 2^(t+1) nodes of degree n / 2^(t+1)
   (node j's low coefficients at j n / 2^(t+1)), so tree[l-1] holds the
   roots.  Numbers are as in gpu_rns.h (NL limbs, results below 2N), all in
   device memory. */

#ifndef _GPU_POLYALG_H
#define _GPU_POLYALG_H 1

#include <stdint.h>
#include <stddef.h>
#include "gpu_rns.h"

struct gpu_polyalg;

gpu_polyalg *gpu_polyalg_create (gpu_rns *R, size_t n);
void gpu_polyalg_destroy (gpu_polyalg *A);

/* F (n numbers) <- the low coefficients of prod (x + roots[i]), i < n;
   with TREE (l pointers to n numbers each), also the product tree. */
void gpu_poly_from_roots (gpu_polyalg *A, uint32_t *F, uint32_t **tree,
                          const uint32_t *roots);

/* I (n numbers) <- 1 / rev(F) mod x^n, rev(F) = 1 + F[n-1] x + ... */
void gpu_poly_inverse (gpu_polyalg *A, uint32_t *I, const uint32_t *F);

/* Prepare F and I for gpu_poly_reduce and gpu_poly_tpeval (their
   transforms); after gpu_poly_inverse. */
void gpu_poly_prepare (gpu_polyalg *A, const uint32_t *F, const uint32_t *I);

/* Rm (n numbers) <- A2 mod F, A2 of 2n - 1 numbers (Rm may be A2). */
void gpu_poly_reduce (gpu_polyalg *A, uint32_t *Rm, const uint32_t *A2);

/* H (n numbers) <- (G - F) H mod F, G monic of degree n (n low
   coefficients); FIRST: H <- G - F. */
void gpu_poly_mulmod (gpu_polyalg *A, uint32_t *H, const uint32_t *G,
                      int first);

/* VALS (n numbers) <- H(-r_i) for the roots of F's tree, H of degree < n */
void gpu_poly_tpeval (gpu_polyalg *A, uint32_t *vals, const uint32_t *H,
                      uint32_t *const *tree);

/* Device numbers: reduce COUNT numbers below N (in place). */
void gpu_zn_reduce (gpu_polyalg *A, uint32_t *x, size_t count);

#endif /* _GPU_POLYALG_H */
