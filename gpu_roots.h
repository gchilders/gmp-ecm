/* gpu_roots.h: the roots of F and G for the GPU polynomial stage 2 (see
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

/* Used only from C++/CUDA.

   The same numbers as GMP-ECM's ecm_rootsF / ecm_rootsG (ecm2.c), for a
   batch of curves with the same N.  With the stage 1 point P = (x : 1) on
   GMP-ECM's curve g Y^2 = X^3 + A X^2 + X, g = x^3 + A x^2 + x (as
   montgomery_to_weierstrass sets it up), and D_S the polynomial x^S (S > 0)
   or Dickson_(|S|, -1) (S < 0):

     roots of F, in order: x_W (D_S (d2 j) P) for j = 1 (mod 6),
       gcd (j, d1) = 1, j = 1, 7, 13, ..., the first dF of them;
     roots of G, block after block: x_W (D_S (d1 m) P) for m = i0, i0 + 1,
       ..., gcd (m, d2) = 1, dF per block;

   x_W the affine x of GMP-ECM's Weierstrass form, x_W (u, v) =
   (3u + A) / (3g).  They come out as numbers for gpu_polyalg (NL limbs,
   below 2N) in device memory, curve after curve: root i of curve c at
   (c dF + i) NL.

   The curve arithmetic uses the schoolbook products of gpu_mont.cuh, TPI
   threads per curve (2 to 32, or wide groups of 64 or 128) with L limbs
   each, up to gpu_roots_max_bits (). */

#ifndef _GPU_ROOTS_H
#define _GPU_ROOTS_H 1

#include <stdint.h>
#include <stddef.h>
#include <gmp.h>

struct gpu_roots;

/* The largest N (in bits) with kernels */
size_t gpu_roots_max_bits (void);

/* For N, the kernel shape (TPI, L; 0, 0 to choose one), GMP-ECM's root
   parameters (S, d1, d2 with gcd (d1, d2) = 1, i0), dF and the number of
   blocks of G (for planning only); NL limbs per output number
   (gpu_rns_limbs); CURVES curves.  NULL if the shape has no kernels. */
gpu_roots *gpu_roots_create (const mpz_t N, uint32_t tpi, uint32_t L, int S,
                             unsigned long d1, unsigned long d2,
                             const mpz_t i0, size_t dF, unsigned long blocks,
                             int nl, uint32_t curves, int verbose);
void gpu_roots_destroy (gpu_roots *G);

/* The CUDA stream (a cudaStream_t) for the roots (default 0); the calls
   below return when their work on it is done, so another stream can run
   meanwhile.  gpu_roots_tree and gpu_roots_mp stay on stream 0. */
void gpu_roots_stream (gpu_roots *G, void *stream);

/* Set up the curves: A[c] (Montgomery A, plain residue) and x[c] (the stage
   1 point), then the progressions of F and G.  Returns 0, or 1 with a
   factor of N in F and its curve in *CURVE (an inversion failed). */
int gpu_roots_curves (gpu_roots *G, mpz_t *A, mpz_t *x, mpz_t f, int *curve);

/* The dF roots of F, or of the next block of G, of every curve into OUT
   (curves * dF numbers).  Return 0, or 1 with a factor in F and its curve
   in *CURVE (a root was the point at infinity mod a prime of N). */
int gpu_roots_F (gpu_roots *G, uint32_t *out, mpz_t f, int *curve);
int gpu_roots_G (gpu_roots *G, uint32_t *out, mpz_t f, int *curve);

/* A level of product trees in Z/N for gpu_polyalg (gpu_polyalg_lowmul;
   CTX is a gpu_roots of the same N and NL): IN holds COUNT / (2D) pairs of
   monic nodes of degree D (their D low coefficients side by side), OUT gets
   their products, monic of degree 2D. */
void gpu_roots_tree (void *ctx, uint32_t *out, const uint32_t *in,
                     size_t count, size_t d);

/* A level of the transposed evaluation in Z/N for gpu_polyalg
   (gpu_polyalg_lowmp): the COUNT values TV of nodes of 2 HALF values, with
   the children's coefficients in TREE, into OUT (see gpu_poly_tpeval). */
void gpu_roots_mp (void *ctx, uint32_t *out, const uint32_t *tv,
                   const uint32_t *tree, size_t count, size_t half);

#endif /* _GPU_ROOTS_H */
