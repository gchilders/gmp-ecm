/* gpu_roots.cu: the roots of F and G for the GPU polynomial stage 2 (see
   gpu_roots.h and gpu_dev/poly_stage2_design.md).

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

/* The curve is used in its twisted Edwards form a X^2 + Y^2 = 1 + d X^2 Y^2,
 * a = (A+2)/g, d = (A-2)/g, where P = (x, (x-1)/(x+1)), in extended
 * coordinates (X : Y : Z : T), T = XY/Z, with the unified addition (11
 * products; all values below 2N, as in gpu_stage1.cu).  A point's Montgomery
 * u is (Z + Y)/(Z - Y), so GMP-ECM's x_W = (3u + A)/(3g) is
 *   ((3 + A) Z + (3 - A) Y) / (3g (Z - Y)).
 * The x of a multiple of P does not depend on which square root of g the
 * Montgomery-to-Edwards map takes (v = 1 here), so these are GMP-ECM's
 * numbers.
 *
 * Roots D_S (v0 + n dv) P along a progression of arguments come from a
 * table of finite differences fd[t] = (Delta^t f)(0) P, t = 0..S, for
 * f (n) = D_S (v0 + n dv): a step replaces fd[t] by fd[t] + fd[t+1] for
 * t = 0..S-1 in that order, and fd[0] runs through f (n) P.  The
 * progressions are our own (the root sets are what matter; the order is
 * GMP-ECM's anyway, through the output slots):
 *   F: j = s + n st with st = 6 m k, m a product of primes >= 5 of d1, s
 *      running over the residues = 1 (mod 6) prime to m: phi (m) k
 *      progressions.  The j not prime to d1, or past the dF-th root, are
 *      skipped: a table gives each step's slot among the roots, or -1.
 *   G: m = i0 + o + n st with st = d2 k, o in [0, st) with
 *      gcd (i0 + o, d2) = 1; with o the r-th of them, m is root number
 *      r + n nr (nr progressions), so a block of dF roots is a run of steps
 *      of every progression and the state carries over to the next block.
 * Each (curve, progression) is an instance of TPI threads.  A launch does a
 * few steps of all of them, storing numerator and denominator of x_W of
 * each wanted root into its slot; then one batched inversion over all
 * curves (Montgomery's trick as a product tree, one inversion on the host)
 * gives the roots.  A denominator that is 0 mod a prime q of N means a
 * multiple of P is the identity mod q: a factor, as GMP-ECM would find it.
 *
 * The tables start from fixed-base windowed scalar multiplications: per
 * curve W[k][j] = j 2^(w k) P for 0 < j < 2^w, so that c P is a sum of one
 * entry per w-bit digit of c.  The coefficients are the same for all
 * curves; the largest, Delta^S f = S! dv^S, is the same for all
 * progressions of a set.
 *
 * The lanes of a group exchange limbs by warp shuffles, so every instance of
 * a warp runs the same products: loops have the same trip count for all
 * instances, and only loads and stores depend on the instance (padding
 * instances repeat the last one and store nothing). */

#include "gpu_roots.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include "cudacommon.h"
#include "gpu_common.cuh"
#include "gpu_mont.cuh"

/* kernels for L up to this (dev builds instantiate fewer) */
#ifndef GPU_ROOTS_MAX_L
#define GPU_ROOTS_MAX_L 24
#endif

/* steps or digits per launch */
#define RT_SLICE 32

/* per-curve constants (Montgomery form, contiguous numbers of n limbs) */
enum { RC_A, RC_D, RC_C1, RC_C2, RC_C3, RC_PX, RC_PY, RC_PZ, RC_PT,
       RC_COUNT };

/* modes of rt_vec_kernel */
enum { RV_UP, RV_DOWN, RV_FINAL };

template<int L>
struct rt_pt {
  uint32_t X[L], Y[L], Z[L], T[L];
};

/* contiguous numbers: number I at I n, this lane's limbs at OFF */
template<int L>
GM_DEV void
rt_cload (uint32_t (&x)[L], const uint32_t * __restrict__ buf, size_t i,
          uint32_t n, uint32_t off)
{
  const uint32_t *p = buf + i * n + off;
#pragma unroll
  for (int k = 0; k < L; k++)
    x[k] = p[k];
}

template<int L>
GM_DEV void
rt_cstore (uint32_t * __restrict__ buf, const uint32_t (&x)[L], size_t i,
           uint32_t n, uint32_t off)
{
  uint32_t *p = buf + i * n + off;
#pragma unroll
  for (int k = 0; k < L; k++)
    p[k] = x[k];
}

/* a point is four numbers I .. I+3 */
template<int L>
GM_DEV void
rt_cload_pt (rt_pt<L> &p, const uint32_t * __restrict__ buf, size_t i,
             uint32_t n, uint32_t off)
{
  rt_cload<L> (p.X, buf, i, n, off);
  rt_cload<L> (p.Y, buf, i + 1, n, off);
  rt_cload<L> (p.Z, buf, i + 2, n, off);
  rt_cload<L> (p.T, buf, i + 3, n, off);
}

template<int L>
GM_DEV void
rt_cstore_pt (uint32_t * __restrict__ buf, const rt_pt<L> &p, size_t i,
              uint32_t n, uint32_t off)
{
  rt_cstore<L> (buf, p.X, i, n, off);
  rt_cstore<L> (buf, p.Y, i + 1, n, off);
  rt_cstore<L> (buf, p.Z, i + 2, n, off);
  rt_cstore<L> (buf, p.T, i + 3, n, off);
}

/* interleaved numbers (the progression tables): limb k of number S of
   thread column C (of T) at (S L + k) T + C, so that warps load whole
   lines */
template<int L>
GM_DEV void
rt_iload (uint32_t (&x)[L], const uint32_t * __restrict__ buf, size_t s,
          uint32_t T, uint32_t c)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    x[k] = buf[(s * L + k) * T + c];
}

template<int L>
GM_DEV void
rt_istore (uint32_t * __restrict__ buf, const uint32_t (&x)[L], size_t s,
           uint32_t T, uint32_t c)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    buf[(s * L + k) * T + c] = x[k];
}

template<int L>
GM_DEV void
rt_iload_pt (rt_pt<L> &p, const uint32_t * __restrict__ buf, size_t s,
             uint32_t T, uint32_t c)
{
  rt_iload<L> (p.X, buf, s, T, c);
  rt_iload<L> (p.Y, buf, s + 1, T, c);
  rt_iload<L> (p.Z, buf, s + 2, T, c);
  rt_iload<L> (p.T, buf, s + 3, T, c);
}

template<int L>
GM_DEV void
rt_istore_pt (uint32_t * __restrict__ buf, const rt_pt<L> &p, size_t s,
              uint32_t T, uint32_t c)
{
  rt_istore<L> (buf, p.X, s, T, c);
  rt_istore<L> (buf, p.Y, s + 1, T, c);
  rt_istore<L> (buf, p.Z, s + 2, T, c);
  rt_istore<L> (buf, p.T, s + 3, T, c);
}

template<int L>
GM_DEV void
rt_copy_pt (rt_pt<L> &r, const rt_pt<L> &a)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    {
      r.X[k] = a.X[k];
      r.Y[k] = a.Y[k];
      r.Z[k] = a.Z[k];
      r.T[k] = a.T[k];
    }
}

/* the identity (0 : 1 : 1 : 0); ONEG is R mod N */
template<int L>
GM_DEV void
rt_identity (rt_pt<L> &p, const uint32_t * __restrict__ oneg, uint32_t off)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    {
      p.X[k] = 0;
      p.T[k] = 0;
      p.Y[k] = oneg[off + k];
      p.Z[k] = oneg[off + k];
    }
}

template<int L>
GM_DEV void
rt_modulus (const gm_group &g, uint32_t (&N)[L], uint32_t (&N2)[L],
            const uint32_t * __restrict__ Ng, uint32_t off)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    N[k] = Ng[off + k];
  gm_dbl<L, false> (g, N2, N);
}

/* r = p + q (unified: also for p = q, or either the identity), with the
   curve constants a, d of CST.  Inputs below 2N, outputs below 2N.  R may
   alias P and Q. */
template<int L>
GM_DEV void
rt_add (const gm_group &g, rt_pt<L> &r, const rt_pt<L> &p,
        const rt_pt<L> &q, const uint32_t * __restrict__ cst, uint32_t n,
        uint32_t off, const uint32_t (&N)[L], const uint32_t (&N2)[L],
        uint32_t np0)
{
  uint32_t A[L], B[L], C[L], D[L], E[L], t[L];
  gm_mul<L, false> (g, A, p.X, q.X, N, np0);
  gm_mul<L, false> (g, B, p.Y, q.Y, N, np0);
  gm_mul<L, false> (g, C, p.T, q.T, N, np0);
  rt_cload<L> (t, cst, RC_D, n, off);
  gm_mul<L, false> (g, C, C, t, N, np0);          /* C = d T1 T2 */
  gm_mul<L, false> (g, D, p.Z, q.Z, N, np0);
  gm_add<L, false> (g, E, p.X, p.Y);              /* < 4N */
  gm_add<L, false> (g, t, q.X, q.Y);
  gm_mul<L, false> (g, E, E, t, N, np0);
  gm_sub<L, false> (g, E, E, A, N2);
  gm_sub<L, false> (g, E, E, B, N2);              /* (X1+Y1)(X2+Y2)-A-B */
  rt_cload<L> (t, cst, RC_A, n, off);
  gm_mul<L, false> (g, A, A, t, N, np0);
  gm_sub<L, false> (g, t, B, A, N2);              /* H = B - a A */
  gm_add<L, false> (g, B, D, C);                  /* G = D + C < 4N */
  gm_sub<L, false> (g, D, D, C, N2);              /* F = D - C */
  gm_mul<L, false> (g, r.X, E, D, N, np0);
  gm_mul<L, false> (g, r.Y, B, t, N, np0);
  gm_mul<L, false> (g, r.T, E, t, N, np0);
  gm_mul<L, false> (g, r.Z, D, B, N, np0);
}

/* modes of rt_setup_kernel */
enum { RS_CHAIN, RS_FILL, RS_SCALAR };

/* The fixed-base tables and the starting tables of the progressions, in one
   kernel with a single addition (each inlined addition costs much compile
   time).  The point j 2^(w k) P of curve c, W[c][k][j], is at numbers
   ((c K + k) NW + j - 1) 4 of W.
   RS_CHAIN: one instance per curve: W[c][k][1] for k in [K0, K1), by
     doublings.
   RS_FILL: one instance per (c, k), i = c K + k: W[c][k][j] =
     W[c][k][j-1] + W[c][k][1] for 1 < j <= NW.
   RS_SCALAR: one instance per (c, e), i = c E + e: digits [K0, K1) of
     |coefficient e| (base 2^w, DIG[e K + k]), added into its table point,
     negated after the last digit if SGN[e].
     Coefficient e < nr S is Delta^(e % S) of progression e / S, e = nr S
     the shared Delta^S (copied to every progression after the last
     digit).  Table point t of progression r of curve c is the interleaved
     point t (numbers 4t .. 4t+3) of thread column (c nr + r) TPI + lane, of
     TS columns. */
template<int L>
__global__ void
rt_setup_kernel (uint32_t tpi, int mode, uint32_t count,
                 const uint32_t * __restrict__ cst, uint32_t * __restrict__ W,
                 uint32_t K, uint32_t w, uint32_t nw, uint32_t k0,
                 uint32_t k1, uint32_t E, uint32_t nr, uint32_t S,
                 const uint8_t * __restrict__ dig,
                 const uint8_t * __restrict__ sgn,
                 uint32_t * __restrict__ state, uint32_t Ts,
                 const uint32_t * __restrict__ Ng,
                 const uint32_t * __restrict__ oneg, uint32_t np0)
{
  gm_group g;
  g.init (tpi);
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / g.tpi;
  const uint32_t n = g.tpi * L, off = g.lane * L;
  const bool live = inst < count;
  const uint32_t i = live ? inst : count - 1;
  const uint32_t c = mode == RS_CHAIN ? i : mode == RS_FILL ? i / K : i / E;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);
  const uint32_t *cc = cst + (size_t) c * RC_COUNT * n;
  const size_t wc = (size_t) c * K * nw * 4;      /* this curve's W */

  rt_pt<L> acc, q;
  uint32_t iters, kstart = 0, e = 0, pt = 0, col = 0;
  bool shared = false;
  if (mode == RS_CHAIN)
    {
      if (k0 == 0)
        {
          rt_cload_pt<L> (acc, cc, RC_PX, n, off);
          if (live)
            rt_cstore_pt<L> (W, acc, wc, n, off);
          kstart = 1;
        }
      else
        {
          rt_cload_pt<L> (acc, W, wc + (size_t) (k0 - 1) * nw * 4, n, off);
          kstart = k0;
        }
      iters = k1 > kstart ? (k1 - kstart) * w : 0;
    }
  else if (mode == RS_FILL)
    {
      rt_cload_pt<L> (acc, W, (size_t) i * nw * 4, n, off);
      iters = nw - 1;
    }
  else
    {
      e = i % E;
      shared = e == nr * S;
      const uint32_t r = shared ? 0 : e / S;
      pt = shared ? S : e % S;
      col = (c * nr + r) * g.tpi + g.lane;
      if (k0 == 0)
        rt_identity<L> (acc, oneg, off);
      else
        rt_iload_pt<L> (acc, state, (size_t) pt * 4, Ts, col);
      iters = k1 - k0;
    }

  for (uint32_t it = 0; it < iters; it++)
    {
      if (mode == RS_CHAIN)
        rt_copy_pt<L> (q, acc);
      else if (mode == RS_FILL)
        rt_cload_pt<L> (q, W, (size_t) i * nw * 4, n, off);
      else
        {
          uint32_t k = k0 + it, d = dig[(size_t) e * K + k];
          if (d != 0)
            rt_cload_pt<L> (q, W, wc + ((size_t) k * nw + d - 1) * 4, n, off);
          else
            rt_identity<L> (q, oneg, off);
        }
      rt_add<L> (g, acc, acc, q, cc, n, off, N, N2, np0);
      if (live && mode == RS_CHAIN && (it + 1) % w == 0)
        rt_cstore_pt<L> (W, acc, wc + (size_t) (kstart + it / w) * nw * 4, n,
                         off);
      if (live && mode == RS_FILL)
        rt_cstore_pt<L> (W, acc, ((size_t) i * nw + it + 1) * 4, n, off);
    }

  if (mode == RS_SCALAR && k1 == K)
    {
      /* -(X : Y : Z : T) = (-X : Y : Z : -T) */
      uint32_t zero[L];
#pragma unroll
      for (int k = 0; k < L; k++)
        zero[k] = 0;
      gm_sub<L, false> (g, q.X, zero, acc.X, N2);
      gm_sub<L, false> (g, q.T, zero, acc.T, N2);
      if (sgn[e])
#pragma unroll
        for (int k = 0; k < L; k++)
          {
            acc.X[k] = q.X[k];
            acc.T[k] = q.T[k];
          }
    }
  if (live && mode == RS_SCALAR)
    {
      rt_istore_pt<L> (state, acc, (size_t) pt * 4, Ts, col);
      if (shared && k1 == K)
        for (uint32_t rr = 1; rr < nr; rr++)
          rt_istore_pt<L> (state, acc, (size_t) S * 4, Ts,
                           (c * nr + rr) * g.tpi + g.lane);
    }
}

/* Steps IT in [IT0, IT1) of all progressions (one instance per (curve c,
   progression r), C nr of them; the state is interleaved over this
   launch's threads).  Step IT emits the root of fd[0] (unless its slot is
   outside the roots wanted now), then moves the table on.
   F (SLOTS != NULL): step n = IT, slot SLOTS[n nr + r] (-1: none).
   G: the roots [BASE, BASE + dF); progression r has root numbers r + n nr,
   so its steps n in [nlo, nhi) are the ones in range, n = nlo + IT; the
   state of an instance with no more steps is not updated. */
template<int L>
__global__ void
rt_step_kernel (uint32_t tpi, uint32_t ninst, uint32_t nr, uint32_t S,
                uint32_t * __restrict__ state,
                const uint32_t * __restrict__ cst,
                const int32_t * __restrict__ slots, uint64_t base,
                uint32_t dF, uint32_t it0, uint32_t it1,
                uint32_t * __restrict__ num, uint32_t * __restrict__ den,
                const uint32_t * __restrict__ Ng, uint32_t np0)
{
  gm_group g;
  g.init (tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t T = gridDim.x * blockDim.x;
  const uint32_t inst = t / g.tpi;
  const uint32_t n = g.tpi * L, off = g.lane * L;
  const bool live = inst < ninst;
  const uint32_t i = live ? inst : ninst - 1;
  const uint32_t c = i / nr, r = i % nr;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);
  const uint32_t *cc = cst + (size_t) c * RC_COUNT * n;

  int64_t nlo = 0, nhi = 0;
  if (slots == NULL)
    {
      int64_t lo = (int64_t) base - r, hi = (int64_t) base + dF - r;
      nlo = lo > 0 ? (lo + nr - 1) / nr : 0;
      nhi = hi > 0 ? (hi + nr - 1) / nr : 0;
    }

  rt_pt<L> cur, nxt;
  for (uint32_t it = it0; it < it1; it++)
    {
      bool act;
      int64_t slot;
      if (slots != NULL)
        {
          act = live;
          slot = slots[(size_t) it * nr + r];
        }
      else
        {
          int64_t nn = nlo + it;
          act = live && nn < nhi;
          slot = (int64_t) r + nn * nr - (int64_t) base;
        }
      const bool emit = act && slot >= 0 && slot < dF;
      const size_t o = (size_t) c * dF + (emit ? slot : 0);

      rt_iload_pt<L> (cur, state, 0, T, t);
      {
        /* numerator (3 + A) Z + (3 - A) Y < 4N, denominator 3g (Z - Y) */
        uint32_t u[L], v[L];
        rt_cload<L> (v, cc, RC_C1, n, off);
        gm_mul<L, false> (g, u, cur.Z, v, N, np0);
        rt_cload<L> (v, cc, RC_C2, n, off);
        gm_mul<L, false> (g, v, cur.Y, v, N, np0);
        gm_add<L, false> (g, u, u, v);
        if (emit)
          rt_cstore<L> (num, u, o, n, off);
        gm_sub<L, false> (g, u, cur.Z, cur.Y, N2);
        rt_cload<L> (v, cc, RC_C3, n, off);
        gm_mul<L, false> (g, u, u, v, N, np0);
        if (emit)
          rt_cstore<L> (den, u, o, n, off);
      }
      for (uint32_t s = 0; s < S; s++)
        {
          rt_iload_pt<L> (nxt, state, (size_t) (s + 1) * 4, T, t);
          rt_add<L> (g, cur, cur, nxt, cc, n, off, N, N2, np0);
          if (act)
            rt_istore_pt<L> (state, cur, (size_t) s * 4, T, t);
          rt_copy_pt<L> (cur, nxt);
        }
    }
}

/* Elementwise products over COUNT instances i (contiguous numbers):
   RV_UP:    OUT[i] = A[2i] A[2i+1] (A[2i] if 2i+1 = NA): a tree level up;
   RV_DOWN:  A holds the inverses of the parents of level B (NA numbers):
             B[2i], B[2i+1] <- A[i] B[2i+1], A[i] B[2i], their inverses;
   RV_FINAL: OUT[i] (NL limbs) = A[i] B[i] / R: numerator times inverse,
             out of Montgomery form. */
template<int L>
__global__ void
rt_vec_kernel (uint32_t tpi, int mode, uint64_t count,
               const uint32_t * __restrict__ a, uint64_t na,
               uint32_t * __restrict__ b, uint32_t * __restrict__ out,
               uint32_t nl, const uint32_t * __restrict__ Ng,
               const uint32_t * __restrict__ oneg, uint32_t np0)
{
  gm_group g;
  g.init (tpi);
  const uint64_t inst = ((uint64_t) blockIdx.x * blockDim.x + threadIdx.x)
                        / g.tpi;
  const uint32_t n = g.tpi * L, off = g.lane * L;
  const bool live = inst < count;
  const uint64_t i = live ? inst : count - 1;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);

  uint32_t x[L], y[L];
  if (mode == RV_UP)
    {
      rt_cload<L> (x, a, 2 * i, n, off);
      if (2 * i + 1 < na)
        rt_cload<L> (y, a, 2 * i + 1, n, off);
      else
        rt_cload<L> (y, oneg, 0, n, off);
      gm_mul<L, false> (g, x, x, y, N, np0);
      if (live)
        rt_cstore<L> (out, x, i, n, off);
    }
  else if (mode == RV_DOWN)
    {
      uint32_t p[L];
      rt_cload<L> (p, a, i, n, off);
      rt_cload<L> (x, b, 2 * i, n, off);
      if (2 * i + 1 < na)
        rt_cload<L> (y, b, 2 * i + 1, n, off);
      else
        rt_cload<L> (y, oneg, 0, n, off);
      gm_mul<L, false> (g, y, p, y, N, np0);
      gm_mul<L, false> (g, x, p, x, N, np0);
      if (live)
        {
          rt_cstore<L> (b, y, 2 * i, n, off);
          if (2 * i + 1 < na)
            rt_cstore<L> (b, x, 2 * i + 1, n, off);
        }
    }
  else
    {
      rt_cload<L> (x, a, i, n, off);
      rt_cload<L> (y, b, i, n, off);
      gm_mul<L, false> (g, x, x, y, N, np0);
#pragma unroll
      for (int k = 0; k < L; k++)
        y[k] = off + k == 0;
      gm_mul<L, false> (g, x, x, y, N, np0);
      if (live)
#pragma unroll
        for (int k = 0; k < L; k++)
          if (off + k < nl)
            out[i * nl + off + k] = x[k];
    }
}

/* ---- kernel table ------------------------------------------------------ */

typedef void (*rt_setup_fn) (uint32_t, int, uint32_t, const uint32_t *,
                             uint32_t *, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, uint32_t,
                             uint32_t, const uint8_t *, const uint8_t *,
                             uint32_t *, uint32_t,
                             const uint32_t *, const uint32_t *, uint32_t);
typedef void (*rt_step_fn) (uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t *, const uint32_t *, const int32_t *,
                            uint64_t, uint32_t, uint32_t, uint32_t,
                            uint32_t *, uint32_t *, const uint32_t *,
                            uint32_t);
typedef void (*rt_vec_fn) (uint32_t, int, uint64_t, const uint32_t *,
                           uint64_t, uint32_t *, uint32_t *, uint32_t,
                           const uint32_t *, const uint32_t *, uint32_t);

struct rt_kernels {
  rt_setup_fn setup;
  rt_step_fn step;
  rt_vec_fn vec;
};

/* The limbs per thread with kernels (few, as each shape takes a while to
   compile); the shape for n limbs is the fewest threads per curve, 2 to 32,
   with L <= 8, else 32 threads; L the next one in this list. */
static const uint32_t rt_Ls[] = { 1, 2, 3, 4, 5, 6, 8, 12, 16, 24 };

template<int L>
static void
rt_put (rt_kernels *t)
{
  if constexpr (L <= GPU_ROOTS_MAX_L)
    {
      t[L].setup = rt_setup_kernel<L>;
      t[L].step = rt_step_kernel<L>;
      t[L].vec = rt_vec_kernel<L>;
    }
}

static const rt_kernels *
rt_get_kernels (uint32_t L)
{
  static rt_kernels t[25];
  static bool init = false;
  if (!init)
    {
      memset (t, 0, sizeof (t));
      rt_put<1> (t);
      rt_put<2> (t);
      rt_put<3> (t);
      rt_put<4> (t);
      rt_put<5> (t);
      rt_put<6> (t);
      rt_put<8> (t);
      rt_put<12> (t);
      rt_put<16> (t);
      rt_put<24> (t);
      init = true;
    }
  return L >= 1 && L <= 24 && t[L].setup != NULL ? &t[L] : NULL;
}

static bool
rt_shape (uint32_t nlimbs, uint32_t *tpi, uint32_t *L)
{
  for (uint32_t t = 2; t <= 32; t *= 2)
    for (uint32_t l : rt_Ls)
      if (l * t >= nlimbs && (l <= 8 || t == 32) && rt_get_kernels (l))
        {
          *tpi = t;
          *L = l;
          return true;
        }
  return false;
}

/* ---- host side --------------------------------------------------------- */

struct rt_z {
  mpz_t z;
  rt_z () { mpz_init (z); }
  ~rt_z () { mpz_clear (z); }
  rt_z (const rt_z &) = delete;
  rt_z &operator= (const rt_z &) = delete;
};

/* one set of progressions (F or G) */
struct rt_set {
  uint32_t nr = 0;        /* progressions per curve */
  uint32_t E = 0;         /* coefficients: nr S + 1 */
  uint64_t st = 0;        /* stride of the progressions (in j or m) */
  uint32_t steps = 0;     /* F: steps */
  uint32_t pinst = 0;     /* padded instances (curves nr) */
  uint32_t maxbits = 0;
  std::vector<rt_z> *co = NULL;   /* the coefficients until digitised */
  uint8_t *d_dig = NULL;
  uint8_t *d_sgn = NULL;  /* signs of the coefficients */
  uint32_t *d_state = NULL;
  int32_t *d_slots = NULL;
};

struct gpu_roots {
  mpz_t N, R, R2;         /* R = 2^(32 n) mod N, R2 = R^2 mod N */
  uint32_t tpi, L, n, np0, C, nl;
  size_t dF;
  unsigned long blocks;
  int S, dickson_a, verbose;
  const rt_kernels *k;
  uint32_t K, w, nw;      /* digits of the coefficients, window */
  uint32_t *d_N, *d_one, *d_cst, *d_W;
  uint32_t *d_num, *d_den, *d_tree;
  rt_set F, G;
  uint64_t gblock;        /* the next block of G */
};

static unsigned long
rt_gcd (unsigned long a, unsigned long b)
{
  while (b)
    {
      unsigned long t = a % b;
      a = b;
      b = t;
    }
  return a;
}

/* r = Dickson_(S, a) (x): D_0 = 2, D_1 = x, D_k = x D_(k-1) - a D_(k-2),
   which is x^S for a = 0 (S > 0) */
static void
rt_dickson (mpz_t r, const mpz_t x, unsigned S, int a)
{
  if (a == 0)
    {
      mpz_pow_ui (r, x, S);
      return;
    }
  mpz_t p, q, t;
  mpz_init_set_ui (p, 2);
  mpz_init_set (q, x);
  mpz_init (t);
  for (unsigned k = 2; k <= S; k++)
    {
      mpz_mul (t, x, q);
      if (a < 0)
        mpz_addmul_ui (t, p, (unsigned long) -a);
      else
        mpz_submul_ui (t, p, (unsigned long) a);
      mpz_swap (p, q);
      mpz_swap (q, t);
    }
  mpz_set (r, q);
  mpz_clears (p, q, t, NULL);
}

/* the coefficients of a set of NR progressions with first arguments V0[r]
   and difference DV: CO[r S + t] = Delta^t f_r (0), CO[nr S] = Delta^S */
static std::vector<rt_z> *
rt_coeffs (uint32_t nr, int S, int a, const std::vector<rt_z> &v0,
           const mpz_t dv)
{
  std::vector<rt_z> *co = new std::vector<rt_z> (nr * S + 1);
  std::vector<rt_z> f (S + 1);
  mpz_t arg;
  mpz_init (arg);
  for (uint32_t r = 0; r < nr; r++)
    {
      for (int i = 0; i <= S; i++)
        {
          mpz_set (arg, v0[r].z);
          mpz_addmul_ui (arg, dv, i);
          rt_dickson (f[i].z, arg, S, a);
        }
      for (int k = 1; k <= S; k++)
        for (int i = S; i >= k; i--)
          mpz_sub (f[i].z, f[i].z, f[i - 1].z);
      for (int t = 0; t < S; t++)
        mpz_set ((*co)[r * S + t].z, f[t].z);
      if (r == 0)
        mpz_set ((*co)[nr * S].z, f[S].z);
      else
        assert (mpz_cmp ((*co)[nr * S].z, f[S].z) == 0);
    }
  mpz_clear (arg);
  return co;
}

static uint32_t
rt_maxbits (const std::vector<rt_z> &co)
{
  size_t b = 1;
  for (const rt_z &c : co)
    b = std::max (b, mpz_sizeinbase (c.z, 2));
  return (uint32_t) b;
}

template<class T>
static T *
rt_upload (const std::vector<T> &h)
{
  T *d;
  CUDA_CHECK (cudaMalloc ((void **) &d, std::max ((size_t) 1, h.size ())
                                        * sizeof (T)));
  if (!h.empty ())
    CUDA_CHECK (cudaMemcpy (d, h.data (), h.size () * sizeof (T),
                            cudaMemcpyHostToDevice));
  return d;
}

/* launch geometry for COUNT instances */
static uint64_t
rt_padded (const gpu_roots *G, uint64_t count)
{
  const uint64_t per = GPU_TPB / G->tpi;
  return (count + per - 1) / per * per;
}

static uint32_t
rt_blocks (const gpu_roots *G, uint64_t count)
{
  return (uint32_t) (rt_padded (G, count) * G->tpi / GPU_TPB);
}

/* progressions per curve wanted for ROOTS roots: enough instances to fill
   the GPU, but at most ROOTS / (4 K) so that the setup (S K additions per
   progression) stays small next to the steps (S per root) */
static uint32_t
rt_nr_target (const gpu_roots *G, double roots, double kbits)
{
  const char *env = getenv ("GMPECM_ROOTS_NR");
  if (env != NULL && atoi (env) > 0)
    return (uint32_t) atoi (env);
  int dev, sm = 1;
  CUDA_CHECK (cudaGetDevice (&dev));
  CUDA_CHECK (cudaDeviceGetAttribute (&sm, cudaDevAttrMultiProcessorCount,
                                      dev));
  double fill = 4.0 * sm * 1024 / ((double) G->tpi * G->C);
  double bal = roots / (4.0 * std::max (1.0, kbits / 4));
  double nr = std::min (fill, bal);
  return nr < 1 ? 1 : (uint32_t) std::min (nr, 65536.0);
}

size_t
gpu_roots_max_bits (void)
{
  uint32_t lmax = 0;
  for (uint32_t l : rt_Ls)
    if (rt_get_kernels (l))
      lmax = l;
  return 32 * 32 * (size_t) lmax - GPU_SPARE_BITS;
}

static void
rt_set_alloc (gpu_roots *G, rt_set &s)
{
  s.E = s.nr * std::abs (G->S) + 1;
  s.pinst = (uint32_t) rt_padded (G, (uint64_t) G->C * s.nr);
  size_t words = (size_t) (std::abs (G->S) + 1) * 4 * G->n * s.pinst;
  CUDA_CHECK (cudaMalloc ((void **) &s.d_state, words * 4));
  CUDA_CHECK (cudaMemset (s.d_state, 0, words * 4));
}

/* F: the first dF j = 1 (mod 6) prime to d1 */
static void
rt_plan_F (gpu_roots *G, unsigned long d1, unsigned long d2)
{
  rt_set &s = G->F;
  std::vector<int32_t> rank;          /* of j = 6 i + 1, or -1 */
  uint64_t j = 1;
  for (size_t cnt = 0; cnt < G->dF; j += 6)
    rank.push_back (rt_gcd (j, d1) == 1 ? (int32_t) cnt++ : -1);
  const uint64_t jmax = j - 6;

  /* m: primes >= 5 of d1 while phi (m) stays within the target */
  const int S = std::abs (G->S);
  double kbits = S * (log2 ((double) d2 * jmax) + 1);
  uint32_t target = rt_nr_target (G, (double) G->dF, kbits);
  unsigned long m = 1, ph = 1, rest = d1;
  for (unsigned long p = 2; rest > 1; p++)
    {
      if (p * p > rest)
        p = rest;
      if (rest % p != 0)
        continue;
      while (rest % p == 0)
        rest /= p;
      if (p >= 5 && ph * (p - 1) <= target)
        {
          m *= p;
          ph *= p - 1;
        }
    }
  const uint32_t kf = std::max ((uint32_t) 1, (uint32_t) (target / ph));
  s.nr = (uint32_t) ph * kf;
  s.st = 6 * (uint64_t) m * kf;
  std::vector<uint64_t> start;
  for (uint32_t q = 0; q < kf; q++)
    for (uint64_t u = 1; u < 6 * m; u += 6)
      if (rt_gcd (u, m) == 1)
        start.push_back (u + 6 * m * q);
  assert (start.size () == s.nr);
  s.steps = (uint32_t) ((jmax - 1) / s.st + 1);

  std::vector<int32_t> slots ((size_t) s.steps * s.nr);
  for (uint32_t nn = 0; nn < s.steps; nn++)
    for (uint32_t r = 0; r < s.nr; r++)
      {
        uint64_t jj = start[r] + nn * s.st;
        slots[(size_t) nn * s.nr + r] = jj <= jmax ? rank[(jj - 1) / 6] : -1;
      }
  s.d_slots = rt_upload (slots);

  std::vector<rt_z> v0 (s.nr);
  mpz_t dv;
  mpz_init (dv);
  for (uint32_t r = 0; r < s.nr; r++)
    {
      mpz_set_ui (v0[r].z, d2);
      mpz_mul_ui (v0[r].z, v0[r].z, start[r]);
    }
  mpz_set_ui (dv, d2);
  mpz_mul_ui (dv, dv, s.st);
  s.co = rt_coeffs (s.nr, S, G->dickson_a, v0, dv);
  s.maxbits = rt_maxbits (*s.co);
  mpz_clear (dv);
}

/* G: m = i0, i0 + 1, ... prime to d2 */
static void
rt_plan_G (gpu_roots *G, unsigned long d1, unsigned long d2, const mpz_t i0)
{
  rt_set &s = G->G;
  const int S = std::abs (G->S);
  const unsigned long i0m = mpz_fdiv_ui (i0, d2);
  unsigned long ph = 0;
  for (unsigned long o = 0; o < d2; o++)
    ph += rt_gcd ((i0m + o) % d2, d2) == 1;
  double last = mpz_get_d (i0) + (double) G->dF * G->blocks * d2 / ph;
  double kbits = S * (log2 ((double) d1 * (last + 1)) + 1);
  uint32_t target = rt_nr_target (G, (double) G->dF * G->blocks, kbits);
  const uint32_t kg = std::max ((uint32_t) 1, (uint32_t) (target / ph));
  s.nr = (uint32_t) ph * kg;
  s.st = (uint64_t) d2 * kg;

  std::vector<rt_z> v0 (s.nr);
  mpz_t dv;
  mpz_init (dv);
  uint32_t r = 0;
  for (uint64_t o = 0; o < s.st; o++)
    if (rt_gcd ((i0m + o) % d2, d2) == 1)
      {
        mpz_add_ui (v0[r].z, i0, o);
        mpz_mul_ui (v0[r].z, v0[r].z, d1);
        r++;
      }
  assert (r == s.nr);
  mpz_set_ui (dv, d1);
  mpz_mul_ui (dv, dv, s.st);
  s.co = rt_coeffs (s.nr, S, G->dickson_a, v0, dv);
  s.maxbits = rt_maxbits (*s.co);
  mpz_clear (dv);
}

static void
rt_digits (gpu_roots *G, rt_set &s)
{
  std::vector<uint8_t> dig ((size_t) s.E * G->K, 0), sgn (s.E);
  mpz_t a;
  mpz_init (a);
  for (uint32_t e = 0; e < s.E; e++)
    {
      sgn[e] = mpz_sgn ((*s.co)[e].z) < 0;
      mpz_abs (a, (*s.co)[e].z);
      for (uint32_t k = 0; k < G->K; k++)
        {
          uint32_t d = 0;
          for (uint32_t b = 0; b < G->w; b++)
            d |= (uint32_t) mpz_tstbit (a, k * G->w + b) << b;
          dig[(size_t) e * G->K + k] = (uint8_t) d;
        }
    }
  mpz_clear (a);
  s.d_dig = rt_upload (dig);
  s.d_sgn = rt_upload (sgn);
  delete s.co;
  s.co = NULL;
}

gpu_roots *
gpu_roots_create (const mpz_t N, uint32_t tpi, uint32_t L, int S,
                  unsigned long d1, unsigned long d2, const mpz_t i0,
                  size_t dF, unsigned long blocks, int nl, uint32_t curves,
                  int verbose)
{
  const uint32_t need = (mpz_sizeinbase (N, 2) + GPU_SPARE_BITS + 31) / 32;
  if (tpi == 0 && !rt_shape (need, &tpi, &L))
    return NULL;
  const rt_kernels *k = rt_get_kernels (L);
  if (k == NULL || tpi < 2 || tpi > 32 || (tpi & (tpi - 1)) != 0
      || tpi * L < need || S == 0 || dF == 0 || curves == 0 || d2 == 0
      || rt_gcd (d1, d2) != 1)
    return NULL;
  gpu_roots *G = new gpu_roots;
  G->k = k;
  G->tpi = tpi;
  G->L = L;
  G->n = tpi * L;
  assert (mpz_sizeinbase (N, 2) + GPU_SPARE_BITS <= 32 * (size_t) G->n);
  assert ((size_t) nl * 32 > mpz_sizeinbase (N, 2) && (uint32_t) nl <= G->n);
  G->C = curves;
  G->nl = nl;
  G->dF = dF;
  G->blocks = std::max (blocks, 1ul);
  G->S = std::abs (S);
  G->dickson_a = S < 0 ? -1 : 0;
  G->verbose = verbose;
  G->gblock = 0;
  mpz_inits (G->N, G->R, G->R2, NULL);
  mpz_set (G->N, N);
  mpz_setbit (G->R, 32 * (size_t) G->n);
  mpz_mod (G->R, G->R, N);
  mpz_mul (G->R2, G->R, G->R);
  mpz_mod (G->R2, G->R2, N);
  G->np0 = gpu_find_np0 (N);

  std::vector<uint32_t> h (G->n);
  gpu_to_limbs (h.data (), G->n, N);
  G->d_N = rt_upload (h);
  gpu_to_limbs (h.data (), G->n, G->R);
  G->d_one = rt_upload (h);

  rt_plan_F (G, d1, d2);
  rt_plan_G (G, d1, d2, i0);
  rt_set_alloc (G, G->F);
  rt_set_alloc (G, G->G);

  /* the window: as wide as fits in a quarter of the free memory */
  size_t free_mem, total_mem;
  CUDA_CHECK (cudaMemGetInfo (&free_mem, &total_mem));
  const uint32_t bits = std::max (G->F.maxbits, G->G.maxbits);
  for (G->w = 4; G->w > 1; G->w--)
    {
      size_t K = (bits + G->w - 1) / G->w;
      size_t bytes = (size_t) curves * K * ((1u << G->w) - 1) * 4 * G->n * 4;
      if (bytes < free_mem / 4)
        break;
    }
  G->nw = (1u << G->w) - 1;
  G->K = (bits + G->w - 1) / G->w;
  rt_digits (G, G->F);
  rt_digits (G, G->G);

  CUDA_CHECK (cudaMalloc ((void **) &G->d_cst,
                          (size_t) curves * RC_COUNT * G->n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &G->d_W,
                          (size_t) curves * G->K * G->nw * 4 * G->n * 4));
  const size_t M = (size_t) curves * dF;
  CUDA_CHECK (cudaMalloc ((void **) &G->d_num, M * G->n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &G->d_den, M * G->n * 4));
  CUDA_CHECK (cudaMalloc ((void **) &G->d_tree, (M + 64) * G->n * 4));

  if (verbose)
    printf ("GPU roots: %u curves, S = %d%s, F: %u progressions (stride "
            "%lu, %u steps), G: %u progressions (stride %lu), %u-bit "
            "coefficients, window %u\n", curves, G->S,
            G->dickson_a ? " (Dickson)" : "", G->F.nr,
            (unsigned long) G->F.st, G->F.steps, G->G.nr,
            (unsigned long) G->G.st, bits, G->w);
  return G;
}

static void
rt_set_free (rt_set &s)
{
  delete s.co;
  if (s.d_dig)
    CUDA_CHECK (cudaFree (s.d_dig));
  if (s.d_sgn)
    CUDA_CHECK (cudaFree (s.d_sgn));
  if (s.d_state)
    CUDA_CHECK (cudaFree (s.d_state));
  if (s.d_slots)
    CUDA_CHECK (cudaFree (s.d_slots));
}

void
gpu_roots_destroy (gpu_roots *G)
{
  if (G == NULL)
    return;
  rt_set_free (G->F);
  rt_set_free (G->G);
  CUDA_CHECK (cudaFree (G->d_N));
  CUDA_CHECK (cudaFree (G->d_one));
  CUDA_CHECK (cudaFree (G->d_cst));
  CUDA_CHECK (cudaFree (G->d_W));
  CUDA_CHECK (cudaFree (G->d_num));
  CUDA_CHECK (cudaFree (G->d_den));
  CUDA_CHECK (cudaFree (G->d_tree));
  mpz_clears (G->N, G->R, G->R2, NULL);
  delete G;
}

/* the starting tables of a set */
static void
rt_start_set (gpu_roots *G, rt_set &s)
{
  const uint64_t count = (uint64_t) G->C * s.E;
  const uint32_t Ts = s.pinst * G->tpi;
  for (uint32_t k0 = 0; k0 < G->K; k0 += RT_SLICE)
    {
      uint32_t k1 = std::min (G->K, k0 + RT_SLICE);
      G->k->setup<<<rt_blocks (G, count), GPU_TPB>>> (
          G->tpi, RS_SCALAR, (uint32_t) count, G->d_cst, G->d_W, G->K, G->w,
          G->nw, k0, k1, s.E, s.nr, G->S, s.d_dig, s.d_sgn, s.d_state, Ts,
          G->d_N,
          G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
    }
}

int
gpu_roots_curves (gpu_roots *G, mpz_t *A, mpz_t *x, mpz_t f, int *curve)
{
  const uint32_t n = G->n;
  std::vector<uint32_t> h ((size_t) G->C * RC_COUNT * n);
  mpz_t g, u, inv, t, c[RC_COUNT];
  mpz_inits (g, u, inv, t, NULL);
  for (int i = 0; i < RC_COUNT; i++)
    mpz_init (c[i]);
  int found = 0;
  for (uint32_t cv = 0; cv < G->C && !found; cv++)
    {
      /* g = x^3 + A x^2 + x, and 1 / (g (x + 1)) */
      mpz_add (g, x[cv], A[cv]);
      mpz_mul (g, g, x[cv]);
      mpz_add_ui (g, g, 1);
      mpz_mul (g, g, x[cv]);
      mpz_mod (g, g, G->N);
      mpz_add_ui (u, x[cv], 1);
      mpz_mul (t, g, u);
      if (!mpz_invert (inv, t, G->N))
        {
          mpz_gcd (f, t, G->N);
          *curve = (int) cv;
          found = 1;
          break;
        }
      mpz_mul (t, inv, u);                 /* 1 / g */
      mpz_add_ui (c[RC_A], A[cv], 2);
      mpz_mul (c[RC_A], c[RC_A], t);       /* a = (A + 2) / g */
      mpz_sub_ui (c[RC_D], A[cv], 2);
      mpz_mul (c[RC_D], c[RC_D], t);       /* d = (A - 2) / g */
      mpz_add_ui (c[RC_C1], A[cv], 3);
      mpz_ui_sub (c[RC_C2], 3, A[cv]);
      mpz_mul_ui (c[RC_C3], g, 3);
      mpz_set (c[RC_PX], x[cv]);
      mpz_sub_ui (c[RC_PY], x[cv], 1);
      mpz_mul (c[RC_PY], c[RC_PY], inv);
      mpz_mul (c[RC_PY], c[RC_PY], g);     /* (x - 1) / (x + 1) */
      mpz_mod (c[RC_PY], c[RC_PY], G->N);
      mpz_set_ui (c[RC_PZ], 1);
      mpz_mul (c[RC_PT], c[RC_PX], c[RC_PY]);
      for (int i = 0; i < RC_COUNT; i++)
        {
          mpz_mul (c[i], c[i], G->R);
          mpz_mod (c[i], c[i], G->N);
          gpu_to_limbs (&h[((size_t) cv * RC_COUNT + i) * n], n, c[i]);
        }
    }
  mpz_clears (g, u, inv, t, NULL);
  for (int i = 0; i < RC_COUNT; i++)
    mpz_clear (c[i]);
  if (found)
    return 1;

  CUDA_CHECK (cudaMemcpy (G->d_cst, h.data (), h.size () * 4,
                          cudaMemcpyHostToDevice));
  for (uint32_t k0 = 0; k0 < G->K; k0 += RT_SLICE)
    {
      uint32_t k1 = std::min (G->K, k0 + RT_SLICE);
      G->k->setup<<<rt_blocks (G, G->C), GPU_TPB>>> (
          G->tpi, RS_CHAIN, G->C, G->d_cst, G->d_W, G->K, G->w, G->nw, k0,
          k1, 0, 0, 0, NULL, NULL, NULL, 0, G->d_N, G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
    }
  if (G->nw > 1)
    {
      G->k->setup<<<rt_blocks (G, (uint64_t) G->C * G->K), GPU_TPB>>> (
          G->tpi, RS_FILL, G->C * G->K, G->d_cst, G->d_W, G->K, G->w, G->nw,
          0, 0, 0, 0, 0, NULL, NULL, NULL, 0, G->d_N, G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
    }
  rt_start_set (G, G->F);
  rt_start_set (G, G->G);
  G->gblock = 0;
  CUDA_CHECK (cudaDeviceSynchronize ());
  return 0;
}

/* The roots from the numerators and denominators of all curves' dF
   slots, into OUT */
static int
rt_finish (gpu_roots *G, uint32_t *out, mpz_t f, int *curve)
{
  const uint32_t n = G->n;
  const uint64_t M = (uint64_t) G->C * G->dF;
  /* levels of the product tree: level 0 is den, the others in d_tree */
  std::vector<uint32_t *> lev (1, G->d_den);
  std::vector<uint64_t> cnt (1, M);
  uint32_t *p = G->d_tree;
  while (cnt.back () > 1)
    {
      uint64_t c = (cnt.back () + 1) / 2;
      G->k->vec<<<rt_blocks (G, c), GPU_TPB>>> (
          G->tpi, RV_UP, c, lev.back (), cnt.back (), NULL, p, 0, G->d_N,
          G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      lev.push_back (p);
      cnt.push_back (c);
      p += c * n;
    }

  std::vector<uint32_t> h (n);
  CUDA_CHECK (cudaMemcpy (h.data (), lev.back (), n * 4,
                          cudaMemcpyDeviceToHost));
  mpz_t v, t;
  mpz_inits (v, t, NULL);
  gpu_from_limbs (v, h.data (), n);
  int found = 0;
  if (!mpz_invert (t, v, G->N))
    {
      /* some root is the identity mod a prime of N: find its curve */
      found = 1;
      mpz_gcd (f, v, G->N);
      *curve = -1;
      std::vector<uint32_t> all ((size_t) M * n);
      CUDA_CHECK (cudaMemcpy (all.data (), G->d_den, all.size () * 4,
                              cudaMemcpyDeviceToHost));
      for (uint32_t c = 0; c < G->C && *curve < 0; c++)
        {
          mpz_set_ui (v, 1);
          for (size_t i = 0; i < G->dF; i++)
            {
              gpu_from_limbs (t, &all[((size_t) c * G->dF + i) * n], n);
              mpz_mul (v, v, t);
              mpz_mod (v, v, G->N);
            }
          mpz_gcd (t, v, G->N);
          if (mpz_cmp_ui (t, 1) != 0)
            {
              mpz_set (f, t);
              *curve = (int) c;
            }
        }
    }
  else
    {
      /* inverse of v / R, in Montgomery form: R^2 / v */
      mpz_mul (t, t, G->R2);
      mpz_mod (t, t, G->N);
      gpu_to_limbs (h.data (), n, t);
      CUDA_CHECK (cudaMemcpy (lev.back (), h.data (), n * 4,
                              cudaMemcpyHostToDevice));
      for (size_t l = lev.size () - 1; l > 0; l--)
        {
          G->k->vec<<<rt_blocks (G, cnt[l]), GPU_TPB>>> (
              G->tpi, RV_DOWN, cnt[l], lev[l], cnt[l - 1], lev[l - 1], NULL,
              0, G->d_N, G->d_one, G->np0);
          CUDA_CHECK (cudaGetLastError ());
        }
      G->k->vec<<<rt_blocks (G, M), GPU_TPB>>> (
          G->tpi, RV_FINAL, M, G->d_num, M, G->d_den, out, G->nl, G->d_N,
          G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      CUDA_CHECK (cudaDeviceSynchronize ());
    }
  mpz_clears (v, t, NULL);
  return found;
}

int
gpu_roots_F (gpu_roots *G, uint32_t *out, mpz_t f, int *curve)
{
  rt_set &s = G->F;
  const uint64_t ninst = (uint64_t) G->C * s.nr;
  for (uint32_t i0 = 0; i0 < s.steps; i0 += RT_SLICE)
    {
      uint32_t i1 = std::min (s.steps, i0 + RT_SLICE);
      G->k->step<<<rt_blocks (G, ninst), GPU_TPB>>> (
          G->tpi, (uint32_t) ninst, s.nr, G->S, s.d_state, G->d_cst,
          s.d_slots, 0, (uint32_t) G->dF, i0, i1, G->d_num, G->d_den,
          G->d_N, G->np0);
      CUDA_CHECK (cudaGetLastError ());
    }
  return rt_finish (G, out, f, curve);
}

int
gpu_roots_G (gpu_roots *G, uint32_t *out, mpz_t f, int *curve)
{
  rt_set &s = G->G;
  const uint64_t ninst = (uint64_t) G->C * s.nr;
  const uint64_t base = G->gblock++ * G->dF;
  /* the most steps any progression has in [base, base + dF) */
  uint32_t its = 0;
  for (uint32_t r = 0; r < s.nr; r++)
    {
      int64_t lo = (int64_t) base - r, hi = (int64_t) (base + G->dF) - r;
      int64_t nlo = lo > 0 ? (lo + s.nr - 1) / s.nr : 0;
      int64_t nhi = hi > 0 ? (hi + s.nr - 1) / s.nr : 0;
      its = std::max (its, (uint32_t) (nhi - nlo));
    }
  for (uint32_t i0 = 0; i0 < its; i0 += RT_SLICE)
    {
      uint32_t i1 = std::min (its, i0 + RT_SLICE);
      G->k->step<<<rt_blocks (G, ninst), GPU_TPB>>> (
          G->tpi, (uint32_t) ninst, s.nr, G->S, s.d_state, G->d_cst, NULL,
          base, (uint32_t) G->dF, i0, i1, G->d_num, G->d_den, G->d_N,
          G->np0);
      CUDA_CHECK (cudaGetLastError ());
    }
  return rt_finish (G, out, f, curve);
}
