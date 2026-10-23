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
 * instances repeat the last one and store nothing).
 *
 * For large N (from 12000 bits, when they time faster: rt_pick_ntt) the
 * same computations run on the transform products of gpu_nmul.cuh
 * instead, one block per instance ("NTT backend" below); and so, chosen
 * separately, do the Z/N tree and transposed-evaluation levels up to
 * degree NT_ZMAX. */

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
#include "gpu_nmul.cuh"

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

template<int L, class G>
GM_DEV void
rt_modulus (const G &g, uint32_t (&N)[L], uint32_t (&N2)[L],
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
template<int L, class G>
GM_DEV void
rt_add (const G &g, rt_pt<L> &r, const rt_pt<L> &p,
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
template<int L, class G>
__device__ __forceinline__ void
rt_setup_body (const G &g, uint32_t tpi, int mode, uint32_t count,
                 const uint32_t * __restrict__ cst, uint32_t * __restrict__ W,
                 uint32_t K, uint32_t w, uint32_t nw, uint32_t k0,
                 uint32_t k1, uint32_t E, uint32_t nr, uint32_t S,
                 const uint8_t * __restrict__ dig,
                 const uint8_t * __restrict__ sgn,
                 uint32_t * __restrict__ state, uint32_t Ts,
                 const uint32_t * __restrict__ Ng,
                 const uint32_t * __restrict__ oneg, uint32_t np0)
{
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
  rt_setup_body<L> (g, tpi, mode, count, cst, W, K, w, nw, k0, k1, E, nr, S, dig, sgn, state, Ts, Ng, oneg, np0);
}

/* the same with a wide group (64 or 128 threads per instance) */
template<int L>
__global__ void
rt_setup_wkernel (uint32_t tpi, int mode, uint32_t count,
                 const uint32_t * __restrict__ cst, uint32_t * __restrict__ W,
                 uint32_t K, uint32_t w, uint32_t nw, uint32_t k0,
                 uint32_t k1, uint32_t E, uint32_t nr, uint32_t S,
                 const uint8_t * __restrict__ dig,
                 const uint8_t * __restrict__ sgn,
                 uint32_t * __restrict__ state, uint32_t Ts,
                 const uint32_t * __restrict__ Ng,
                 const uint32_t * __restrict__ oneg, uint32_t np0)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  rt_setup_body<L> (g, tpi, mode, count, cst, W, K, w, nw, k0, k1, E, nr, S, dig, sgn, state, Ts, Ng, oneg, np0);
}

/* Steps IT in [IT0, IT1) of all progressions (one instance per (curve c,
   progression r), C nr of them; the state is interleaved over this
   launch's threads).  Step IT emits the root of fd[0] (unless its slot is
   outside the roots wanted now), then moves the table on.
   F (SLOTS != NULL): step n = IT, slot SLOTS[n nr + r] (-1: none).
   G: the roots [BASE, BASE + dF); progression r has root numbers r + n nr,
   so its steps n in [nlo, nhi) are the ones in range, n = nlo + IT; the
   state of an instance with no more steps is not updated. */
template<int L, class G>
__device__ __forceinline__ void
rt_step_body (const G &g, uint32_t tpi, uint32_t ninst, uint32_t nr, uint32_t S,
                uint32_t * __restrict__ state,
                const uint32_t * __restrict__ cst,
                const int32_t * __restrict__ slots, uint64_t base,
                uint32_t dF, uint32_t it0, uint32_t it1,
                uint32_t * __restrict__ num, uint32_t * __restrict__ den,
                const uint32_t * __restrict__ Ng, uint32_t np0)
{
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
  rt_step_body<L> (g, tpi, ninst, nr, S, state, cst, slots, base, dF, it0, it1, num, den, Ng, np0);
}

/* the same with a wide group (64 or 128 threads per instance) */
template<int L>
__global__ void
rt_step_wkernel (uint32_t tpi, uint32_t ninst, uint32_t nr, uint32_t S,
                uint32_t * __restrict__ state,
                const uint32_t * __restrict__ cst,
                const int32_t * __restrict__ slots, uint64_t base,
                uint32_t dF, uint32_t it0, uint32_t it1,
                uint32_t * __restrict__ num, uint32_t * __restrict__ den,
                const uint32_t * __restrict__ Ng, uint32_t np0)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  rt_step_body<L> (g, tpi, ninst, nr, S, state, cst, slots, base, dF, it0, it1, num, den, Ng, np0);
}

/* Elementwise products over COUNT instances i (contiguous numbers):
   RV_UP:    OUT[i] = A[2i] A[2i+1] (A[2i] if 2i+1 = NA): a tree level up;
   RV_DOWN:  A holds the inverses of the parents of level B (NA numbers):
             B[2i], B[2i+1] <- A[i] B[2i+1], A[i] B[2i], their inverses;
   RV_FINAL: OUT[i] (NL limbs) = A[i] B[i] / R: numerator times inverse,
             out of Montgomery form. */
template<int L, class G>
__device__ __forceinline__ void
rt_vec_body (const G &g, uint32_t tpi, int mode, uint64_t count,
               const uint32_t * __restrict__ a, uint64_t na,
               uint32_t * __restrict__ b, uint32_t * __restrict__ out,
               uint32_t nl, const uint32_t * __restrict__ Ng,
               const uint32_t * __restrict__ oneg, uint32_t np0)
{
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
  rt_vec_body<L> (g, tpi, mode, count, a, na, b, out, nl, Ng, oneg, np0);
}

/* the same with a wide group (64 or 128 threads per instance) */
template<int L>
__global__ void
rt_vec_wkernel (uint32_t tpi, int mode, uint64_t count,
               const uint32_t * __restrict__ a, uint64_t na,
               uint32_t * __restrict__ b, uint32_t * __restrict__ out,
               uint32_t nl, const uint32_t * __restrict__ Ng,
               const uint32_t * __restrict__ oneg, uint32_t np0)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  rt_vec_body<L> (g, tpi, mode, count, a, na, b, out, nl, Ng, oneg, np0);
}

/* number I (NL limbs, plain, below 2N) into this lane's L limbs */
template<int L>
GM_DEV void
rt_nload (uint32_t (&x)[L], const uint32_t * __restrict__ buf, size_t i,
          uint32_t nl, uint32_t off)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    x[k] = off + k < nl ? buf[i * nl + off + k] : 0;
}

/* One level of a product tree in Z/N (plain numbers of NL limbs below 2N):
   the nodes x^d + A, x^d + B of IN (D numbers each, node pairs side by
   side) give x^2d + C, C the 2D numbers at the same place in OUT:
     C_k = sum_(i + j = k) a_i b_j + (a_(k-d) + b_(k-d) if k >= d).
   One instance per output number (COUNT of them); every instance runs all
   D products i (with b_j = 0 out of range: a warp's lanes must run the
   same products).  The sum is of Montgomery products a_i b_j / R, so it is
   multiplied by R^2 / R at the end (R2G: R^2 mod N). */
template<int L, class G>
__device__ __forceinline__ void
rt_tree_body (const G &g, uint32_t tpi, uint64_t count, uint32_t d,
                const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                uint32_t nl, const uint32_t * __restrict__ Ng,
                const uint32_t * __restrict__ R2g, uint32_t np0)
{
  const uint64_t inst = ((uint64_t) blockIdx.x * blockDim.x + threadIdx.x)
                        / g.tpi;
  const uint32_t off = g.lane * L;
  const bool live = inst < count;
  const uint64_t o = live ? inst : count - 1;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);
  const uint64_t pair = o / (2 * d);
  const int64_t k = (int64_t) (o % (2 * d));
  const uint32_t *A = in + pair * 2 * d * nl, *Bp = A + (size_t) d * nl;

  uint32_t acc[L], x[L], y[L];
#pragma unroll
  for (int t = 0; t < L; t++)
    acc[t] = 0;
  for (uint32_t i = 0; i < d; i++)
    {
      const int64_t j = k - i;
      rt_nload<L> (x, A, i, nl, off);
      if (j >= 0 && j < d)
        rt_nload<L> (y, Bp, (size_t) j, nl, off);
      else
#pragma unroll
        for (int t = 0; t < L; t++)
          y[t] = 0;
      gm_mul<L, false> (g, x, x, y, N, np0);
      gm_add<L, false> (g, acc, acc, x);
      gm_sub<L, false> (g, acc, acc, N2, N2);     /* below 2N */
    }
#pragma unroll
  for (int t = 0; t < L; t++)
    y[t] = R2g[off + t];
  gm_mul<L, false> (g, acc, acc, y, N, np0);
  /* + a_(k-d) + b_(k-d) */
  if (k >= d)
    {
      rt_nload<L> (x, A, (size_t) (k - d), nl, off);
      rt_nload<L> (y, Bp, (size_t) (k - d), nl, off);
    }
  else
#pragma unroll
    for (int t = 0; t < L; t++)
      x[t] = y[t] = 0;
  gm_add<L, false> (g, acc, acc, x);
  gm_sub<L, false> (g, acc, acc, N2, N2);
  gm_add<L, false> (g, acc, acc, y);
  gm_sub<L, false> (g, acc, acc, N2, N2);
  if (live)
#pragma unroll
    for (int t = 0; t < L; t++)
      if (off + t < nl)
        out[o * nl + off + t] = acc[t];
}

template<int L>
__global__ void
rt_tree_kernel (uint32_t tpi, uint64_t count, uint32_t d,
                const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                uint32_t nl, const uint32_t * __restrict__ Ng,
                const uint32_t * __restrict__ R2g, uint32_t np0)
{
  gm_group g;
  g.init (tpi);
  rt_tree_body<L> (g, tpi, count, d, in, out, nl, Ng, R2g, np0);
}

/* the same with a wide group (64 or 128 threads per instance) */
template<int L>
__global__ void
rt_tree_wkernel (uint32_t tpi, uint64_t count, uint32_t d,
                const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                uint32_t nl, const uint32_t * __restrict__ Ng,
                const uint32_t * __restrict__ R2g, uint32_t np0)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  rt_tree_body<L> (g, tpi, count, d, in, out, nl, Ng, R2g, np0);
}

/* One level of the transposed evaluation in Z/N (see gpu_poly_tpeval): for
   each node (K = 2 HALF values of TV, its children L, R of degree HALF in
   TREE at the same place), the new values are
     out[i]        = sum_j R_j TV[i + j] + TV[HALF + i],
     out[HALF + i] = sum_j L_j TV[i + j] + TV[HALF + i],   i, j < HALF.
   One instance per output number (COUNT); all run HALF products. */
template<int L, class G>
__device__ __forceinline__ void
rt_mp_body (const G &g, uint32_t tpi, uint64_t count, uint32_t half,
              const uint32_t * __restrict__ tv,
              const uint32_t * __restrict__ tree, uint32_t * __restrict__ out,
              uint32_t nl, const uint32_t * __restrict__ Ng,
              const uint32_t * __restrict__ R2g, uint32_t np0)
{
  const uint64_t inst = ((uint64_t) blockIdx.x * blockDim.x + threadIdx.x)
                        / g.tpi;
  const uint32_t off = g.lane * L;
  const bool live = inst < count;
  const uint64_t o = live ? inst : count - 1;
  uint32_t N[L], N2[L];
  rt_modulus<L> (g, N, N2, Ng, off);
  const uint64_t k = 2 * (uint64_t) half, node = o / k;
  const uint32_t r = (uint32_t) (o % k), i = r % half;
  const uint32_t *T = tv + node * k * nl;
  const uint32_t *child = tree + (node * k + (r < half ? half : 0)) * nl;

  uint32_t acc[L], x[L], y[L];
#pragma unroll
  for (int t = 0; t < L; t++)
    acc[t] = 0;
  for (uint32_t j = 0; j < half; j++)
    {
      rt_nload<L> (x, child, j, nl, off);
      rt_nload<L> (y, T, i + j, nl, off);
      gm_mul<L, false> (g, x, x, y, N, np0);
      gm_add<L, false> (g, acc, acc, x);
      gm_sub<L, false> (g, acc, acc, N2, N2);
    }
#pragma unroll
  for (int t = 0; t < L; t++)
    y[t] = R2g[off + t];
  gm_mul<L, false> (g, acc, acc, y, N, np0);
  rt_nload<L> (x, T, half + i, nl, off);
  gm_add<L, false> (g, acc, acc, x);
  gm_sub<L, false> (g, acc, acc, N2, N2);
  if (live)
#pragma unroll
    for (int t = 0; t < L; t++)
      if (off + t < nl)
        out[o * nl + off + t] = acc[t];
}

template<int L>
__global__ void
rt_mp_kernel (uint32_t tpi, uint64_t count, uint32_t half,
              const uint32_t * __restrict__ tv,
              const uint32_t * __restrict__ tree, uint32_t * __restrict__ out,
              uint32_t nl, const uint32_t * __restrict__ Ng,
              const uint32_t * __restrict__ R2g, uint32_t np0)
{
  gm_group g;
  g.init (tpi);
  rt_mp_body<L> (g, tpi, count, half, tv, tree, out, nl, Ng, R2g, np0);
}

/* the same with a wide group (64 or 128 threads per instance) */
template<int L>
__global__ void
rt_mp_wkernel (uint32_t tpi, uint64_t count, uint32_t half,
              const uint32_t * __restrict__ tv,
              const uint32_t * __restrict__ tree, uint32_t * __restrict__ out,
              uint32_t nl, const uint32_t * __restrict__ Ng,
              const uint32_t * __restrict__ R2g, uint32_t np0)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  rt_mp_body<L> (g, tpi, count, half, tv, tree, out, nl, Ng, R2g, np0);
}

/* ---- NTT backend (gpu_nmul.cuh) ----------------------------------------- */

/* For large N the curve arithmetic can run on transform products instead:
 * numbers are D signed digits (D words, contiguous; Montgomery form with
 * R = B^D - 1), one block of D/8 threads per instance, each block looping
 * over instances with its own scratch (NT_SCR words).  The same tables as
 * above in the same order, with D words per number; progression state
 * point t of instance i at numbers (i (S + 1) + t) 4.  Each curve's a, d,
 * C1, C2, C3 are also kept as hats (RH_*).  An addition is 98 transforms
 * per prime: the second point's coordinates and X2 + Y2 are transformed
 * once each, the five products with them take the first point's numbers
 * as digits; E, F, G, H are transformed once for the last four products.
 * In a step, addition k + 1's first point is addition k's second (not yet
 * updated), so its hats carry over: 88 transforms. */
enum { RH_A, RH_D, RH_C1, RH_C2, RH_C3, RH_COUNT };
#define NT_NUMS 16                 /* scratch numbers; 12 .. 15: a point */
#define NT_HATS 14                 /* two sets of 5 (X, Y, Z, T, X + Y), 4 */
#define NT_SCR(D) ((size_t) (NT_NUMS + 4 * NT_HATS) * (D))
/* threads per SM the register budget allows: 64 registers at 4096 */
#define NT_MINB(lg) ((lg) >= 12 ? 2 : 1024 / ((1 << (lg)) / 8) / 2)

template<int LG>
static __device__ __forceinline__ void
nt_copy_pt (int32_t *r, const int32_t *a)
{
  constexpr int D = 1 << LG;
#pragma unroll
  for (int k = 0; k < 4; k++)
    nm_copy<LG> (r + k * D, a + k * D);
}

/* R = P + Q (points of four numbers X, Y, Z, T), as rt_add; HC: the
   curve's hats; R may alias P or Q.  HQ (scratch hat set 0 or 1) gets
   the hats of Q's X, Y, Z, T, X + Y; HP, if not NULL, holds P's. */
template<int LG>
static __device__ __noinline__ void
nt_add (const nm_ctx &c, int32_t *R, const int32_t *P, const int32_t *Q,
        const uint32_t *hc, int32_t *s, int hq, int hp)
{
  constexpr int D = 1 << LG;
  int32_t *S1 = s, *S2 = s + D, *A = s + 2 * D, *B = s + 3 * D;
  int32_t *C = s + 4 * D, *Dd = s + 5 * D, *E = s + 6 * D, *F = s + 7 * D;
  int32_t *Gg = s + 8 * D, *H = s + 9 * D;
  uint32_t *h = (uint32_t *) (s + NT_NUMS * D);
  uint32_t *q = h + hq * 20 * D, *p = h + hp * 20 * D;
  uint32_t *hX = q, *hY = q + 4 * D, *hZ = q + 8 * D, *hT = q + 12 * D;
  uint32_t *hS = q + 16 * D, *hE = h + 40 * D, *hF = h + 44 * D;
  uint32_t *hG = h + 48 * D, *hH = h + 52 * D;
  nm_add<LG> (S2, Q, Q + D);
  nm_hat<LG> (c, hX, Q);
  nm_hat<LG> (c, hY, Q + D);
  nm_hat<LG> (c, hZ, Q + 2 * D);
  nm_hat<LG> (c, hT, Q + 3 * D);
  nm_hat<LG> (c, hS, S2);
  if (hp >= 0)
    {
      nm_mulhh<LG> (c, A, p, hX);
      nm_mulhh<LG> (c, B, p + 4 * D, hY);
      nm_mulhh<LG> (c, C, p + 12 * D, hT);
      nm_mulhh<LG> (c, Dd, p + 8 * D, hZ);
      nm_mulhh<LG> (c, E, p + 16 * D, hS);
    }
  else
    {
      nm_add<LG> (S1, P, P + D);
      nm_mul<LG> (c, A, P, hX);
      nm_mul<LG> (c, B, P + D, hY);
      nm_mul<LG> (c, C, P + 3 * D, hT);
      nm_mul<LG> (c, Dd, P + 2 * D, hZ);
      nm_mul<LG> (c, E, S1, hS);
    }
  nm_mul<LG> (c, C, C, hc + RH_D * 4 * D);        /* C = d T1 T2 */
  nm_sub<LG> (E, E, A);
  nm_sub<LG> (E, E, B);                           /* (X1+Y1)(X2+Y2)-A-B */
  nm_mul<LG> (c, A, A, hc + RH_A * 4 * D);
  nm_sub<LG> (H, B, A);                           /* H = B - a A */
  nm_add<LG> (Gg, Dd, C);                         /* G = D + C */
  nm_sub<LG> (F, Dd, C);                          /* F = D - C */
  nm_hat<LG> (c, hE, E);
  nm_hat<LG> (c, hF, F);
  nm_hat<LG> (c, hG, Gg);
  nm_hat<LG> (c, hH, H);
  nm_mulhh<LG> (c, R, hE, hF);
  nm_mulhh<LG> (c, R + D, hG, hH);
  nm_mulhh<LG> (c, R + 3 * D, hE, hH);
  nm_mulhh<LG> (c, R + 2 * D, hF, hG);
}

/* as rt_setup_body; identity digits zero skipped (the same point up to a
   projective factor, so the same roots) */
template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_setup_kernel (const __grid_constant__ nm_ctx c, int mode, uint32_t count,
                 const int32_t * __restrict__ cst,
                 const uint32_t * __restrict__ chat, int32_t * __restrict__ W,
                 uint32_t K, uint32_t w, uint32_t nw, uint32_t k0,
                 uint32_t k1, uint32_t E, uint32_t nr, uint32_t S,
                 const uint8_t * __restrict__ dig,
                 const uint8_t * __restrict__ sgn,
                 int32_t * __restrict__ state, const int32_t * __restrict__ idp,
                 int32_t * __restrict__ scr)
{
  constexpr size_t D = 1 << LG;
  int32_t *s = scr + blockIdx.x * NT_SCR (D), *acc = s + 12 * D;
  for (uint32_t i = blockIdx.x; i < count; i += gridDim.x)
    {
      const uint32_t cv = mode == RS_CHAIN ? i : mode == RS_FILL ? i / K : i / E;
      const int32_t *cc = cst + (size_t) cv * RC_COUNT * D;
      const uint32_t *hc = chat + (size_t) cv * RH_COUNT * 4 * D;
      const size_t wc = (size_t) cv * K * nw * 4;          /* numbers */
      if (mode == RS_CHAIN)
        {
          uint32_t kstart;
          if (k0 == 0)
            {
              nt_copy_pt<LG> (acc, cc + RC_PX * D);
              nt_copy_pt<LG> (W + wc * D, acc);
              kstart = 1;
            }
          else
            {
              nt_copy_pt<LG> (acc, W + (wc + (size_t) (k0 - 1) * nw * 4) * D);
              kstart = k0;
            }
          const uint32_t iters = k1 > kstart ? (k1 - kstart) * w : 0;
          for (uint32_t it = 0; it < iters; it++)
            {
              nt_add<LG> (c, acc, acc, acc, hc, s, 0, -1);
              if ((it + 1) % w == 0)
                nt_copy_pt<LG> (W + (wc + (size_t) (kstart + it / w) * nw * 4) * D,
                                acc);
            }
        }
      else if (mode == RS_FILL)
        {
          const int32_t *base = W + (size_t) i * nw * 4 * D;
          for (uint32_t it = 0; it + 1 < nw; it++)
            nt_add<LG> (c, W + ((size_t) i * nw + it + 1) * 4 * D,
                        W + ((size_t) i * nw + it) * 4 * D, base, hc, s, 0,
                        -1);
        }
      else
        {
          const uint32_t e = i % E;
          const bool shared = e == nr * S;
          const uint32_t r = shared ? 0 : e / S, pt = shared ? S : e % S;
          int32_t *st = state + (((size_t) (cv * nr + r) * (S + 1) + pt) * 4) * D;
          if (k0 == 0)
            nt_copy_pt<LG> (st, idp);
          for (uint32_t k = k0; k < k1; k++)
            {
              const uint32_t d = dig[(size_t) e * K + k];
              if (d != 0)
                nt_add<LG> (c, st, st,
                            W + (wc + ((size_t) k * nw + d - 1) * 4) * D, hc,
                            s, 0, -1);
            }
          if (k1 == K)
            {
              if (sgn[e])
                {
                  nm_neg<LG> (st, st);
                  nm_neg<LG> (st + 3 * D, st + 3 * D);
                }
              if (shared)
                for (uint32_t rr = 1; rr < nr; rr++)
                  nt_copy_pt<LG> (state + (((size_t) (cv * nr + rr) * (S + 1) + S) * 4) * D,
                                  st);
            }
        }
    }
}

/* as rt_step_body */
template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_step_kernel (const __grid_constant__ nm_ctx c, uint32_t ninst, uint32_t nr,
                uint32_t S, int32_t * __restrict__ state,
                const uint32_t * __restrict__ chat,
                const int32_t * __restrict__ slots, uint64_t base,
                uint32_t dF, uint32_t it0, uint32_t it1,
                int32_t * __restrict__ num, int32_t * __restrict__ den,
                int32_t * __restrict__ scr)
{
  constexpr size_t D = 1 << LG;
  int32_t *s = scr + blockIdx.x * NT_SCR (D);
  int32_t *u = s + 10 * D, *v = s + 11 * D, *dump = s + 12 * D;
  for (uint32_t inst = blockIdx.x; inst < ninst; inst += gridDim.x)
    {
      const uint32_t cv = inst / nr, r = inst % nr;
      const uint32_t *hc = chat + (size_t) cv * RH_COUNT * 4 * D;
      int32_t *st = state + (size_t) inst * (S + 1) * 4 * D;
      int64_t nlo = 0, nhi = 0;
      if (slots == NULL)
        {
          int64_t lo = (int64_t) base - r, hi = (int64_t) base + dF - r;
          nlo = lo > 0 ? (lo + nr - 1) / nr : 0;
          nhi = hi > 0 ? (hi + nr - 1) / nr : 0;
        }
      for (uint32_t it = it0; it < it1; it++)
        {
          bool act;
          int64_t slot;
          if (slots != NULL)
            {
              act = true;
              slot = slots[(size_t) it * nr + r];
            }
          else
            {
              int64_t nn = nlo + it;
              act = nn < nhi;
              slot = (int64_t) r + nn * nr - (int64_t) base;
            }
          if (!act)
            break;                /* no more steps for this instance */
          const bool emit = slot >= 0 && slot < dF;
          const size_t o = ((size_t) cv * dF + (emit ? slot : 0)) * D;
          /* numerator (3 + A) Z + (3 - A) Y, denominator 3g (Z - Y) */
          if (emit)
            {
              nm_mul<LG> (c, u, st + 2 * D, hc + RH_C1 * 4 * D);
              nm_mul<LG> (c, v, st + D, hc + RH_C2 * 4 * D);
              nm_add<LG> (num + o, u, v);
              nm_sub<LG> (u, st + 2 * D, st + D);
              nm_mul<LG> (c, den + o, u, hc + RH_C3 * 4 * D);
            }
          for (uint32_t k = 0; k < S; k++)
            nt_add<LG> (c, st + (size_t) k * 4 * D, st + (size_t) k * 4 * D,
                        st + (size_t) (k + 1) * 4 * D, hc, s, k & 1,
                        k > 0 ? (int) ((k - 1) & 1) : -1);
        }
      (void) dump;
    }
}

/* as rt_vec_body (OUT: D words per number for RV_UP, NL limbs for
   RV_FINAL) */
template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_vec_kernel (const __grid_constant__ nm_ctx c, int mode, uint64_t count,
               const int32_t * __restrict__ a, uint64_t na,
               int32_t * __restrict__ b, uint32_t * __restrict__ out,
               uint32_t nl, int32_t * __restrict__ scr)
{
  constexpr size_t D = 1 << LG;
  int32_t *s = scr + blockIdx.x * NT_SCR (D);
  uint32_t *h = (uint32_t *) (s + NT_NUMS * D);
  for (uint64_t i = blockIdx.x; i < count; i += gridDim.x)
    {
      if (mode == RV_UP)
        {
          int32_t *o = (int32_t *) out + i * D;
          if (2 * i + 1 < na)
            {
              nm_hat<LG> (c, h, a + (2 * i + 1) * D);
              nm_mul<LG> (c, o, a + 2 * i * D, h);
            }
          else
            nm_copy<LG> (o, a + 2 * i * D);
        }
      else if (mode == RV_DOWN)
        {
          nm_hat<LG> (c, h, a + i * D);
          if (2 * i + 1 < na)
            nm_mul<LG> (c, s, b + (2 * i + 1) * D, h);
          else
            nm_copy<LG> (s, a + i * D);
          nm_mul<LG> (c, s + D, b + 2 * i * D, h);
          nm_copy<LG> (b + 2 * i * D, s);
          if (2 * i + 1 < na)
            nm_copy<LG> (b + (2 * i + 1) * D, s + D);
        }
      else
        {
          nm_hat<LG> (c, h, b + i * D);
          nm_mul<LG> (c, s, a + i * D, h);
          nm_mul<LG> (c, s, s, c.hone);             /* out of Montgomery form */
          int32_t x[8];
          nm_load<LG> (x, s);
          nm_to_limbs<LG> (c, out + i * nl, nl, x);
        }
    }
}

/* The Z/N tree and transposed-evaluation levels (as rt_tree_body and
   rt_mp_body) for degrees up to NT_ZMAX: per node, the hats of its
   numbers, the b's (tree) or children (mp) put in Montgomery form first
   (times R^2 / R), so that each output coefficient is one sum of
   pointwise products in the transform domain and one reduction:
   about 12 (tree) or 18 (mp) transforms per coefficient whatever D.
   Scratch per block: NT_ZSCR words. */
#define NT_ZMAX 16
#define NT_ZSCR(D) ((size_t) (4 * 4 * NT_ZMAX + 3) * (D))

template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_tree_kernel (const __grid_constant__ nm_ctx c, uint64_t count, uint32_t d,
                const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                uint32_t nl, int32_t * __restrict__ scr)
{
  constexpr size_t D = 1 << LG;
  int32_t *x = scr + blockIdx.x * NT_ZSCR (D), *y = x + D, *z = y + D;
  uint32_t *ha = (uint32_t *) (z + D), *hb = ha + (size_t) d * 4 * D;
  const uint64_t pairs = count / (2 * d);
  for (uint64_t pr = blockIdx.x; pr < pairs; pr += gridDim.x)
    {
      const uint32_t *A = in + pr * 2 * d * nl, *Bp = A + (size_t) d * nl;
      for (uint32_t i = 0; i < d; i++)
        {
          nm_from_limbs<LG> (c, x, A + (size_t) i * nl, nl);
          nm_hat<LG> (c, ha + (size_t) i * 4 * D, x);
          nm_from_limbs<LG> (c, x, Bp + (size_t) i * nl, nl);
          nm_mul<LG> (c, x, x, c.hR2);
          nm_hat<LG> (c, hb + (size_t) i * 4 * D, x);
        }
      for (uint32_t k = 0; k < 2 * d; k++)
        {
          uint32_t acc[2][2][8];
          nm_acc_clear<LG> (acc);
          for (uint32_t i = k >= d ? k - d + 1 : 0; i < d && i <= k; i++)
            nm_acc_add<LG> (c, acc, ha + (size_t) i * 4 * D,
                            hb + (size_t) (k - i) * 4 * D);
          nm_acc_redc<LG> (c, z, acc);
          /* + a_(k-d) + b_(k-d) */
          if (k >= d)
            {
              nm_from_limbs<LG> (c, x, A + (size_t) (k - d) * nl, nl);
              nm_from_limbs<LG> (c, y, Bp + (size_t) (k - d) * nl, nl);
              nm_add<LG> (z, z, x);
              nm_add<LG> (z, z, y);
            }
          int32_t v[8];
          nm_load<LG> (v, z);
          nm_to_limbs<LG> (c, out + (pr * 2 * d + k) * nl, nl, v);
        }
    }
}

template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_mp_kernel (const __grid_constant__ nm_ctx c, uint64_t count, uint32_t half,
              const uint32_t * __restrict__ tv,
              const uint32_t * __restrict__ tree, uint32_t * __restrict__ out,
              uint32_t nl, int32_t * __restrict__ scr)
{
  constexpr size_t D = 1 << LG;
  int32_t *x = scr + blockIdx.x * NT_ZSCR (D), *z = x + 2 * D;
  /* hats: the 2 half - 1 values used, then the children L, R */
  uint32_t *ht = (uint32_t *) (z + D), *hl = ht + (size_t) 2 * half * 4 * D;
  uint32_t *hr = hl + (size_t) half * 4 * D;
  const uint64_t k = 2 * (uint64_t) half, nodes = count / k;
  for (uint64_t nd = blockIdx.x; nd < nodes; nd += gridDim.x)
    {
      const uint32_t *T = tv + nd * k * nl, *Lc = tree + nd * k * nl;
      const uint32_t *Rc = Lc + (size_t) half * nl;
      for (uint32_t m = 0; m + 1 < k; m++)
        {
          nm_from_limbs<LG> (c, x, T + (size_t) m * nl, nl);
          nm_hat<LG> (c, ht + (size_t) m * 4 * D, x);
        }
      for (uint32_t j = 0; j < half; j++)
        {
          nm_from_limbs<LG> (c, x, Lc + (size_t) j * nl, nl);
          nm_mul<LG> (c, x, x, c.hR2);
          nm_hat<LG> (c, hl + (size_t) j * 4 * D, x);
          nm_from_limbs<LG> (c, x, Rc + (size_t) j * nl, nl);
          nm_mul<LG> (c, x, x, c.hR2);
          nm_hat<LG> (c, hr + (size_t) j * 4 * D, x);
        }
      for (uint32_t r = 0; r < k; r++)
        {
          const uint32_t i = r % half;
          const uint32_t *hc = r < half ? hr : hl;
          uint32_t acc[2][2][8];
          nm_acc_clear<LG> (acc);
          for (uint32_t j = 0; j < half; j++)
            nm_acc_add<LG> (c, acc, hc + (size_t) j * 4 * D,
                            ht + (size_t) (i + j) * 4 * D);
          nm_acc_redc<LG> (c, z, acc);
          nm_from_limbs<LG> (c, x, T + (size_t) (half + i) * nl, nl);
          nm_add<LG> (z, z, x);
          int32_t v[8];
          nm_load<LG> (v, z);
          nm_to_limbs<LG> (c, out + (nd * k + r) * nl, nl, v);
        }
    }
}

/* the hats of each curve's a, d, C1, C2, C3 (numbers RC_A .. RC_C3) */
template<int LG>
__global__ void __launch_bounds__ ((1 << LG) / 8, NT_MINB (LG))
nt_chat_kernel (const __grid_constant__ nm_ctx c, uint32_t count,
                const int32_t * __restrict__ cst, uint32_t * __restrict__ chat)
{
  constexpr size_t D = 1 << LG;
  for (uint32_t i = blockIdx.x; i < count; i += gridDim.x)
    {
      const uint32_t cv = i / RH_COUNT, k = i % RH_COUNT;
      nm_hat<LG> (c, chat + ((size_t) cv * RH_COUNT + k) * 4 * D,
                  cst + ((size_t) cv * RC_COUNT + k) * D);
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
typedef void (*rt_tree_fn) (uint32_t, uint64_t, uint32_t, const uint32_t *,
                            uint32_t *, uint32_t, const uint32_t *,
                            const uint32_t *, uint32_t);
typedef void (*rt_mp_fn) (uint32_t, uint64_t, uint32_t, const uint32_t *,
                          const uint32_t *, uint32_t *, uint32_t,
                          const uint32_t *, const uint32_t *, uint32_t);

/* one shape's kernels, for groups of TPI <= 32 or for wide groups */
struct rt_kset {
  rt_setup_fn setup;
  rt_step_fn step;
  rt_vec_fn vec;
  rt_tree_fn tree;
  rt_mp_fn mp;
};

struct rt_kernels {
  rt_kset n, w;       /* narrow (TPI 2 to 32), wide (TPI 64, 128) */
};

/* The limbs per thread with kernels (few, as each shape takes a while to
   compile; wide groups only for L >= 6; L = 17 so that N just past 2^16
   bits, 2049 limbs, gets 128 x 17 rather than 128 x 24).
   The shape for n limbs is the one with the fewest limbs TPI L >= n
   (products cost about (TPI L)^2), then the fewest threads per curve (fewer
   shuffles or barriers), within a register budget: L <= 12 for the roots
   (at 1024 bits 2 x 17 needs 255 registers and spills in the step kernel,
   1.6 times slower than 4 x 9 on an A10), L <= 17 for the Z/N tree and
   middle product kernels (2 x 17 there is faster than 4 x 9).  Past the
   budget (N above 32 x 12 or 128 x 12 limbs), any L: for the roots first
   within a warp (rt_shape). */
static const uint32_t rt_Ls[] = { 1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 24 };

/* ZN: only the Z/N kernels (tree and mp) */
template<int L, bool ZN = false>
static void
rt_put (rt_kernels *t)
{
  if constexpr (L <= GPU_ROOTS_MAX_L)
    {
      t[L].n.tree = rt_tree_kernel<L>;
      t[L].n.mp = rt_mp_kernel<L>;
      if constexpr (!ZN)
        {
          t[L].n.setup = rt_setup_kernel<L>;
          t[L].n.step = rt_step_kernel<L>;
          t[L].n.vec = rt_vec_kernel<L>;
        }
      if constexpr (L >= 6)
        {
          t[L].w.tree = rt_tree_wkernel<L>;
          t[L].w.mp = rt_mp_wkernel<L>;
          if constexpr (!ZN)
            {
              t[L].w.setup = rt_setup_wkernel<L>;
              t[L].w.step = rt_step_wkernel<L>;
              t[L].w.vec = rt_vec_wkernel<L>;
            }
        }
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
      rt_put<9> (t);
      rt_put<12> (t);
      rt_put<16> (t);
      rt_put<17> (t);
      rt_put<24> (t);
      init = true;
    }
  return L >= 1 && L <= 24 && t[L].n.tree != NULL ? &t[L] : NULL;
}

/* the kernels for TPI threads of L limbs, or NULL; ZN: the Z/N ones */
static const rt_kset *
rt_kset_for (uint32_t tpi, uint32_t L, bool zn)
{
  const rt_kernels *k = rt_get_kernels (L);
  if (k == NULL || tpi < 2 || tpi > 128 || (tpi & (tpi - 1)) != 0)
    return NULL;
  const rt_kset *ks = tpi > 32 ? &k->w : &k->n;
  return (zn ? ks->tree != NULL : ks->setup != NULL) ? ks : NULL;
}

/* Passes: for the roots, first within a warp and the budget; then the
   fewest limbs within a warp (any L) or within the budget (any group),
   ties going to fewer threads; then anything.  The second pass takes
   32 x 16 over 64 x 8 (385 to 512 limbs, from 12284 bits), 32 x 17 over
   64 x 9 (513 to 544) and 32 x 24 over 64 x 12 (577 to 768), which make
   stage 2 13-31% faster on an A10 (shared memory between the warps of a
   wide group costs more than limbs past the budget), but not 32 x 24 over
   64 x 9 (545 to 576 limbs).  The Z/N kernels: within the budget, then
   anything. */
static bool
rt_shape (uint32_t nlimbs, uint32_t *tpi, uint32_t *L, bool zn = false)
{
  const uint32_t lmax = zn ? 17 : 12;
  for (int pass = zn ? 1 : 0; pass < 3; pass++)
    {
      uint32_t best = 0;
      for (uint32_t t = 2; t <= (pass == 0 ? 32u : 128u); t *= 2)
        for (uint32_t l : rt_Ls)
          if (l * t >= nlimbs
              && (pass == 2 || l <= lmax || (pass == 1 && !zn && t <= 32))
              && rt_kset_for (t, l, zn) != NULL
              && (best == 0 || l * t < best))
            {
              best = l * t;
              *tpi = t;
              *L = l;
            }
      if (best != 0)
        return true;
    }
  return false;
}

/* ---- host side --------------------------------------------------------- */

/* GMPECM_ROOTS_PROF set: time each kind of launch (synchronising),
   printed by gpu_roots_destroy */
enum { RP_CHAIN, RP_FILL, RP_SCALAR, RP_STEP, RP_UP, RP_DOWN, RP_FINAL,
       RP_HOST, RP_N };
static const char *rp_name[RP_N] = { "chain", "fill", "scalar", "step",
                                     "inverse up", "inverse down", "final",
                                     "host" };
static struct {
  int on = -1;
  double ms[RP_N] = { 0 };
  long calls[RP_N] = { 0 };
  cudaEvent_t a, b;
} rtp;

static void
rtp_begin (cudaStream_t st)
{
  if (rtp.on < 0)
    {
      rtp.on = getenv ("GMPECM_ROOTS_PROF") != NULL;
      if (rtp.on)
        {
          cudaEventCreate (&rtp.a);
          cudaEventCreate (&rtp.b);
        }
    }
  if (rtp.on)
    cudaEventRecord (rtp.a, st);
}

static void
rtp_end (cudaStream_t st, int cat)
{
  if (!rtp.on)
    return;
  float ms;
  cudaEventRecord (rtp.b, st);
  cudaEventSynchronize (rtp.b);
  cudaEventElapsedTime (&ms, rtp.a, rtp.b);
  rtp.ms[cat] += ms;
  rtp.calls[cat]++;
}

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
  const rt_kset *k;
  uint32_t K, w, nw;      /* digits of the coefficients, window */
  uint32_t *d_N, *d_one, *d_R2, *d_cst, *d_W;
  /* the Z/N kernels' own shape: ZTPI threads of ZL limbs, ZN limbs, N and
     R^2 mod N (R = 2^(32 ZN)) on ZN limbs */
  const rt_kset *zk;
  uint32_t ztpi, zL, zn;
  uint32_t *d_zN, *d_zR2;
  uint32_t *d_num, *d_den, *d_tree;
  rt_set F, G;
  uint64_t gblock;        /* the next block of G */
  cudaStream_t stream;    /* for everything but gpu_roots_tree, _mp */
  struct rt_ntt *nt;      /* the NTT backend for the curve arithmetic, or
                             NULL (then n is TPI L limbs; else n = D) */
  struct rt_ntt *zt;      /* the NTT tables for the Z/N levels, or NULL
                             (may be nt) */
};

/* the NTT backend's tables and scratch */
struct rt_ntt {
  nm_host h;
  uint32_t D, TPG, slots;   /* slots: blocks per launch (resident ones) */
  size_t smem;
  uint32_t *d_chat;         /* per curve the hats RH_* */
  int32_t *d_id;            /* the identity point */
  int32_t *d_scr;           /* slots NT_SCR (D) words */
  uint32_t zslots;          /* blocks of the Z/N tree and mp kernels */
  int32_t *d_zscr;          /* their scratch (zslots NT_ZSCR (D)), or NULL
                               until first used */
};

/* launch KERN<lg> on min (COUNT, CAP) blocks (each looping over its
   share of the COUNT units) on stream ST */
#define NT_RUNC(NT, KERN, COUNT, CAP, ST, ...) do {                          \
    const rt_ntt *nt_ = (NT);                                                 \
    const uint32_t g_ = (uint32_t) std::min<uint64_t> ((COUNT), (CAP));       \
    if (g_ > 0)                                                               \
      switch (nt_->h.c.lg)                                                    \
        {                                                                     \
        case 8: KERN<8><<<g_, 32, nt_->smem, (ST)>>> (nt_->h.c, __VA_ARGS__); break; \
        case 9: KERN<9><<<g_, 64, nt_->smem, (ST)>>> (nt_->h.c, __VA_ARGS__); break; \
        case 10: KERN<10><<<g_, 128, nt_->smem, (ST)>>> (nt_->h.c, __VA_ARGS__); break; \
        case 11: KERN<11><<<g_, 256, nt_->smem, (ST)>>> (nt_->h.c, __VA_ARGS__); break; \
        case 12: KERN<12><<<g_, 512, nt_->smem, (ST)>>> (nt_->h.c, __VA_ARGS__); break; \
        }                                                                     \
    CUDA_CHECK (cudaGetLastError ());                                         \
  } while (0)
#define NT_RUN(G, KERN, COUNT, ...)                                           \
  NT_RUNC ((G)->nt, KERN, COUNT, (G)->nt->slots, (G)->stream, __VA_ARGS__)

template<int LG>
static uint32_t
nt_slots (size_t smem)
{
  int dev, sm, b = 0;
  CUDA_CHECK (cudaGetDevice (&dev));
  CUDA_CHECK (cudaDeviceGetAttribute (&sm, cudaDevAttrMultiProcessorCount,
                                      dev));
  CUDA_CHECK (cudaOccupancyMaxActiveBlocksPerMultiprocessor (
      &b, nt_step_kernel<LG>, (1 << LG) / 8, smem));
  return (uint32_t) std::max (1, b) * sm;
}

/* The NTT tables for N with a transform length of at least 2^LGMIN, or
   NULL (N too large, or sharing a factor with B^D - 1 for every b) */
static rt_ntt *
rt_ntt_create (const mpz_t N, int lgmin)
{
  int lg, b;
  if (!nm_shape (mpz_sizeinbase (N, 2), lgmin, NM_MAX_LG, &lg, &b))
    return NULL;
  rt_ntt *nt = new rt_ntt;
  bool ok = false;
  for (; b <= NM_BMAX (lg) && !ok; b++)
    ok = nm_ctx_create (nt->h, N, lg, b);
  if (!ok)
    {
      delete nt;
      return NULL;
    }
  nt->D = 1u << lg;
  nt->TPG = nt->D / 8;
  nt->smem = nm_smem_bytes (lg);
  switch (lg)
    {
    case 8: nt->slots = nt_slots<8> (nt->smem); break;
    case 9: nt->slots = nt_slots<9> (nt->smem); break;
    case 10: nt->slots = nt_slots<10> (nt->smem); break;
    case 11: nt->slots = nt_slots<11> (nt->smem); break;
    default: nt->slots = nt_slots<12> (nt->smem); break;
    }
  nt->d_chat = NULL;
  /* the identity (0 : 1 : 1 : 0): R mod N for 1 */
  std::vector<int32_t> id ((size_t) 4 * nt->D, 0), one;
  nm_digits (one, nt->h.R, nt->D, nt->h.c.b);
  std::copy (one.begin (), one.end (), id.begin () + nt->D);
  std::copy (one.begin (), one.end (), id.begin () + 2 * nt->D);
  CUDA_CHECK (cudaMalloc ((void **) &nt->d_id, id.size () * 4));
  CUDA_CHECK (cudaMemcpy (nt->d_id, id.data (), id.size () * 4,
                          cudaMemcpyHostToDevice));
  CUDA_CHECK (cudaMalloc ((void **) &nt->d_scr,
                          (size_t) nt->slots * NT_SCR (nt->D) * 4));
  nt->zslots = std::max (1u, nt->slots / 2);
  nt->d_zscr = NULL;
  return nt;
}

static void rt_ntt_zscr (rt_ntt *nt);

static void
rt_ntt_destroy (rt_ntt *nt)
{
  if (nt == NULL)
    return;
  if (nt->d_chat)
    CUDA_CHECK (cudaFree (nt->d_chat));
  CUDA_CHECK (cudaFree (nt->d_id));
  CUDA_CHECK (cudaFree (nt->d_scr));
  if (nt->d_zscr)
    CUDA_CHECK (cudaFree (nt->d_zscr));
  nm_ctx_free (nt->h);
  delete nt;
}

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

/* Progressions per curve wanted for ROOTS roots.  The steps run at the
   arithmetic ceiling of the shape only with enough threads (fewer and they
   are latency-bound; more cost setup: about 256 per SM was best on an A10
   at 1024 bits), so: enough
   instances for that, but at most ROOTS / K, so that the setup (S K
   additions per progression, K digits) stays below the steps (S per
   root).  GMPECM_ROOTS_NR sets it, GMPECM_ROOTS_FILL the threads per SM. */
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
  env = getenv ("GMPECM_ROOTS_FILL");
  const double per_sm = env != NULL && atoi (env) > 0 ? atoi (env) : 256;
  double fill = sm * per_sm / ((double) G->tpi * G->C);
  if (G->nt != NULL)        /* one block per instance: twice the slots */
    fill = 2.0 * G->nt->slots / G->C;
  double bal = roots / std::max (1.0, kbits / 4);
  double nr = std::min (fill, bal);
  return nr < 1 ? 1 : (uint32_t) std::min (nr, 65536.0);
}

size_t
gpu_roots_max_bits (void)
{
  /* the largest TPI L with both the roots' and the Z/N kernels */
  size_t nmax = 0;
  for (uint32_t t = 2; t <= 128; t *= 2)
    for (uint32_t l : rt_Ls)
      if (rt_kset_for (t, l, false) != NULL && rt_kset_for (t, l, true) != NULL)
        nmax = std::max (nmax, (size_t) t * l);
  return 32 * nmax - GPU_SPARE_BITS;
}

size_t
gpu_roots_words (const mpz_t N)
{
  const size_t bits = mpz_sizeinbase (N, 2);
  size_t w = (bits + GPU_SPARE_BITS + 31) / 32 + 32;
  int lg, b;
  if (bits >= 12000 && nm_shape (bits, NM_MIN_LG, NM_MAX_LG, &lg, &b))
    w = std::max (w, (size_t) 1 << lg);
  return w;
}

size_t
gpu_roots_scratch_bytes (const mpz_t N)
{
  const size_t bits = mpz_sizeinbase (N, 2);
  int lg, b;
  if (bits < 12000 || !nm_shape (bits, NM_MIN_LG, NM_MAX_LG, &lg, &b))
    return 0;
  int dev, sm;
  CUDA_CHECK (cudaGetDevice (&dev));
  CUDA_CHECK (cudaDeviceGetAttribute (&sm, cudaDevAttrMultiProcessorCount,
                                      dev));
  /* at most 1024 threads of D / 8 per SM, half of them for the Z/N
     levels */
  return (size_t) sm * (8192 >> lg)
         * (NT_SCR ((size_t) 1 << lg) + NT_ZSCR ((size_t) 1 << lg) / 2) * 4;
}

static void
rt_set_alloc (gpu_roots *G, rt_set &s)
{
  s.E = s.nr * std::abs (G->S) + 1;
  s.pinst = G->nt != NULL ? G->C * s.nr
            : (uint32_t) rt_padded (G, (uint64_t) G->C * s.nr);
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

/* seconds per unit of KERNEL work, best of two runs after a warm-up */
template<class F>
static double
rt_time (F launch, double units)
{
  cudaEvent_t e0, e1;
  CUDA_CHECK (cudaEventCreate (&e0));
  CUDA_CHECK (cudaEventCreate (&e1));
  double best = 1e30;
  for (int r = 0; r < 3; r++)
    {
      CUDA_CHECK (cudaEventRecord (e0, 0));
      launch ();
      CUDA_CHECK (cudaEventRecord (e1, 0));
      CUDA_CHECK (cudaEventSynchronize (e1));
      float ms;
      CUDA_CHECK (cudaEventElapsedTime (&ms, e0, e1));
      if (r > 0)
        best = std::min (best, ms * 1e-3 / units);
    }
  CUDA_CHECK (cudaEventDestroy (e0));
  CUDA_CHECK (cudaEventDestroy (e1));
  return best;
}

/* seconds per product of the vec kernel's RV_UP (generic products: the
   NTT transforms both operands) on zeros (no data-dependent work), on
   the roots' schoolbook shape of G or, with NT, the NTT */
static double
rt_time_up (gpu_roots *G, rt_ntt *nt)
{
  const uint64_t count = 1024;
  const size_t words = nt != NULL ? nt->D : G->n;
  uint32_t *a, *o;
  CUDA_CHECK (cudaMalloc ((void **) &a, 2 * count * words * 4));
  CUDA_CHECK (cudaMalloc ((void **) &o, count * words * 4));
  CUDA_CHECK (cudaMemset (a, 0, 2 * count * words * 4));
  const double t = rt_time ([&] {
      if (nt != NULL)
        NT_RUNC (nt, nt_vec_kernel, count, nt->slots, (cudaStream_t) 0,
                 RV_UP, count, (const int32_t *) a, 2 * count,
                 (int32_t *) NULL, o, 0u, nt->d_scr);
      else
        {
          G->k->vec<<<rt_blocks (G, count), GPU_TPB>>> (
              G->tpi, RV_UP, count, a, 2 * count, NULL, o, 0, G->d_N,
              G->d_one, G->np0);
          CUDA_CHECK (cudaGetLastError ());
        }
    }, (double) count);
  CUDA_CHECK (cudaFree (a));
  CUDA_CHECK (cudaFree (o));
  return t;
}

/* seconds per output coefficient of a Z/N tree level of degree 8 (zeros),
   with the Z/N shape of G or, with NT, the NTT */
static double
rt_time_tree (gpu_roots *G, rt_ntt *nt)
{
  const uint32_t d = 8;
  const uint64_t count = (uint64_t) 2 * d * (nt != NULL ? nt->zslots : 32);
  uint32_t *a, *o;
  CUDA_CHECK (cudaMalloc ((void **) &a, count * G->nl * 4));
  CUDA_CHECK (cudaMalloc ((void **) &o, count * G->nl * 4));
  CUDA_CHECK (cudaMemset (a, 0, count * G->nl * 4));
  if (nt != NULL)
    rt_ntt_zscr (nt);
  const double t = rt_time ([&] {
      if (nt != NULL)
        NT_RUNC (nt, nt_tree_kernel, count / (2 * d), nt->zslots,
                 (cudaStream_t) 0, count, d, a, o, (uint32_t) G->nl,
                 nt->d_zscr);
      else
        {
          const uint64_t per = GPU_TPB / G->ztpi;
          G->zk->tree<<<(unsigned) ((count + per - 1) / per), GPU_TPB>>> (
              G->ztpi, count, d, a, o, G->nl, G->d_zN, G->d_zR2, G->np0);
          CUDA_CHECK (cudaGetLastError ());
        }
    }, (double) count);
  CUDA_CHECK (cudaFree (a));
  CUDA_CHECK (cudaFree (o));
  return t;
}

/* (limbs, NTT for the roots, NTT for the Z/N levels) for the sizes timed
   so far */
struct rt_choice { uint32_t need; bool roots, zn; };
static std::vector<rt_choice> rt_choices;

/* The NTT tables for G's N into G->nt (the roots' curve arithmetic) and
   G->zt (the Z/N tree and evaluation levels), each or both NULL for the
   schoolbook kernels.  GMPECM_ROOTS_NTT=1 forces the NTT for both (at any
   size it takes), =0 rules it out; GMPECM_ROOTS_ZNTT does the same for
   the Z/N levels alone.  Otherwise, from 12000 bits on (the roots only
   when no shape was forced), each part is timed both ways once per size
   (printed with GMPECM_ROOTS_PROF): the roots on generic products, where
   the NTT may be up to 1.3 times slower (an addition reuses transforms,
   the step kernels keep more state than these products: 1.4 to 1.8 times
   faster in the roots than timed at 26000 bits on an A10), the Z/N levels
   on a level of degree 8. */
static void
rt_pick_ntt (gpu_roots *G, const mpz_t N, uint32_t need, bool forced)
{
  const char *er = getenv ("GMPECM_ROOTS_NTT"), *ez = getenv ("GMPECM_ROOTS_ZNTT");
  const int fr = er != NULL && *er != '\0' ? atoi (er) : -1;
  const int fz = ez != NULL && *ez != '\0' ? atoi (ez) : fr;
  const bool big = mpz_sizeinbase (N, 2) >= 12000;
  bool want_r = fr == 1 || (fr < 0 && big && !forced);
  bool want_z = fz == 1 || (fz < 0 && big);
  G->nt = G->zt = NULL;
  if (!want_r && !want_z)
    return;
  rt_ntt *nt = rt_ntt_create (N, NM_MIN_LG);
  if (nt == NULL)
    return;
  if ((want_r && fr < 0) || (want_z && fz < 0))
    {
      const rt_choice *ch = NULL;
      for (const rt_choice &c : rt_choices)
        if (c.need == need)
          ch = &c;
      rt_choice c = { need, true, true };
      if (ch != NULL)
        c = *ch;
      else
        {
          const double rn = rt_time_up (G, nt), rs = rt_time_up (G, NULL);
          const double zn = rt_time_tree (G, nt), zs = rt_time_tree (G, NULL);
          c.roots = rn < 1.3 * rs;
          c.zn = zn < zs;
          rt_choices.push_back (c);
          if (G->verbose || getenv ("GMPECM_ROOTS_PROF") != NULL)
            printf ("GPU roots: NTT d = %u, b = %d against schoolbook: "
                    "products %.2f us (%u x %u: %.2f), Z/N level %.2f us "
                    "(%u x %u: %.2f): roots %s, Z/N %s\n", nt->D, nt->h.c.b,
                    rn * 1e6, G->tpi, G->L, rs * 1e6, zn * 1e6, G->ztpi,
                    G->zL, zs * 1e6, c.roots ? "NTT" : "schoolbook",
                    c.zn ? "NTT" : "schoolbook");
        }
      if (fr < 0)
        want_r = want_r && c.roots;
      if (fz < 0)
        want_z = want_z && c.zn;
    }
  if (!want_r && !want_z)
    {
      rt_ntt_destroy (nt);
      return;
    }
  G->nt = want_r ? nt : NULL;
  G->zt = want_z ? nt : NULL;
  /* the Z/N levels' scratch now, not at their first use: by then the
     stage 2 driver has given the free memory to the polynomial work */
  if (want_z)
    rt_ntt_zscr (nt);
}

gpu_roots *
gpu_roots_create (const mpz_t N, uint32_t tpi, uint32_t L, int S,
                  unsigned long d1, unsigned long d2, const mpz_t i0,
                  size_t dF, unsigned long blocks, int nl, uint32_t curves,
                  int verbose)
{
  const uint32_t need = (mpz_sizeinbase (N, 2) + GPU_SPARE_BITS + 31) / 32;
  const bool forced = tpi != 0 || getenv ("GMPECM_ROOTS_SHAPE") != NULL;
  if (tpi == 0 && getenv ("GMPECM_ROOTS_SHAPE") != NULL)
    sscanf (getenv ("GMPECM_ROOTS_SHAPE"), "%u,%u", &tpi, &L);
  if (tpi == 0 && !rt_shape (need, &tpi, &L))
    return NULL;
  const rt_kset *k = rt_kset_for (tpi, L, false);
  if (k == NULL
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
  G->stream = 0;
  G->nt = G->zt = NULL;
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
  gpu_to_limbs (h.data (), G->n, G->R2);
  G->d_R2 = rt_upload (h);
  {
    uint32_t zt = tpi, zl = L;
    const char *zs = getenv ("GMPECM_ROOTS_ZSHAPE");   /* development */
    if (zs == NULL || sscanf (zs, "%u,%u", &zt, &zl) != 2
        || rt_kset_for (zt, zl, true) == NULL || zt * zl < need)
      rt_shape (need, &zt, &zl, true);
    G->zk = rt_kset_for (zt, zl, true);
    G->ztpi = zt;
    G->zL = zl;
    G->zn = zt * zl;
    std::vector<uint32_t> hz (G->zn);
    gpu_to_limbs (hz.data (), G->zn, N);
    G->d_zN = rt_upload (hz);
    mpz_t r;
    mpz_init (r);
    mpz_setbit (r, 32 * (size_t) G->zn);
    mpz_mod (r, r, N);
    mpz_mul (r, r, r);
    mpz_mod (r, r, N);
    gpu_to_limbs (hz.data (), G->zn, r);
    G->d_zR2 = rt_upload (hz);
    mpz_clear (r);
  }

  /* the NTT backend: numbers of D words, R = B^D - 1 */
  rt_pick_ntt (G, N, need, forced);
  if (G->nt != NULL)
    {
      G->n = G->nt->D;
      mpz_set (G->R, G->nt->h.R);
      mpz_set (G->R2, G->nt->h.R2);
      CUDA_CHECK (cudaMalloc ((void **) &G->nt->d_chat,
                              (size_t) curves * RH_COUNT * 4 * G->n * 4));
    }

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

  if (verbose && (G->nt != NULL || G->zt != NULL))
    {
      const rt_ntt *t = G->nt != NULL ? G->nt : G->zt;
      printf ("GPU roots: NTT products (d = %u, b = %d, %u blocks) for%s%s\n",
              t->D, t->h.c.b, t->slots, G->nt ? " the roots" : "",
              G->zt ? " the Z/N levels" : "");
    }
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
gpu_roots_stream (gpu_roots *G, void *stream)
{
  G->stream = (cudaStream_t) stream;
}

void
gpu_roots_destroy (gpu_roots *G)
{
  if (G == NULL)
    return;
  if (rtp.on > 0)
    {
      printf ("GPU roots kernels:");
      for (int i = 0; i < RP_N; i++)
        if (rtp.calls[i])
          printf (" %s %.1f ms (%ld)", rp_name[i], rtp.ms[i], rtp.calls[i]);
      printf ("\n");
    }
  rt_set_free (G->F);
  rt_set_free (G->G);
  rt_ntt_destroy (G->nt);
  if (G->zt != G->nt)
    rt_ntt_destroy (G->zt);
  CUDA_CHECK (cudaFree (G->d_N));
  CUDA_CHECK (cudaFree (G->d_one));
  CUDA_CHECK (cudaFree (G->d_R2));
  CUDA_CHECK (cudaFree (G->d_zN));
  CUDA_CHECK (cudaFree (G->d_zR2));
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
  const uint32_t Ts = G->nt != NULL ? 0 : s.pinst * G->tpi;
  for (uint32_t k0 = 0; k0 < G->K; k0 += RT_SLICE)
    {
      uint32_t k1 = std::min (G->K, k0 + RT_SLICE);
      rtp_begin (G->stream);
      if (G->nt != NULL)
        NT_RUN (G, nt_setup_kernel, count, RS_SCALAR, (uint32_t) count,
                (const int32_t *) G->d_cst, G->nt->d_chat, (int32_t *) G->d_W,
                G->K, G->w, G->nw, k0, k1, s.E, s.nr, (uint32_t) G->S,
                s.d_dig, s.d_sgn, (int32_t *) s.d_state, G->nt->d_id,
                G->nt->d_scr);
      else
        G->k->setup<<<rt_blocks (G, count), GPU_TPB, 0, G->stream>>> (
            G->tpi, RS_SCALAR, (uint32_t) count, G->d_cst, G->d_W, G->K, G->w,
            G->nw, k0, k1, s.E, s.nr, G->S, s.d_dig, s.d_sgn, s.d_state, Ts,
            G->d_N,
            G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_SCALAR);
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
          uint32_t *hp = &h[((size_t) cv * RC_COUNT + i) * n];
          if (G->nt != NULL)
            {
              std::vector<int32_t> dg;
              nm_digits (dg, c[i], (int) n, G->nt->h.c.b);
              memcpy (hp, dg.data (), (size_t) n * 4);
            }
          else
            gpu_to_limbs (hp, n, c[i]);
        }
    }
  mpz_clears (g, u, inv, t, NULL);
  for (int i = 0; i < RC_COUNT; i++)
    mpz_clear (c[i]);
  if (found)
    return 1;

  CUDA_CHECK (cudaMemcpyAsync (G->d_cst, h.data (), h.size () * 4,
                               cudaMemcpyHostToDevice, G->stream));
  if (G->nt != NULL)
    {
      const uint32_t cnt = G->C * RH_COUNT;
      rtp_begin (G->stream);
      NT_RUN (G, nt_chat_kernel, cnt, cnt, (const int32_t *) G->d_cst,
              G->nt->d_chat);
      rtp_end (G->stream, RP_CHAIN);
      for (uint32_t k0 = 0; k0 < G->K; k0 += RT_SLICE)
        {
          uint32_t k1 = std::min (G->K, k0 + RT_SLICE);
          rtp_begin (G->stream);
          NT_RUN (G, nt_setup_kernel, G->C, RS_CHAIN, G->C,
                  (const int32_t *) G->d_cst, G->nt->d_chat,
                  (int32_t *) G->d_W, G->K, G->w, G->nw, k0, k1, 0u, 0u, 0u,
                  (const uint8_t *) NULL, (const uint8_t *) NULL,
                  (int32_t *) NULL, G->nt->d_id, G->nt->d_scr);
          rtp_end (G->stream, RP_CHAIN);
        }
      if (G->nw > 1)
        {
          rtp_begin (G->stream);
          NT_RUN (G, nt_setup_kernel, (uint64_t) G->C * G->K, RS_FILL,
                  G->C * G->K, (const int32_t *) G->d_cst, G->nt->d_chat,
                  (int32_t *) G->d_W, G->K, G->w, G->nw, 0u, 0u, 0u, 0u, 0u,
                  (const uint8_t *) NULL, (const uint8_t *) NULL,
                  (int32_t *) NULL, G->nt->d_id, G->nt->d_scr);
          rtp_end (G->stream, RP_FILL);
        }
      rt_start_set (G, G->F);
      rt_start_set (G, G->G);
      G->gblock = 0;
      CUDA_CHECK (cudaStreamSynchronize (G->stream));
      return 0;
    }
  for (uint32_t k0 = 0; k0 < G->K; k0 += RT_SLICE)
    {
      uint32_t k1 = std::min (G->K, k0 + RT_SLICE);
      rtp_begin (G->stream);
      G->k->setup<<<rt_blocks (G, G->C), GPU_TPB, 0, G->stream>>> (
          G->tpi, RS_CHAIN, G->C, G->d_cst, G->d_W, G->K, G->w, G->nw, k0,
          k1, 0, 0, 0, NULL, NULL, NULL, 0, G->d_N, G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_CHAIN);
    }
  if (G->nw > 1)
    {
      rtp_begin (G->stream);
      G->k->setup<<<rt_blocks (G, (uint64_t) G->C * G->K), GPU_TPB, 0, G->stream>>> (
          G->tpi, RS_FILL, G->C * G->K, G->d_cst, G->d_W, G->K, G->w, G->nw,
          0, 0, 0, 0, 0, NULL, NULL, NULL, 0, G->d_N, G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_FILL);
    }
  rt_start_set (G, G->F);
  rt_start_set (G, G->G);
  G->gblock = 0;
  CUDA_CHECK (cudaStreamSynchronize (G->stream));
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
      rtp_begin (G->stream);
      if (G->nt != NULL)
        NT_RUN (G, nt_vec_kernel, c, RV_UP, c, (const int32_t *) lev.back (),
                cnt.back (), (int32_t *) NULL, p, 0u, G->nt->d_scr);
      else
        G->k->vec<<<rt_blocks (G, c), GPU_TPB, 0, G->stream>>> (
            G->tpi, RV_UP, c, lev.back (), cnt.back (), NULL, p, 0, G->d_N,
            G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_UP);
      lev.push_back (p);
      cnt.push_back (c);
      p += c * n;
    }

  std::vector<uint32_t> h (n);
  CUDA_CHECK (cudaMemcpyAsync (h.data (), lev.back (), n * 4,
                               cudaMemcpyDeviceToHost, G->stream));
  CUDA_CHECK (cudaStreamSynchronize (G->stream));
  mpz_t v, t;
  mpz_inits (v, t, NULL);
  if (G->nt != NULL)
    {
      nm_value (v, (const int32_t *) h.data (), (int) n, G->nt->h.c.b);
      mpz_mod (v, v, G->N);
    }
  else
    gpu_from_limbs (v, h.data (), n);
  int found = 0;
  if (!mpz_invert (t, v, G->N))
    {
      /* some root is the identity mod a prime of N: find its curve */
      found = 1;
      mpz_gcd (f, v, G->N);
      *curve = -1;
      std::vector<uint32_t> all ((size_t) M * n);
      CUDA_CHECK (cudaMemcpyAsync (all.data (), G->d_den, all.size () * 4,
                                   cudaMemcpyDeviceToHost, G->stream));
      CUDA_CHECK (cudaStreamSynchronize (G->stream));
      for (uint32_t c = 0; c < G->C && *curve < 0; c++)
        {
          mpz_set_ui (v, 1);
          for (size_t i = 0; i < G->dF; i++)
            {
              if (G->nt != NULL)
                nm_value (t, (const int32_t *) &all[((size_t) c * G->dF + i) * n],
                          (int) n, G->nt->h.c.b);
              else
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
      if (G->nt != NULL)
        {
          std::vector<int32_t> dg;
          nm_digits (dg, t, (int) n, G->nt->h.c.b);
          memcpy (h.data (), dg.data (), (size_t) n * 4);
        }
      else
        gpu_to_limbs (h.data (), n, t);
      CUDA_CHECK (cudaMemcpyAsync (lev.back (), h.data (), n * 4,
                                   cudaMemcpyHostToDevice, G->stream));
      for (size_t l = lev.size () - 1; l > 0; l--)
        {
          rtp_begin (G->stream);
          if (G->nt != NULL)
            NT_RUN (G, nt_vec_kernel, cnt[l], RV_DOWN, cnt[l],
                    (const int32_t *) lev[l], cnt[l - 1], (int32_t *) lev[l - 1],
                    (uint32_t *) NULL, 0u, G->nt->d_scr);
          else
            G->k->vec<<<rt_blocks (G, cnt[l]), GPU_TPB, 0, G->stream>>> (
                G->tpi, RV_DOWN, cnt[l], lev[l], cnt[l - 1], lev[l - 1], NULL,
                0, G->d_N, G->d_one, G->np0);
          CUDA_CHECK (cudaGetLastError ());
          rtp_end (G->stream, RP_DOWN);
        }
      rtp_begin (G->stream);
      if (G->nt != NULL)
        NT_RUN (G, nt_vec_kernel, M, RV_FINAL, M, (const int32_t *) G->d_num,
                M, (int32_t *) G->d_den, out, (uint32_t) G->nl, G->nt->d_scr);
      else
        G->k->vec<<<rt_blocks (G, M), GPU_TPB, 0, G->stream>>> (
            G->tpi, RV_FINAL, M, G->d_num, M, G->d_den, out, G->nl, G->d_N,
            G->d_one, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_FINAL);
      CUDA_CHECK (cudaStreamSynchronize (G->stream));
    }
  mpz_clears (v, t, NULL);
  return found;
}

/* the Z/N levels' scratch, on first use */
static void
rt_ntt_zscr (rt_ntt *nt)
{
  if (nt->d_zscr == NULL)
    CUDA_CHECK (cudaMalloc ((void **) &nt->d_zscr,
                            (size_t) nt->zslots * NT_ZSCR (nt->D) * 4));
}

void
gpu_roots_tree (void *ctx, uint32_t *out, const uint32_t *in, size_t count,
                size_t d)
{
  gpu_roots *G = (gpu_roots *) ctx;
  if (G->zt != NULL && d <= NT_ZMAX)
    {
      rt_ntt_zscr (G->zt);
      NT_RUNC (G->zt, nt_tree_kernel, count / (2 * d), G->zt->zslots,
               (cudaStream_t) 0, (uint64_t) count, (uint32_t) d, in, out,
               (uint32_t) G->nl, G->zt->d_zscr);
      return;
    }
  const uint64_t per = GPU_TPB / G->ztpi;
  G->zk->tree<<<(unsigned) ((count + per - 1) / per), GPU_TPB>>> (
      G->ztpi, count, (uint32_t) d, in, out, G->nl, G->d_zN, G->d_zR2,
      G->np0);
  CUDA_CHECK (cudaGetLastError ());
}

void
gpu_roots_mp (void *ctx, uint32_t *out, const uint32_t *tv,
              const uint32_t *tree, size_t count, size_t half)
{
  gpu_roots *G = (gpu_roots *) ctx;
  if (G->zt != NULL && half <= NT_ZMAX)
    {
      rt_ntt_zscr (G->zt);
      NT_RUNC (G->zt, nt_mp_kernel, count / (2 * half), G->zt->zslots,
               (cudaStream_t) 0, (uint64_t) count, (uint32_t) half, tv, tree,
               out, (uint32_t) G->nl, G->zt->d_zscr);
      return;
    }
  const uint64_t per = GPU_TPB / G->ztpi;
  G->zk->mp<<<(unsigned) ((count + per - 1) / per), GPU_TPB>>> (
      G->ztpi, count, (uint32_t) half, tv, tree, out, G->nl, G->d_zN,
      G->d_zR2, G->np0);
  CUDA_CHECK (cudaGetLastError ());
}

int
gpu_roots_F (gpu_roots *G, uint32_t *out, mpz_t f, int *curve)
{
  rt_set &s = G->F;
  const uint64_t ninst = (uint64_t) G->C * s.nr;
  for (uint32_t i0 = 0; i0 < s.steps; i0 += RT_SLICE)
    {
      uint32_t i1 = std::min (s.steps, i0 + RT_SLICE);
      rtp_begin (G->stream);
      if (G->nt != NULL)
        NT_RUN (G, nt_step_kernel, ninst, (uint32_t) ninst, s.nr,
                (uint32_t) G->S, (int32_t *) s.d_state, G->nt->d_chat,
                s.d_slots, (uint64_t) 0, (uint32_t) G->dF, i0, i1,
                (int32_t *) G->d_num, (int32_t *) G->d_den, G->nt->d_scr);
      else
        G->k->step<<<rt_blocks (G, ninst), GPU_TPB, 0, G->stream>>> (
            G->tpi, (uint32_t) ninst, s.nr, G->S, s.d_state, G->d_cst,
            s.d_slots, 0, (uint32_t) G->dF, i0, i1, G->d_num, G->d_den,
            G->d_N, G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_STEP);
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
      rtp_begin (G->stream);
      if (G->nt != NULL)
        NT_RUN (G, nt_step_kernel, ninst, (uint32_t) ninst, s.nr,
                (uint32_t) G->S, (int32_t *) s.d_state, G->nt->d_chat,
                (const int32_t *) NULL, base, (uint32_t) G->dF, i0, i1,
                (int32_t *) G->d_num, (int32_t *) G->d_den, G->nt->d_scr);
      else
        G->k->step<<<rt_blocks (G, ninst), GPU_TPB, 0, G->stream>>> (
            G->tpi, (uint32_t) ninst, s.nr, G->S, s.d_state, G->d_cst, NULL,
            base, (uint32_t) G->dF, i0, i1, G->d_num, G->d_den, G->d_N,
            G->np0);
      CUDA_CHECK (cudaGetLastError ());
      rtp_end (G->stream, RP_STEP);
    }
  return rt_finish (G, out, f, curve);
}
