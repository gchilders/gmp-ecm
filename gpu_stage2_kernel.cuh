/* gpu_stage2_kernel.cuh: GPU ECM stage 2 kernels (baby-step giant-step).

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

/* Instantiated per (L, SOLO) by the gpu_kset_*.cu files; see gpu_stage2.cu
   for the method and the memory layout. */

#ifndef _GPU_STAGE2_KERNEL_CUH
#define _GPU_STAGE2_KERNEL_CUH 1

#include "gpu_common.cuh"
#include "gpu_mont.cuh"

/* state slots.  AUX holds 2Q during setup and the running product of the
   Z's while the baby steps are put on a shared denominator; BZ holds that
   denominator. */
enum { S_PREV_X, S_PREV_Z, S_CUR_X, S_CUR_Z, S_DQ_X, S_DQ_Z, S_ACC, S_BZ,
       S_AUX_X, S_AUX_Z, S_COUNT };

template<int L>
__device__ __forceinline__ void
s2_load (uint32_t (&x)[L], const uint32_t * __restrict__ buf, uint32_t s,
         uint32_t T, uint32_t t)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    x[k] = buf[((size_t) s * L + k) * T + t];
}

template<int L>
__device__ __forceinline__ void
s2_store (uint32_t * __restrict__ buf, const uint32_t (&x)[L], uint32_t s,
          uint32_t T, uint32_t t)
{
#pragma unroll
  for (int k = 0; k < L; k++)
    buf[((size_t) s * L + k) * T + t] = x[k];
}

template<int L>
__device__ __forceinline__ void
s2_copy (uint32_t (&r)[L], const uint32_t (&a)[L])
{
#pragma unroll
  for (int k = 0; k < L; k++)
    r[k] = a[k];
}

/* (X : Z) = 2 (X1 : Z1).  Inputs < 2N, outputs < 2N.  May alias. */
template<int L, bool SOLO>
__device__ __forceinline__ void
xdbl (const gm_group &g, uint32_t (&X)[L], uint32_t (&Z)[L],
      const uint32_t (&X1)[L], const uint32_t (&Z1)[L], uint32_t d,
      const uint32_t (&N)[L], const uint32_t (&N2)[L], uint32_t np0)
{
  uint32_t t1[L], t2[L], t3[L];
  gm_add<L, SOLO> (g, t1, X1, Z1);              /* < 4N */
  gm_sub<L, SOLO> (g, t2, X1, Z1, N2);          /* < 2N */
  gm_sqr<L, SOLO> (g, t1, t1, N, np0);          /* A = (X+Z)^2 */
  gm_sqr<L, SOLO> (g, t2, t2, N, np0);          /* B = (X-Z)^2 */
  gm_sub<L, SOLO> (g, t3, t1, t2, N2);          /* C = A - B */
  gm_mul<L, SOLO> (g, X, t1, t2, N, np0);       /* X = A B */
  gm_mul_ui<L, SOLO> (g, t1, t3, d, N, np0);    /* a24 C < 3N */
  gm_add<L, SOLO> (g, t1, t1, t2);              /* B + a24 C < 5N */
  gm_mul<L, SOLO> (g, Z, t3, t1, N, np0);       /* Z = C (B + a24 C) */
}

/* (X : Z) = P + Q given D = P - Q.  Inputs < 2N, outputs < 2N.  The output
   may alias P or Q but not D. */
template<int L, bool SOLO>
__device__ __forceinline__ void
xadd (const gm_group &g, uint32_t (&X)[L], uint32_t (&Z)[L],
      const uint32_t (&XP)[L], const uint32_t (&ZP)[L],
      const uint32_t (&XQ)[L], const uint32_t (&ZQ)[L],
      const uint32_t (&XD)[L], const uint32_t (&ZD)[L],
      const uint32_t (&N)[L], const uint32_t (&N2)[L], uint32_t np0)
{
  uint32_t t1[L], t2[L], t3[L];
  gm_sub<L, SOLO> (g, t1, XP, ZP, N2);
  gm_add<L, SOLO> (g, t2, XQ, ZQ);
  gm_mul<L, SOLO> (g, t1, t1, t2, N, np0);      /* U = (XP-ZP)(XQ+ZQ) */
  gm_add<L, SOLO> (g, t2, XP, ZP);
  gm_sub<L, SOLO> (g, t3, XQ, ZQ, N2);
  gm_mul<L, SOLO> (g, t2, t2, t3, N, np0);      /* V = (XP+ZP)(XQ-ZQ) */
  gm_add<L, SOLO> (g, t3, t1, t2);              /* U + V < 4N */
  gm_sub<L, SOLO> (g, t1, t1, t2, N2);          /* U - V < 2N */
  gm_sqr<L, SOLO> (g, t3, t3, N, np0);
  gm_sqr<L, SOLO> (g, t1, t1, N, np0);
  gm_mul<L, SOLO> (g, X, ZD, t3, N, np0);
  gm_mul<L, SOLO> (g, Z, XD, t1, N, np0);
}

/* The baby-step table has two slots per baby step b: X at slot b, Z at
 * slot nbaby + b.  s2_norm then puts the baby steps on one shared
 * denominator, as cuda-sieve's ECM does: X_b becomes
 * bx_b = X_b prod_{c != b} Z_c, and BZ = prod_c Z_c, so (bx_b : BZ) is
 * still the point jQ.  A term of giant step G is then
 *   X(G) BZ - bx_b Z(G) = (X(G) Z_b - X_b Z(G)) prod_{c != b} Z_c,
 * with X(G) BZ shared by all the baby steps: still two products per pair,
 * and no inversion.  The extra factor is a unit unless some Z_c shares a
 * factor q with N (c Q = O mod q); then every other term is 0 mod q, so
 * the gcd is a multiple of q, and the host replays a curve whose gcd is N.
 *
 * The one-off kernels below each do a slice of their work per launch, with
 * the rest of their state in STATE, so that the host can keep every launch
 * near 100 ms (a display GPU must not stall for long, and Windows resets
 * one that runs a kernel for more than 2 s). */

/* Build the baby-step table and DQ, and set up the giant steps at
 * G_1 = DQ, G_2 = 2 DQ (the host starts the main kernel at k = 2, with
 * empty masks up to the first giant step it needs; that costs one xadd per
 * skipped step, about B2min / D, which is small next to the B2 / D steps).
 * The work is one loop with a single xdbl and a single xadd call site:
 * inlining several of each, for every kernel shape, made compiles slow.
 *
 * Steps: 0: Q2 = 2Q; 1 .. nodd-1: the next odd multiple jQ = (2 step + 1) Q
 * (j <= D/2) by xadd, stored if gcd (j, D) = 1; then nodd: DQ = 2 (D/2)Q;
 * nodd + 1: G_2 = 2 DQ.  This launch does steps [STEP0, STEP1); NEXT0 is the
 * index in JLIST of the first baby step not yet stored (the host counts it).
 * Between launches, cur = jQ, prev = (j-2)Q, Q2 and DQ live in STATE. */
template<int L, bool SOLO>
__global__ void __launch_bounds__ (GPU_TPB, 1)
s2_setup_kernel (const uint32_t * __restrict__ xin,
                 const uint32_t * __restrict__ Ng,
                 const uint32_t * __restrict__ oneg,
                 uint32_t np0, uint32_t sigma0, uint32_t tpi,
                 uint32_t halfD, const uint32_t * __restrict__ jlist,
                 uint32_t nbaby,
                 uint32_t * __restrict__ baby, uint32_t * __restrict__ state,
                 uint32_t step0, uint32_t step1, uint32_t next0,
                 const uint32_t * __restrict__ dlist)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t T = gridDim.x * blockDim.x;
  const uint32_t inst = t / g.tpi;
  const uint32_t n = g.tpi * L;
  const uint32_t off = g.lane * L;
  /* curve i has d = sigma0 + i, or dlist[i] (a resumed batch) */
  const uint32_t d = dlist != NULL ? dlist[inst] : sigma0 + inst;
  const uint32_t nodd = (halfD + 1) / 2;   /* odd j in [1, D/2] */

  uint32_t N[L], N2[L];
#pragma unroll
  for (int k = 0; k < L; k++)
    N[k] = Ng[off + k];
  gm_dbl<L, SOLO> (g, N2, N);

  uint32_t Q2X[L], Q2Z[L], cX[L], cZ[L], pX[L], pZ[L], DX[L], DZ[L];
  uint32_t next = next0;
  if (step0 == 0)
    {
      /* cur = Q, prv = -Q (x(-Q) = x(Q)): j = 1 */
#pragma unroll
      for (int k = 0; k < L; k++)
        {
          cZ[k] = oneg[off + k];
          cX[k] = xin[(size_t) inst * n + off + k];
        }
      s2_copy<L> (pX, cX);
      s2_copy<L> (pZ, cZ);
      s2_store<L> (baby, cX, 0, T, t);
      s2_store<L> (baby, cZ, nbaby, T, t);
      next = 1;
    }
  else
    {
      s2_load<L> (cX, state, S_CUR_X, T, t);
      s2_load<L> (cZ, state, S_CUR_Z, T, t);
      s2_load<L> (pX, state, S_PREV_X, T, t);
      s2_load<L> (pZ, state, S_PREV_Z, T, t);
      s2_load<L> (Q2X, state, S_AUX_X, T, t);
      s2_load<L> (Q2Z, state, S_AUX_Z, T, t);
      s2_load<L> (DX, state, S_DQ_X, T, t);
      s2_load<L> (DZ, state, S_DQ_Z, T, t);
    }

  for (uint32_t step = step0; step < step1; step++)
    {
      if (step == 0 || step >= nodd)
        {
          /* doublings: Q2 = 2Q, DQ = 2 cur, G2 = 2 DQ (into cur); at
             step 0, cur is Q */
          uint32_t sX[L], sZ[L], rX[L], rZ[L];
          if (step <= nodd)
            {
              s2_copy<L> (sX, cX);
              s2_copy<L> (sZ, cZ);
            }
          else
            {
              s2_copy<L> (sX, DX);
              s2_copy<L> (sZ, DZ);
            }
          xdbl<L, SOLO> (g, rX, rZ, sX, sZ, d, N, N2, np0);
          if (step == 0)
            {
              s2_copy<L> (Q2X, rX);
              s2_copy<L> (Q2Z, rZ);
            }
          else if (step == nodd)
            {
              s2_copy<L> (DX, rX);
              s2_copy<L> (DZ, rZ);
            }
          else
            {
              s2_copy<L> (cX, rX);
              s2_copy<L> (cZ, rZ);
            }
        }
      else
        {
          uint32_t nX[L], nZ[L];
          xadd<L, SOLO> (g, nX, nZ, cX, cZ, Q2X, Q2Z, pX, pZ, N, N2, np0);
          s2_copy<L> (pX, cX);
          s2_copy<L> (pZ, cZ);
          s2_copy<L> (cX, nX);
          s2_copy<L> (cZ, nZ);
          if (next < nbaby && jlist[next] == 2 * step + 1)
            {
              s2_store<L> (baby, cX, next, T, t);
              s2_store<L> (baby, cZ, nbaby + next, T, t);
              next++;
            }
        }
    }

  if (step1 == nodd + 2)
    {
      /* done: prev = G_1 = DQ, cur = G_2, acc = 1 */
      s2_copy<L> (pX, DX);
      s2_copy<L> (pZ, DZ);
#pragma unroll
      for (int k = 0; k < L; k++)
        Q2X[k] = oneg[off + k];
      s2_store<L> (state, Q2X, S_ACC, T, t);
    }
  s2_store<L> (state, cX, S_CUR_X, T, t);
  s2_store<L> (state, cZ, S_CUR_Z, T, t);
  s2_store<L> (state, pX, S_PREV_X, T, t);
  s2_store<L> (state, pZ, S_PREV_Z, T, t);
  s2_store<L> (state, Q2X, S_AUX_X, T, t);
  s2_store<L> (state, Q2Z, S_AUX_Z, T, t);
  s2_store<L> (state, DX, S_DQ_X, T, t);
  s2_store<L> (state, DZ, S_DQ_Z, T, t);
}

/* Put the baby steps on the shared denominator, iterations [I0, I1) of
 * 2 nbaby.  Forward, i = b < nbaby: bx_b = X_b prod_{c<b} Z_c, with the
 * running product in AUX_X; it ends at BZ = prod Z_c.  Backward,
 * i = 2 nbaby - 1 - b: bx_b *= prod_{c>b} Z_c.  The products by 1 at the
 * start of each pass, and the backward pass's last update of the running
 * product, are skipped. */
template<int L, bool SOLO>
__global__ void __launch_bounds__ (GPU_TPB, 1)
s2_norm_kernel (uint32_t * __restrict__ baby, uint32_t nbaby,
                uint32_t * __restrict__ state,
                const uint32_t * __restrict__ Ng, uint32_t np0,
                uint32_t tpi, uint32_t i0, uint32_t i1)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t T = gridDim.x * blockDim.x;
  const uint32_t off = g.lane * L;

  uint32_t N[L], acc[L], x[L], z[L];
#pragma unroll
  for (int k = 0; k < L; k++)
    N[k] = Ng[off + k];
  s2_load<L> (acc, state, S_AUX_X, T, t);
  for (uint32_t i = i0; i < i1; i++)
    {
      const bool fwd = i < nbaby;
      const uint32_t b = fwd ? i : 2 * nbaby - 1 - i;
      const bool first = i == 0 || i == nbaby;
      if (i == nbaby)
        s2_store<L> (state, acc, S_BZ, T, t);
      s2_load<L> (z, baby, nbaby + b, T, t);
      if (!first)
        {
          s2_load<L> (x, baby, b, T, t);
          gm_mul<L, SOLO> (g, x, x, acc, N, np0);
          s2_store<L> (baby, x, b, T, t);
        }
      if (first)
        s2_copy<L> (acc, z);
      else if (fwd || b > 0)
        gm_mul<L, SOLO> (g, acc, acc, z, N, np0);
    }
  s2_store<L> (state, acc, S_AUX_X, T, t);
}

/* Process COUNT giant steps: for each, multiply in the terms of the baby
 * steps set in its mask (MW words), then advance.  Only the current giant
 * step G_k and X(G_k) BZ live in registers; the accumulator, BZ, DQ and
 * G_{k-1} are read from STATE when needed (a few loads per term next to two
 * products), which keeps register use near 10L instead of 16L and avoids
 * spills. */
template<int L, bool SOLO>
struct s2_main_occupancy {
  /* Minimum resident blocks per SM asked of the compiler.  At these shapes
     the register for X(G) BZ otherwise tips the compiler into halving the
     occupancy, which cost 4-13% (A10, sm_86); asking for 3 or 4 blocks
     gets back the speed of the kernel without it. */
  enum { blocks = (!SOLO && L == 7) ? 3 : (!SOLO && L == 8) ? 4 : 1 };
};

template<int L, bool SOLO>
__global__ void
__launch_bounds__ (GPU_TPB, s2_main_occupancy<L, SOLO>::blocks)
s2_main_kernel (const uint32_t * __restrict__ masks, uint32_t count,
                uint32_t mw, const uint32_t * __restrict__ baby,
                uint32_t * __restrict__ state,
                const uint32_t * __restrict__ Ng, uint32_t np0, uint32_t tpi)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t T = gridDim.x * blockDim.x;
  const uint32_t off = g.lane * L;

  uint32_t N[L], N2[L];
#pragma unroll
  for (int k = 0; k < L; k++)
    N[k] = Ng[off + k];
  gm_dbl<L, SOLO> (g, N2, N);

  uint32_t cX[L], cZ[L];
  s2_load<L> (cX, state, S_CUR_X, T, t);
  s2_load<L> (cZ, state, S_CUR_Z, T, t);

  for (uint32_t c = 0; c < count; c++)
    {
      /* X(G) BZ, shared by the giant step's terms; computing it even when
         the mask is empty (rare for D = 2310) keeps register use lower */
      uint32_t gx[L];
      s2_load<L> (gx, state, S_BZ, T, t);
      gm_mul<L, SOLO> (g, gx, gx, cX, N, np0);
      for (uint32_t w = 0; w < mw; w++)
        {
          uint32_t bits = masks[c * mw + w];
          while (bits != 0)
            {
              uint32_t bi = w * 32 + __ffs (bits) - 1;
              bits &= bits - 1;
              /* acc *= X(G) BZ - bx_j Z(G) */
              uint32_t b[L], u[L];
              s2_load<L> (b, baby, bi, T, t);
              gm_mul<L, SOLO> (g, b, b, cZ, N, np0);
              gm_sub<L, SOLO> (g, u, gx, b, N2);
              s2_load<L> (b, state, S_ACC, T, t);
              gm_mul<L, SOLO> (g, b, b, u, N, np0);
              s2_store<L> (state, b, S_ACC, T, t);
            }
        }

      /* G_{k+1} = G_k + DQ, difference G_{k-1} (xadd, with operands
         loaded as needed) */
      uint32_t a[L], b[L], u[L];
      s2_load<L> (u, state, S_DQ_X, T, t);
      s2_load<L> (b, state, S_DQ_Z, T, t);
      gm_add<L, SOLO> (g, a, u, b);              /* XQ + ZQ */
      gm_sub<L, SOLO> (g, u, u, b, N2);          /* XQ - ZQ */
      gm_sub<L, SOLO> (g, b, cX, cZ, N2);        /* XP - ZP */
      gm_mul<L, SOLO> (g, a, b, a, N, np0);      /* U */
      gm_add<L, SOLO> (g, b, cX, cZ);            /* XP + ZP */
      gm_mul<L, SOLO> (g, u, b, u, N, np0);      /* V */
      gm_add<L, SOLO> (g, b, a, u);              /* U + V */
      gm_sub<L, SOLO> (g, a, a, u, N2);          /* U - V */
      gm_sqr<L, SOLO> (g, b, b, N, np0);
      gm_sqr<L, SOLO> (g, a, a, N, np0);
      s2_load<L> (u, state, S_PREV_X, T, t);
      gm_mul<L, SOLO> (g, a, u, a, N, np0);      /* Z' = X(G_{k-1}) (U-V)^2 */
      s2_load<L> (u, state, S_PREV_Z, T, t);
      gm_mul<L, SOLO> (g, b, u, b, N, np0);      /* X' = Z(G_{k-1}) (U+V)^2 */
      s2_store<L> (state, cX, S_PREV_X, T, t);
      s2_store<L> (state, cZ, S_PREV_Z, T, t);
      s2_copy<L> (cX, b);
      s2_copy<L> (cZ, a);
    }

  s2_store<L> (state, cX, S_CUR_X, T, t);
  s2_store<L> (state, cZ, S_CUR_Z, T, t);
}

#endif /* _GPU_STAGE2_KERNEL_CUH */
