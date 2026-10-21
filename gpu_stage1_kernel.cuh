/* gpu_stage1_kernel.cuh: GPU ECM stage 1 kernel (Montgomery ladder).

Copyright 2021 Seth Troisi, 2026 Greg Childers

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

/* Instantiated per (L, SOLO) by the gpu_kset_*.cu files; see gpu_stage1.cu
   for the bounds. */

#ifndef _GPU_STAGE1_KERNEL_CUH
#define _GPU_STAGE1_KERNEL_CUH 1

#include "gpu_common.cuh"
#include "gpu_mont.cuh"

template<int L, bool SOLO, class G>
__device__ __forceinline__ void
ladder_step (const G &g,
             uint32_t (&aX)[L], uint32_t (&aZ)[L],
             uint32_t (&bX)[L], uint32_t (&bZ)[L],
             uint32_t d, const uint32_t (&N)[L], const uint32_t (&N2)[L],
             uint32_t np0, const uint32_t * __restrict__ xd)
{
  uint32_t t1[L], t2[L];

  gm_add<L, SOLO> (g, t1, bZ, bX);
  gm_sub<L, SOLO> (g, bZ, bZ, bX, N2);
  gm_add<L, SOLO> (g, bX, aZ, aX);
  gm_sub<L, SOLO> (g, aZ, aZ, aX, N2);
  gm_mul<L, SOLO> (g, aX, t1, aZ, N, np0);     /* aX = CB */
  gm_mul<L, SOLO> (g, t1, bZ, bX, N, np0);     /* t1 = DA */
  gm_sqr<L, SOLO> (g, bZ, bX, N, np0);         /* bZ = AA */
  gm_sqr<L, SOLO> (g, bX, aZ, N, np0);         /* bX = BB */
  gm_sub<L, SOLO> (g, aZ, bZ, bX, N2);         /* aZ = K */
  gm_mul<L, SOLO> (g, bZ, bZ, bX, N, np0);     /* bZ = AA*BB = new aX */
  gm_mul_ui<L, SOLO> (g, t2, aZ, d, N, np0);   /* t2 = dK */
  gm_add<L, SOLO> (g, bX, bX, t2);             /* bX = BB + dK */
  gm_mul<L, SOLO> (g, aZ, aZ, bX, N, np0);     /* aZ = new aZ */
  gm_add<L, SOLO> (g, bX, t1, aX);             /* bX = DA + CB */
  gm_sub<L, SOLO> (g, t1, t1, aX, N2);         /* t1 = DA - CB */
  gm_sqr<L, SOLO> (g, bX, bX, N, np0);         /* bX = new bX */
  gm_sqr<L, SOLO> (g, t1, t1, N, np0);
  if (xd == NULL)
    gm_dbl<L, SOLO> (g, t1, t1);               /* t1 = new bZ: x(P) = 2 */
  else
    {
      /* a resumed curve: the difference P - Q is the starting point
         (x0 : 1), whose x0 (< N, this thread's limbs) is read here rather
         than held in registers through the whole ladder */
#pragma unroll
      for (int k = 0; k < L; k++)
        t2[k] = xd[k];
      gm_mul<L, SOLO> (g, t1, t1, t2, N, np0);   /* t1 = new bZ < 2N */
    }

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      aX[k] = bZ[k];
      bZ[k] = t1[k];
    }
}

template<int L>
__device__ __forceinline__ void
swap_limbs (uint32_t (&a)[L], uint32_t (&b)[L])
{
#pragma unroll
  for (int k = 0; k < L; k++)
    {
      uint32_t t = a[k];
      a[k] = b[k];
      b[k] = t;
    }
}

/* Process bits [b_start, b_start + b_count) of s, most significant first.
 * data holds, per curve, aX, aZ, bX, bZ (n limbs each, Montgomery form).
 * Curve i has d = sigma0 + i, or dlist[i] when dlist is not NULL.  The
 * ladder runs from (P, 2P) with P = (2 : 1), or with P = (x0 : 1) for a
 * resumed curve when x0g (n limbs per curve, Montgomery form) is given.
 * Every thread of the grid owns a real (or padding) curve: no early exits,
 * the shuffles and barriers need whole groups. */
template<int L, bool SOLO, class G>
__device__ __forceinline__ void
gpu_stage1_body (const G &g, const uint32_t * __restrict__ s_bits,
                 uint64_t s_nbits, uint64_t b_start, uint64_t b_count,
                 uint32_t * __restrict__ data,
                 const uint32_t * __restrict__ Ng,
                 uint32_t np0, uint32_t sigma0,
                 const uint32_t * __restrict__ dlist,
                 const uint32_t * __restrict__ x0g)
{
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / g.tpi;
  const uint32_t n = g.tpi * L;
  const uint32_t off = g.lane * L;

  uint32_t N[L], N2[L], aX[L], aZ[L], bX[L], bZ[L];
  uint32_t *dp = data + (size_t) inst * 4 * n + off;

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      N[k] = Ng[off + k];
      aX[k] = dp[k];
      aZ[k] = dp[n + k];
      bX[k] = dp[2 * n + k];
      bZ[k] = dp[3 * n + k];
    }
  gm_dbl<L, SOLO> (g, N2, N);

  const uint32_t d = dlist != NULL ? dlist[inst] : sigma0 + inst;
  const uint32_t *xd = x0g != NULL ? x0g + (size_t) inst * n + off : NULL;

  int swapped = 0;
  for (uint64_t b = b_start; b < b_start + b_count; b++)
    {
      uint64_t nth = s_nbits - 1 - b;
      int bit = (s_bits[nth >> 5] >> (nth & 31)) & 1;
      if (bit != swapped)
        {
          swapped = bit;
          swap_limbs<L> (aX, bX);
          swap_limbs<L> (aZ, bZ);
        }
      ladder_step<L, SOLO> (g, aX, aZ, bX, bZ, d, N, N2, np0, xd);
    }
  if (swapped)
    {
      swap_limbs<L> (aX, bX);
      swap_limbs<L> (aZ, bZ);
    }

#pragma unroll
  for (int k = 0; k < L; k++)
    {
      dp[k] = aX[k];
      dp[n + k] = aZ[k];
      dp[2 * n + k] = bX[k];
      dp[3 * n + k] = bZ[k];
    }
}

/* TPI = 1 (SOLO) or 2..32 threads per curve, within a warp */
template<int L, bool SOLO>
__global__ void __launch_bounds__ (GPU_TPB)
gpu_stage1_kernel (const uint32_t * __restrict__ s_bits, uint64_t s_nbits,
                   uint64_t b_start, uint64_t b_count,
                   uint32_t * __restrict__ data,
                   const uint32_t * __restrict__ Ng,
                   uint32_t np0, uint32_t sigma0, uint32_t tpi,
                   const uint32_t * __restrict__ dlist,
                   const uint32_t * __restrict__ x0g)
{
  gm_group g;
  g.init (SOLO ? 1 : tpi);
  gpu_stage1_body<L, SOLO> (g, s_bits, s_nbits, b_start, b_count, data, Ng,
                            np0, sigma0, dlist, x0g);
}

/* TPI = 64 or 128 threads per curve (gm_wgroup) */
template<int L>
__global__ void __launch_bounds__ (GPU_TPB)
gpu_stage1_wide_kernel (const uint32_t * __restrict__ s_bits, uint64_t s_nbits,
                        uint64_t b_start, uint64_t b_count,
                        uint32_t * __restrict__ data,
                        const uint32_t * __restrict__ Ng,
                        uint32_t np0, uint32_t sigma0, uint32_t tpi,
                        const uint32_t * __restrict__ dlist,
                        const uint32_t * __restrict__ x0g)
{
  __shared__ uint32_t smem[GM_WSHARED_WORDS (GPU_TPB, L)];
  gm_wgroup g;
  g.init (tpi, smem, L);
  gpu_stage1_body<L, false> (g, s_bits, s_nbits, b_start, b_count, data, Ng,
                             np0, sigma0, dlist, x0g);
}

#endif /* _GPU_STAGE1_KERNEL_CUH */
