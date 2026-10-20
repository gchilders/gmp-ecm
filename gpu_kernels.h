/* gpu_kernels.h: registry of the GPU ECM kernels, one set per shape.

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

/* The kernels are templates on L (limbs per thread) and SOLO (one thread
   per curve).  Their instantiations are spread over the gpu_kset_*.cu files
   so that they compile in parallel; each file fills its part of the table
   below. */

#ifndef _GPU_KERNELS_H
#define _GPU_KERNELS_H 1

#include <stdint.h>
#include "gpu_common.cuh"

typedef void (*gpu_s1_fn) (const uint32_t *, uint64_t, uint64_t, uint64_t,
                           uint32_t *, const uint32_t *,
                           uint32_t, uint32_t, uint32_t,
                           const uint32_t *, const uint32_t *);
typedef void (*gpu_s2_setup_fn) (const uint32_t *, const uint32_t *,
                                 const uint32_t *, uint32_t, uint32_t,
                                 uint32_t, uint32_t, const uint32_t *,
                                 uint32_t, uint32_t *, uint32_t *,
                                 uint32_t, uint32_t, uint32_t,
                                 const uint32_t *);
typedef void (*gpu_s2_main_fn) (const uint32_t *, uint32_t, uint32_t,
                                const uint32_t *, uint32_t *,
                                const uint32_t *, uint32_t, uint32_t);
typedef void (*gpu_s2_norm_fn) (uint32_t *, uint32_t, uint32_t *,
                                const uint32_t *, uint32_t, uint32_t,
                                uint32_t, uint32_t);

struct gpu_kernel_set {
  gpu_s1_fn s1;
  gpu_s2_setup_fn s2_setup;
  gpu_s2_norm_fn s2_norm;
  gpu_s2_main_fn s2_main;
};

struct gpu_kernel_table {
  gpu_kernel_set solo[GPU_SOLO_MAX_L + 1];
  gpu_kernel_set dist[GPU_DIST_MAX_L + 1];
};

/* Kernels for TPI threads per curve (1 = solo) and L limbs per thread, or
   NULL if not compiled. */
const gpu_kernel_set *gpu_kernels (uint32_t tpi, uint32_t L);

#endif /* _GPU_KERNELS_H */
