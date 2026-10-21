/* gpu_kset.cuh: instantiate the GPU ECM kernels for one range of L.

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

/* Included by gpu_kset_NN.cu after defining
     GPU_KSET          the set number NN
     GPU_KSET_SOLO_LO, GPU_KSET_SOLO_HI   solo L range (empty if LO > HI)
     GPU_KSET_DIST_LO, GPU_KSET_DIST_HI   distributed L range              */

#include "gpu_kernels.h"
#include "gpu_stage1_kernel.cuh"
#include "gpu_stage2_kernel.cuh"

template<int L, int HI, bool SOLO, bool DONE = (L > HI)>
struct gpu_kset_fill {
  static void run (gpu_kernel_set *t)
  {
    t[L].s1 = gpu_stage1_kernel<L, SOLO>;
    t[L].s2_setup = s2_setup_kernel<L, SOLO>;
    t[L].s2_norm = s2_norm_kernel<L, SOLO>;
    t[L].s2_main = s2_main_kernel<L, SOLO>;
    gpu_kset_fill<L + 1, HI, SOLO>::run (t);
  }
};
template<int L, int HI, bool SOLO>
struct gpu_kset_fill<L, HI, SOLO, true> {
  static void run (gpu_kernel_set *) {}
};

/* wide groups: stage 1 for L >= 2, stage 2 for L >= 7 */
template<int L, int HI, bool DONE = (L > HI)>
struct gpu_kset_fill_wide {
  static void run (gpu_kernel_set *t)
  {
    if constexpr (L >= 2)
      t[L].s1 = gpu_stage1_wide_kernel<L>;
    /* stage 2 only for the shapes chosen above the one-warp sizes:
       64 threads with L >= 13, 128 with L >= 7 */
    if constexpr (L >= 7)
      {
        t[L].s2_setup = s2_setup_wide_kernel<L>;
        t[L].s2_norm = s2_norm_wide_kernel<L>;
        t[L].s2_main = s2_main_wide_kernel<L>;
      }
    gpu_kset_fill_wide<L + 1, HI>::run (t);
  }
};
template<int L, int HI>
struct gpu_kset_fill_wide<L, HI, true> {
  static void run (gpu_kernel_set *) {}
};

#define GPU_KSET_NAME2(n) gpu_kset_ ## n
#define GPU_KSET_NAME(n) GPU_KSET_NAME2 (n)

void GPU_KSET_NAME (GPU_KSET) (gpu_kernel_table *t)
{
#if GPU_KSET_SOLO_LO <= GPU_KSET_SOLO_HI
  gpu_kset_fill<GPU_KSET_SOLO_LO, GPU_KSET_SOLO_HI, true>::run (t->solo);
#endif
#if GPU_KSET_DIST_LO <= GPU_KSET_DIST_HI
  gpu_kset_fill<GPU_KSET_DIST_LO, GPU_KSET_DIST_HI, false>::run (t->dist);
  gpu_kset_fill_wide<GPU_KSET_DIST_LO, GPU_KSET_DIST_HI>::run (t->wide);
#endif
}
