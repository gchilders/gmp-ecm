/* gpu_kset_14.cu: GPU ECM kernels for solo L 17..18 (even only), no
   distributed L (see gpu_kset.cuh).  Copyright 2026 Greg Childers; GPL v3
   or later. */
#define GPU_KSET 14
#define GPU_KSET_SOLO_LO 17
#define GPU_KSET_SOLO_HI 18
#define GPU_KSET_DIST_LO 1
#define GPU_KSET_DIST_HI 0
#include "gpu_kset.cuh"
