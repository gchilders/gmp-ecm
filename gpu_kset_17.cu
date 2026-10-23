/* gpu_kset_17.cu: GPU ECM kernels for solo L 23..24 (even only), no
   distributed L (see gpu_kset.cuh).  Copyright 2026 Greg Childers; GPL v3
   or later. */
#define GPU_KSET 17
#define GPU_KSET_SOLO_LO 23
#define GPU_KSET_SOLO_HI 24
#define GPU_KSET_DIST_LO 1
#define GPU_KSET_DIST_HI 0
#include "gpu_kset.cuh"
