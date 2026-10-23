/* gpu_kset_19.cu: GPU ECM kernels for solo L 27..28 (even only), no
   distributed L (see gpu_kset.cuh).  Copyright 2026 Greg Childers; GPL v3
   or later. */
#define GPU_KSET 19
#define GPU_KSET_SOLO_LO 27
#define GPU_KSET_SOLO_HI 28
#define GPU_KSET_DIST_LO 1
#define GPU_KSET_DIST_HI 0
#include "gpu_kset.cuh"
